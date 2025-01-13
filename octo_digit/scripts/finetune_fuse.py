import datetime
from functools import partial
import os

from jax.experimental import multihost_utils
from absl import app, flags, logging
import flax
from flax.traverse_util import flatten_dict
import jax
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from ml_collections import config_flags, ConfigDict
import optax
import tensorflow as tf
import tqdm
import wandb

from octo.utils.typing import Data
from octo.utils import jax_utils
from octo.data.dataset import make_single_dataset
from octo.utils.jax_utils import initialize_compilation_cache
from octo.utils.spec import ModuleSpec
from octo.utils.train_callbacks import (
    SaveCallback,
    ValidationCallback,
    LanguageCallback,
)
from octo.utils.logging_utils import print_separator, pretty_print_dict, append_identity_to_metrics
from octo.utils.train_utils import (
    check_config_diff,
    create_optimizer,
    format_name_with_config,
    merge_params,
    process_text_fuse,
    Timer,
    TrainState,
)
from octo.utils.fuse_utils import (
    fuse_loss_fn_contrastive,
    fuse_loss_fn_generative,
    fuse_decode_ids,
)
from octo.model.octo_model import OctoModel
from octo.model.octo_module import OctoModule
try:
    from jax_smi import initialise_tracking  # type: ignore
    initialise_tracking()
except ImportError:
    pass

FLAGS = flags.FLAGS

flags.DEFINE_string("name", "experiment", "Experiment name.")
flags.DEFINE_bool("debug", False, "Debug config (no wandb logging)")

default_config_file = os.path.join(
    os.path.dirname(__file__), "configs/finetune_fuse.py"
)
config_flags.DEFINE_config_file(
    "config",
    default_config_file,
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)

def main(_):
    initialize_compilation_cache()
    devices = jax.devices()
    assert FLAGS.config.batch_size % jax.device_count() == 0, (
        f"Batch size {FLAGS.config.batch_size} must be divisible by number of devices {jax.device_count()}")
    assert FLAGS.config.batch_size % jax.process_count() == 0, (
        f"Batch size {FLAGS.config.batch_size} must be divisible by number of hosts {jax.process_count()}")
    
    logging.info(
        f"""
        Octo Finetuning Script
        ======================
        Pretrained model: {FLAGS.config.pretrained_path}
        Finetuning Dataset: {FLAGS.config.dataset_kwargs.name}
        Data dir: {FLAGS.config.dataset_kwargs.data_dir}
        Task Modality: {FLAGS.config.modality}
        Finetuning Mode: {FLAGS.config.finetuning_mode}
        
        # Workers: {jax.process_count()}
        # Devices: {jax.device_count()}
        Batch size: {FLAGS.config.batch_size} ({FLAGS.config.batch_size // len(devices) } per device)
        # Steps: {FLAGS.config.num_steps}
        # Window size: {FLAGS.config.window_size}
    """
    )

    #########
    #
    # Setup Jax Data Parallelism
    #
    #########

    # create a 1D mesh with a single axis named "batch"
    mesh = Mesh(jax.devices(), axis_names="batch")
    
    # replicated sharding -- does not shard arrays
    replicated_sharding = NamedSharding(mesh, PartitionSpec())
    # Our batches will be data-parallel sharded -- each device will get a slice of the batch
    dp_sharding = NamedSharding(mesh, PartitionSpec("batch"))

    def shard(batch):
        return multihost_utils.host_local_array_to_global_array(
            batch, mesh, PartitionSpec("batch")
        )


    # prevent tensorflow from using GPU memory since it's only used for data loading
    tf.config.set_visible_devices([], "GPU")
    
    # make sure each process loads different data
    tf.random.set_seed(FLAGS.config.seed + jax.process_index())

    #########
    #
    # Setup WandB
    #
    #########

    name = format_name_with_config(
        FLAGS.name,
        FLAGS.config.to_dict(),
    )
    wandb_id = "{name}_{time}".format(
        name=name,
        time=datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
    )
    wandb_id = jax_utils.host_broadcast_str(wandb_id)
    if jax.process_index() == 0:
        wandb.init(
            config=FLAGS.config.to_dict(),
            id=wandb_id,
            name=name,
            mode="disabled" if FLAGS.debug else None,
            **FLAGS.config.wandb,
        )


    #########
    #
    # Load Pretrained model + optionally modify config
    #
    #########
    pretrained_model_kwargs = {"checkpoint_path": FLAGS.config.pretrained_path}
    if hasattr(FLAGS.config, "pretrained_step"): 
        pretrained_model_kwargs["step"] = FLAGS.config.pretrained_step

    pretrained_model = OctoModel.load_pretrained(
        **pretrained_model_kwargs
    )

    flat_config = flax.traverse_util.flatten_dict(
        pretrained_model.config, keep_empty_nodes=True
    )

    for d_key in flax.traverse_util.flatten_dict(
        FLAGS.config.get("config_delete_keys", ConfigDict()).to_dict()
    ):
        for c_key in list(flat_config.keys()):
            if ".".join(c_key).startswith(".".join(d_key)):
                del flat_config[c_key]

    
    update_config = FLAGS.config.get('update_config', ConfigDict())
    flat_update_config = flatten_dict(update_config.to_dict())
    flat_config.update(flat_update_config)
    config = ConfigDict(flax.traverse_util.unflatten_dict(flat_config))
    config = config.to_dict()

    check_config_diff(config, pretrained_model.config)

    #########
    #
    # Setup Data Loader
    #
    #########
    dataset = make_single_dataset(
        FLAGS.config.dataset_kwargs,
        traj_transform_kwargs=FLAGS.config.traj_transform_kwargs,
        frame_transform_kwargs=FLAGS.config.frame_transform_kwargs,
        train=True,
    )

    FLAGS.config.batch_size //= jax.process_count()
    if FLAGS.config.get('text_processor', None): 
        text_processor = ModuleSpec.instantiate(FLAGS.config["text_processor"])()
    else:
        text_processor = ModuleSpec.instantiate(config["text_processor"])()

    def process_batch(batch):
        batch = process_text_fuse(batch, text_processor)
        del batch["dataset_name"]
        return batch

    train_data_iter = (
        dataset.repeat()
        .unbatch()
        .shuffle(FLAGS.config.shuffle_buffer_size)
        .batch(FLAGS.config.batch_size)
        .iterator()
    )
    
    train_data_iter = map(
        shard,
        map(
            process_batch,
            train_data_iter
        )
    )
    example_batch = next(train_data_iter)

    logging.info(f"Batch size: {example_batch['action'].shape[0]}")
    logging.info(f"Number of devices: {jax.device_count()}")
    logging.info(
        f"Batch size per device: {example_batch['action'].shape[0] // jax.device_count()}"
    )

    # print example batch 
    print_separator(logging.info)
    logging.info('Example batch:\n\n')
    pretty_print_dict(
        jax.tree_map(lambda arr: f'{arr.shape}  {arr.dtype}', example_batch), 
        log_func=logging.info
    )
    print_separator(logging.info)


    #########
    #
    # Create new model and merge parameters
    #
    #########
    rng = jax.random.PRNGKey(FLAGS.config.seed)
    rng, init_rng = jax.random.split(rng)
    model = OctoModel.from_config(
        config,
        example_batch,
        text_processor,
        rng=init_rng,
        dataset_statistics=dataset.dataset_statistics,
    )
    merged_params = merge_params(model.params, pretrained_model.params)
    model = model.replace(params=merged_params)
    del pretrained_model
    for loader in FLAGS.config.pretrained_loaders:
        if not callable(loader):  # Means that it is a ModuleSpec
            loader = ModuleSpec.instantiate(loader)
        model = model.replace(params=loader(model.params))


    # Log all parameters
    flattened = flatten_dict(model.params)
    for key in flattened.keys():
        logging.debug(key)


    #########
    #
    # Setup Optimizer and Train State
    #
    #########
    params = model.params
    if FLAGS.config.optimizer.frozen_keys is None:
        FLAGS.config.optimizer.frozen_keys = model.config["optimizer"]["frozen_keys"]

    tx, lr_callable, param_norm_callable = create_optimizer(
        params,
        **FLAGS.config.optimizer.to_dict(),
    )
    train_state = TrainState.create(
        model=model,
        tx=tx,
        rng=rng,
    )
    train_state = jax.device_get(train_state)

    #########
    #
    # Save all metadata
    #
    #########

    if FLAGS.config.save_dir is not None:
        save_dir = tf.io.gfile.join(
            FLAGS.config.save_dir,
            FLAGS.config.wandb.project,
            FLAGS.config.wandb.group or "",
            wandb_id,
        )
        if jax.process_index() == 0:
            wandb.config.update(dict(save_dir=save_dir), allow_val_change=True)
        logging.info("Saving to %s", save_dir)
        save_callback = SaveCallback(save_dir)

        # Add window_size to top of config, to make eval easier
        new_config = ConfigDict(model.config)
        new_config["window_size"] = example_batch["observation"][
            "timestep_pad_mask"
        ].shape[1]
        model = model.replace(config=new_config)

        # Save finetuning config since it's not saved by SaveCallback, i.e. as part of model.save_pretrained()
        with tf.io.gfile.GFile(
            tf.io.gfile.join(save_dir, "finetune_config.json"), "w"
        ) as config_file:
            config_file.write(FLAGS.config.to_json_best_effort())
    else:
        save_dir = None
        save_callback = SaveCallback(None)
        logging.warning("save_dir not passed in, not saving checkpoints")

    example_batch_spec = jax.tree_map(
        lambda arr: (arr.shape, str(arr.dtype)), example_batch
    )
    if jax.process_index() == 0:
        wandb.config.update(
            dict(example_batch_spec=example_batch_spec), allow_val_change=True
        )

    #########
    #
    # Define loss, train_step, and eval_step
    #
    #########

    loss_fn_contrastive = fuse_loss_fn_contrastive
    loss_fn_generative = partial(fuse_loss_fn_generative, mask_invalid_language=FLAGS.config.get("mask_invalid_language", True))

    def loss_fn_action(bound_module: OctoModule, batch: Data, train: bool = True):
        transformer_embeddings = bound_module.octo_transformer(
            batch["observation"],
            batch["task"],
            batch["observation"]["timestep_pad_mask"],
            train=train,
        )
        action_loss, action_metrics = bound_module.heads["action"].loss(
            transformer_embeddings,  # action head knows to pull out the "action" readout_key
            batch["action"],
            batch["observation"]["timestep_pad_mask"],
            batch["action_pad_mask"],
            train=train,
        )
        action_metrics = append_identity_to_metrics(action_metrics, 'ac')
        return action_loss, action_metrics
    
    
    def loss_fn(params, batch, rng, train=True, use_action_loss=True, use_contrastive_loss=True, use_generative_loss=True, **kwargs): 
        info = {}
        loss = 0.0
        bound_module = model.module.bind({"params": params}, rngs={"dropout": rng})
        if use_action_loss:
            ac_loss, ac_metrics = loss_fn_action(bound_module, batch, train, **kwargs)
            info.update(ac_metrics)
            loss += ac_loss
            
        if use_contrastive_loss: 
            lang_loss, lang_metrics = loss_fn_contrastive(bound_module, batch, train, **kwargs)
            info.update(lang_metrics)
            loss += lang_loss
        
        if use_generative_loss:
            gen_loss, gen_metrics = loss_fn_generative(model=model, params=params, rng=rng, batch=batch, **kwargs)
            info.update(gen_metrics)
            loss += gen_loss
        
        info['loss_total'] = loss
        return loss, info

    # Data parallelism
    # Model is replicated across devices, data is   split across devices
    @partial(
        jax.jit,
        in_shardings=(replicated_sharding, dp_sharding),
        out_shardings=(replicated_sharding, replicated_sharding),
        # allows jax to modify `state` in-place, saving a lot of memory
        donate_argnums=0,
    )
    def train_step(state: TrainState, batch):
        rng, dropout_rng = jax.random.split(state.rng)
        (loss, info), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            state.model.params, batch, dropout_rng, train=True, use_action_loss=True, use_contrastive_loss=True, use_generative_loss=True,
        )
        grad_norm = optax.global_norm(grads)
        updates, _ = state.tx.update(grads, state.opt_state, state.model.params)
        update_norm = optax.global_norm(updates)
        info.update(
            {
                "grad_norm": grad_norm,
                "update_norm": update_norm,
                "param_norm": param_norm_callable(state.model.params),
                "learning_rate": lr_callable(state.step),
            }
        )
        new_state = state.apply_gradients(grads=grads, rng=rng)
        return new_state, info

    #########
    #
    # Build validation callback
    #
    #########
    dataset_kwargs_list = [FLAGS.config.dataset_kwargs]
    val_callback = ValidationCallback(
        loss_fn=partial(loss_fn, use_action_loss=True, use_contrastive_loss=True, use_generative_loss=True),
        process_batch_fn=process_batch,
        text_processor=text_processor,
        val_dataset_kwargs_list=dataset_kwargs_list,
        dataset_kwargs=FLAGS.config,
        modes_to_evaluate=["text_conditioned",],
        **FLAGS.config.val_kwargs,
    )

    lang_callback = LanguageCallback(
        get_ids_fn=fuse_decode_ids,
        process_batch_fn=process_batch,
        text_processor=text_processor,
        lang_dataset_kwargs_list=dataset_kwargs_list,
        dataset_kwargs=FLAGS.config,
        **FLAGS.config.val_kwargs,
    )

    #########
    #
    # Train loop
    #
    #########
    def wandb_log(info, step):
        if jax.process_index() == 0:
            wandb.log(flatten_dict(info, sep="/"), step=step)

    timer = Timer()
    for i in tqdm.tqdm(
        range(0, int(FLAGS.config.num_steps)),
        total=int(FLAGS.config.num_steps),
        dynamic_ncols=True,
    ):
        timer.tick("total")

        with timer("dataset"):
            batch = next(train_data_iter)
        

        with timer("train"):
            train_state, update_info = train_step(train_state, batch)

        timer.tock("total")

        if (i + 1) % FLAGS.config.log_interval == 0:
            update_info = jax.device_get(update_info)
            wandb_log(
                {"training": update_info, "timer": timer.get_average_times()}, step=i
            )

        if (i + 1) % FLAGS.config.eval_interval == 0:
            logging.info("Evaluating...")

            with timer("val"):
                val_metrics = val_callback(train_state, i + 1)
                wandb_log(val_metrics, step=i)

        if (i + 1) % FLAGS.config.lang_interval == 0:
            logging.info("Evaluating lang...")

            with timer("lang"):
                lang_metrics = lang_callback(train_state, i + 1)
                wandb_log(lang_metrics, step=i)

        if (i + 1) % FLAGS.config.save_interval == 0 and save_dir is not None:
            logging.info("Saving checkpoint...")
            save_callback(train_state, i + 1)


if __name__ == "__main__":
    app.run(main)
