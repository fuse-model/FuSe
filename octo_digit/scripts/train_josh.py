# WARNING: importing tensorflow too late can silence important logging (╯°□°)╯︵ ┻━┻
import tensorflow as tf

# isort: split

import datetime
from functools import partial
import os
import os.path as osp

from absl import app, flags, logging
from flax.traverse_util import flatten_dict
import jax
from jax.experimental import multihost_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from ml_collections import config_flags
import optax
import tqdm
import wandb
import numpy as np
import octo
from octo.data.dataset import make_interleaved_dataset, make_single_dataset
from octo.data.oxe import make_oxe_dataset_kwargs_and_weights
from octo.model.octo_model import OctoModel
from octo.model.octo_module import OctoModule
from octo.model.bcz_model import BczModel
from octo.model.bcz_module import BczModule
from octo.utils import jax_utils
from octo.utils.spec import ModuleSpec
from octo.utils.train_callbacks import (
    RolloutVisualizationCallback,
    SaveCallback,
    ValidationCallback,
    VisualizationCallback,
)
from octo.utils.train_utils import (
    create_optimizer,
    filter_eval_datasets,
    format_name_with_config,
    process_text,
    process_lang_list,
    # process_lang_list_muse,
    process_dropout_annotations,
    Timer,
    TrainState,
)
from octo.utils.typing import Data
from octo.data.utils.data_utils import AnnotationSelectionManager

FLAGS = flags.FLAGS

flags.DEFINE_string("name", "experiment", "Experiment name.")
flags.DEFINE_bool("debug", False, "Debug config (no wandb logging)")
flags.DEFINE_integer("o_window_size", -1, "override window size")
flags.DEFINE_integer("o_batch_size", -1, "override batch size")
flags.DEFINE_integer("o_steps", -1, "override step ct")

config_dir = os.path.join(os.path.dirname(__file__), "configs")
config_flags.DEFINE_config_file(
    "config",
    ''
    # os.path.join(config_dir, "config.py:transformer_bc"),
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)


def main(_):
    jax_utils.initialize_compilation_cache()
    if FLAGS.o_batch_size > 0: 
        FLAGS.config.batch_size = FLAGS.o_batch_size
    if FLAGS.o_window_size > 0: 
        FLAGS.config.window_size = FLAGS.o_window_size
    # assert FLAGS.config.dataset_kwargs.batch_size % jax.device_count() == 0
    # assert FLAGS.config.dataset_kwargs.batch_size % jax.process_count() == 0
    devices = jax.devices()
    assert (
        FLAGS.config.batch_size % len(devices) == 0
    ), f"Batch size ({FLAGS.config.batch_size}) must be divisible by the number of devices ({len(devices)})"
    assert (
        FLAGS.config.viz_kwargs.eval_batch_size % len(devices) == 0
    ), f"Eval batch size ({FLAGS.config.viz_kwargs.eval_batch_size}) must be divisible by the number of devices ({len(devices)})"


    logging.info(
        f"""
        Octo Training Script
        ======================
        Dataset: {FLAGS.config.dataset_kwargs.name}
        Data dir: {FLAGS.config.dataset_kwargs.data_dir}
        Task Modality: {FLAGS.config.modality}

        # Devices: {jax.device_count()}
        Batch size: {FLAGS.config.batch_size} ({FLAGS.config.batch_size // len(devices) } per device)
        # Steps: {FLAGS.config.num_steps}
        # Window size: {FLAGS.config.window_size}
    """
    )
    # create a 1D mesh with a single axis named "batch"
    mesh = Mesh(jax.devices(), axis_names="batch")
    # Our batches will be data-parallel sharded -- each device will get a slice of the batch
    dp_sharding = NamedSharding(mesh, PartitionSpec("batch"))
    # Our model will be replicated across devices (we are only doing data parallelism, not model parallelism)
    replicated_sharding = NamedSharding(mesh, PartitionSpec())

    # prevent tensorflow from using GPUs
    tf.config.set_visible_devices([], "GPU")

    # make sure each process loads different data
    tf.random.set_seed(FLAGS.config.seed + jax.process_index())

    # set up wandb and logging
    if FLAGS.config.get("wandb_resume_id", None) is None:
        # start_step = 0
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

        if FLAGS.config.save_dir is not None:
            save_dir = tf.io.gfile.join(
                FLAGS.config.save_dir,
                FLAGS.config.wandb.project,
                FLAGS.config.wandb.group or "",
                wandb_id,
            )
            logging.info("Saving to %s", save_dir)
            if jax.process_index() == 0:
                wandb.config.update(dict(save_dir=save_dir), allow_val_change=True)
        else:
            save_dir = None
            logging.info("save_dir not passed in, not saving checkpoints")
    else:
        # resume previous run
        # start_step = FLAGS.config.wandb_resume_step
        wandb_run = wandb.Api().run(FLAGS.config.wandb_resume_id)
        if jax.process_index() == 0:
            wandb.init(
                project=wandb_run.project,
                id=wandb_run.id,
                entity=wandb_run.entity,
                resume="must",
            )
        save_dir = wandb_run.config["save_dir"]
        logging.info("Resuming run %s", FLAGS.config.wandb_resume_id)



    if jax.process_index() == 0:
        codebase_directory = osp.abspath(osp.join(osp.dirname(octo.__file__), ".."))
        wandb.run.log_code(codebase_directory)


    Model = BczModel if FLAGS.config.get('is_bcz', False) else OctoModel
    # set up text tokenization (this needs to happen after batching but before sharding)
    if FLAGS.config.text_processor is None:
        text_processor = None
    else:
        text_processor = ModuleSpec.instantiate(FLAGS.config.text_processor)()

    train_data = make_single_dataset(
        FLAGS.config.dataset_kwargs,
        traj_transform_kwargs=FLAGS.config.traj_transform_kwargs,
        frame_transform_kwargs=FLAGS.config.frame_transform_kwargs,
        train=True,
    )
    
    annotation_manager: AnnotationSelectionManager = train_data.supp_info['annotation_manager']
    
    if FLAGS.config['dataset_kwargs']['language_key'] == 'multimodal_annotations':  
        process_text_func = partial(process_dropout_annotations, batch_size=FLAGS.config.batch_size, keys=np.array(train_data.annotation_keys), probabilities=dataset.annotation_probabilities)
    elif FLAGS.config['dataset_kwargs']['language_key'] == 'all_lang_list': 
        # if Model == OctoModel:
        process_text_func = partial(process_lang_list, batch_size=FLAGS.config.batch_size, annotation_manager=annotation_manager)
        # elif Model == BczModel: 
        #     process_text_func = partial(process_lang_list_muse, batch_size=FLAGS.config.batch_size, annotation_manager=annotation_manager)
        # else: 
        #     raise RuntimeError
    else: 
        process_text_func = process_text

    def process_batch(batch):
        batch = process_text_func(batch, text_processor)
        del batch["dataset_name"]
        return batch
    


    train_data_iter = (
        train_data.repeat()
        .unbatch()
        .shuffle(FLAGS.config.shuffle_buffer_size)
        .batch(FLAGS.config.batch_size)
        .iterator()
    )
    train_data_iter = map(process_batch, train_data_iter)

    example_batch = next(train_data_iter)
    logging.info(f"Batch size: {example_batch['action'].shape[0]}")
    logging.info(f"Number of devices: {jax.device_count()}")
    logging.info(
        f"Batch size per device: {example_batch['action'].shape[0] // jax.device_count()}"
    )
    
    MAX_KEY_LEN = 15
    INDENT_SIZE = MAX_KEY_LEN + 4
    INDENT = ''.join([' ' for _ in range(INDENT_SIZE)])
    def recursive_dict_print(dictionary, prefix="", log_func=print): 
        lines = [] 
        def dfs(dictionary, prefix=""):
            for key, val in dictionary.items(): 
                key = key[:MAX_KEY_LEN]
                if isinstance(val, dict): 
                    lines.append(f'{prefix}{key}')
                    new_prefix = prefix + INDENT
                    dfs(val, new_prefix)
                else: 
                    indent = ''.join([' ' for _ in range(INDENT_SIZE - len(key))])
                    lines.append(f'{prefix}{key}:{indent}{val.shape}  {val.dtype}')
        dfs(dictionary, prefix)
        return lines
    example_lines = []
    example_lines.append("############################################")
    example_lines.append('Example batch:')
    example_lines.append('\n\n')
    example_lines.extend(recursive_dict_print(example_batch))

    example_lines.append('\n\n')
    example_lines.append("############################################")
    logging.info('\n'.join(example_lines))

    # set up model and initialize weights
    rng = jax.random.PRNGKey(FLAGS.config.seed)
    rng, init_rng = jax.random.split(rng)
    model = Model.from_config(
        FLAGS.config.to_dict(),
        example_batch,
        text_processor,
        verbose=True,
        rng=init_rng,
        dataset_statistics=train_data.dataset_statistics,
    )

    # create optimizer
    tx, lr_callable, param_norm_callable = create_optimizer(
        model.params,
        **FLAGS.config.optimizer.to_dict(),
    )

    # Load pretrained weights (e.g. text encoder) if necessary
    for loader in FLAGS.config.pretrained_loaders:
        if not callable(loader):  # Means that it is a ModuleSpec
            loader = ModuleSpec.instantiate(loader)
        model = model.replace(params=loader(model.params))

    # create train state
    train_state = TrainState.create(rng, model, tx)
    save_callback = SaveCallback(save_dir)
    
    if FLAGS.config.get("wandb_resume_id", None) is not None:
        train_state = save_callback.state_checkpointer.restore(
            save_callback.state_checkpointer.latest_step(), items=train_state
        )
        checkpoint_step = int(train_state.step)
        logging.info("Restored checkpoint from %s", save_dir)
        if FLAGS.config.start_step is not None:
            start_step = FLAGS.config.start_step  # start_step overrides checkpoint
        else:
            start_step = checkpoint_step
        logging.info("Starting training from step %d", start_step)
    else:
        start_step = FLAGS.config.start_step or 0
    train_state = train_state.replace(step=start_step)

    # refreshes the train state so it doesn't crash w/ certain pre-trained loaders
    train_state = jax.device_get(train_state)

    def append_identity_to_metrics(metrics: dict, identity_suffix: str) -> dict: 
        processed_metrics = {}
        for key, val in metrics.items(): 
            processed_metrics[f'{key}_{identity_suffix}'] = val
        return processed_metrics

    if Model == OctoModel:
        def loss_fn_ac(bound_module: OctoModule, batch: Data, train: bool = True, specify_lang_key: str = ''):
            cache_lang = batch['task']['language_instruction']
            if specify_lang_key: 
                batch['task']['language_instruction'] = batch['task'][annotation_manager.key_map[specify_lang_key]]
            # print('AC LOSS\n' * 100, batch['task']['language_instruction'])
            transformer_embeddings = bound_module.octo_transformer(
                batch["observation"],
                batch["task"],
                batch["observation"]["timestep_pad_mask"],
                train=train,
            )
            if specify_lang_key: 
                batch['task']['language_instruction'] = cache_lang
            action_loss, action_metrics = bound_module.heads["action"].loss(
                transformer_embeddings,  # action head knows to pull out the "action" readout_key
                batch["action"],
                batch["observation"]["timestep_pad_mask"],
                batch["action_pad_mask"],
                train=train,
            )
            action_metrics = append_identity_to_metrics(action_metrics, 'ac')
            return action_loss, action_metrics

        lang_loss_keys = [annotation_manager.key_map[key] for key in annotation_manager.reconstruction_loss_keys ]
        num_different_annotation_types =  len(lang_loss_keys)
        reconstruction_weight = FLAGS.config['reconstruction_loss_weight']
        effective_weight =  reconstruction_weight * 1.0 / num_different_annotation_types if num_different_annotation_types else 0.0
        
        def loss_fn_lang(bound_module: OctoModule, batch: Data, train: bool = True, **kwargs): 
            total_loss = 0.0
            info = {}
            cached_lang = batch['task'].pop('language_instruction')
            transformer_embeddings_no_lang = bound_module.octo_transformer(
                batch["observation"],
                batch["task"],
                batch["observation"]["timestep_pad_mask"],
                train=train,
            )
            for lang_key in lang_loss_keys: 
                batch['task']['language_instruction'] = batch['task'][lang_key]
                # print('LANG LOSS\n'*100, lang_key, batch['task']['language_instruction'])
                true_language_embeddings = bound_module.octo_transformer.embed_language(batch['task'], train=train)
                lang_loss, lang_metrics = bound_module.heads[f"language_{lang_key}"].loss(
                    transformer_embeddings_no_lang,
                    true_language_embeddings,
                    batch["observation"]["timestep_pad_mask"],
                    train=train,
                )
                lang_metrics = append_identity_to_metrics(lang_metrics, identity_suffix=annotation_manager.rev_key_map[lang_key])      
                total_loss += lang_loss
                info.update(lang_metrics)
            batch['task']['language_instruction'] = cached_lang 
            total_loss *= effective_weight
            return total_loss, info


        def loss_fn(params, batch, rng, train=True, eval_ac=True, eval_lang=True, **kwargs): 
            info = {}
            loss = 0.0
            bound_module = model.module.bind({"params": params}, rngs={"dropout": rng})
            if eval_ac:
                ac_loss, ac_metrics = loss_fn_ac(bound_module, batch, train, **kwargs)
                info.update(ac_metrics)
                loss += ac_loss
            if eval_lang: 
                lang_loss, lang_metrics = loss_fn_lang(bound_module, batch, train, **kwargs)
                info.update(lang_metrics)
                loss += lang_loss
            
            info['loss_total'] = loss
            return loss, info
    elif Model == BczModel:
        def loss_fn_calc(bound_module: BczModule, batch: Data, train: bool = True):
            encoding = bound_module.bcz_encoder(
                batch["observation"],
                batch["task"],
                batch["observation"]["timestep_pad_mask"],
                train=train,
            )
            action_loss, action_metrics = bound_module.heads["action"].loss(
                encoding,  # action head knows to pull out the "action" readout_key
                batch["action"],
                batch["observation"]["timestep_pad_mask"],
                batch["action_pad_mask"],
                train=train,
            )
            metrics = append_identity_to_metrics(action_metrics, 'ac')
            lang_loss, lang_metrics = bound_module.heads['language'].loss(
                encoding, 
                batch['task']['language_instruction'], 
                batch['observation']['timestep_pad_mask'], 
                train=train
            )
            metrics.update(append_identity_to_metrics(lang_metrics, 'lang'))
            loss = action_loss + lang_loss
            return loss, metrics

        def loss_fn(params, batch, rng, train=True,  **kwargs): 
            bound_module = model.module.bind({"params": params}, rngs={"dropout": rng})
            loss, info = loss_fn_calc(bound_module, batch, train=train)
            info['loss_total'] = loss
            return loss, info 
    
    else: 
        raise ValueError(Model)

    @partial(
        jax.jit,
        # state is replicated, batch is data-parallel
        in_shardings=(replicated_sharding, dp_sharding),
        out_shardings=(replicated_sharding, replicated_sharding),
        # allows jax to modify `state` in-place, saving a lot of memory
        donate_argnums=0,
    )
    def train_step(state: TrainState, batch: Data):
        rng, dropout_rng = jax.random.split(state.rng)
        (loss, info), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            state.model.params, batch, dropout_rng, train=True
        )
        logging.info('GRADS:    ')
        for k, v in grads.items(): 
            logging.info(k, v)
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

    # val_datasets_kwargs_list, _ = filter_eval_datasets(
    #     FLAGS.config.dataset_kwargs["dataset_kwargs_list"],
    #     FLAGS.config.dataset_kwargs["sample_weights"],
    #     FLAGS.config.eval_datasets,
    # )
    dataset_kwargs_list = [FLAGS.config.dataset_kwargs]
    modes_to_evaluate = ["text_conditioned"]
    if Model == OctoModel: 
        val_callback = ValidationCallback(
            loss_fns={ 
                'ac_only': partial(loss_fn, eval_ac=True, eval_lang=False), 
                'combined': loss_fn
            },
            lang_modes = {
                'ac_only': annotation_manager.valid_keys
            },
            process_batch_fn=process_batch,
            text_processor=text_processor,
            val_dataset_kwargs_list=dataset_kwargs_list,
            dataset_kwargs=FLAGS.config,
            modes_to_evaluate=modes_to_evaluate,
            **FLAGS.config.val_kwargs,
        )
    elif Model == BczModel: 
        val_callback = ValidationCallback(
            loss_fns={ 
                'loss': loss_fn, 
            },
            lang_modes = {},
            process_batch_fn=process_batch,
            text_processor=text_processor,
            val_dataset_kwargs_list=dataset_kwargs_list,
            dataset_kwargs=FLAGS.config,
            modes_to_evaluate=modes_to_evaluate,
            **FLAGS.config.val_kwargs,
        )
        
    else: 
        raise ValueError(Model)
    # viz_callback = VisualizationCallback(
    #     text_processor=text_processor,
    #     val_dataset_kwargs_list=val_datasets_kwargs_list,
    #     dataset_kwargs=FLAGS.config.dataset_kwargs,
    #     **FLAGS.config.viz_kwargs.to_dict(),
    # )
    if "rollout_kwargs" in FLAGS.config:
        rollout_kwargs = FLAGS.config.rollout_kwargs.to_dict()
        dataset_name = rollout_kwargs.pop("dataset_name")
        rollout_callback = RolloutVisualizationCallback(
            text_processor=text_processor,
            unnormalization_statistics=train_data.dataset_statistics[dataset_name][
                "action"
            ],
            **rollout_kwargs,
        )
    else:
        rollout_callback = None

    def wandb_log(info, step):
        if jax.process_index() == 0:
            wandb.log(flatten_dict(info, sep="/"), step=step)

    timer = Timer()
    for i in tqdm.tqdm(
        range(start_step, int(FLAGS.config.num_steps)),
        total=int(FLAGS.config.num_steps),
        initial=start_step,
        dynamic_ncols=True,
    ):
        timer.tick("total")

        with timer("dataset"):
            batch = next(train_data_iter)

        with timer("train"):
            train_state, update_info = train_step(train_state, batch)

        if (i + 1) % FLAGS.config.save_interval == 0:
            save_callback(train_state, i + 1)

        if (i + 1) % FLAGS.config.eval_interval == 0:
            logging.info("Evaluating...")
            with timer("eval"):
                val_metrics = val_callback(train_state, i + 1)
                wandb_log(val_metrics, step=i + 1)

        # if (i + 1) % FLAGS.config.viz_interval == 0:
        #     logging.info("Visualizing...")
        #     with timer("visualize"):
        #         viz_metrics = viz_callback(train_state, i + 1)
        #         wandb_log(viz_metrics, step=i + 1)

        #     if rollout_callback is not None:
        #         with timer("rollout"):
        #             rollout_metrics = rollout_callback(train_state, i + 1)
        #             wandb_log(rollout_metrics, step=i + 1)

        timer.tock("total")
        if (i + 1) % FLAGS.config.log_interval == 0:
            update_info = jax.device_get(update_info)
            wandb_log(
                {"training": update_info, "timer": timer.get_average_times()},
                step=i + 1,
            )


if __name__ == "__main__":
    app.run(main)
