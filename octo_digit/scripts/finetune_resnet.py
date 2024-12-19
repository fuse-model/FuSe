import datetime
from functools import partial
import os

import jax.numpy as jnp
from absl import app, flags, logging
import flax
from flax.traverse_util import flatten_dict, unflatten_dict
import jax
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from ml_collections import config_flags, ConfigDict
import importlib
from octo.data.utils.text_processing import MuseEmbedding
from octo.model.components.tokenizers import BinTokenizer, LowdimObsTokenizer, ImageTokenizer, UnsqueezingImageTokenizer, ProjectionTokenizer, SiglipTokenizer
import optax
import tensorflow as tf
import tqdm
import wandb
from octo.model.components.vit_encoders import ResNet26, SmallStem32
from octo.model.resnet_model import ResnetModel


from octo.model.components.vit_encoders import SmallStem16
from octo.data.dataset import make_single_dataset
from octo.model.octo_model import OctoModel
from octo.utils.jax_utils import initialize_compilation_cache
from octo.utils.spec import ModuleSpec
from octo.utils.train_callbacks import (
    RolloutVisualizationCallback,
    SaveCallback,
    ValidationCallback,
    VisualizationCallback,
    GradCAMVisualizationCallback,
)
from octo.utils.train_utils import (
    check_config_diff,
    create_optimizer,
    format_name_with_config,
    merge_params,
    process_text,
    Timer,
    TrainState,
)

try:
    from jax_smi import initialise_tracking  # type: ignore

    initialise_tracking()
except ImportError:
    pass

FLAGS = flags.FLAGS

flags.DEFINE_string("name", "experiment", "Experiment name.")
flags.DEFINE_bool("debug", False, "Debug config (no wandb logging)")
flags.DEFINE_integer("o_window_size", -1, "override window size")
flags.DEFINE_integer("o_batch_size", -1, "override batch size")
flags.DEFINE_integer("o_steps", -1, "override step ct")

flags.DEFINE_integer("o_img_emb", -1, "override resnet image embedding")
flags.DEFINE_integer("o_num_layers", -1, "override mlp hidden layers")
flags.DEFINE_integer("o_hidden_layer_width", -1, "override hidden layer width")

default_config_file = os.path.join(
    os.path.dirname(__file__), "configs/finetune_config.py"
)
config_flags.DEFINE_config_file(
    "config",
    default_config_file,
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)


MAX_KEY_LEN = 15
INDENT_SIZE = MAX_KEY_LEN + 4
INDENT = ''.join([' ' for _ in range(INDENT_SIZE)])
def recursive_dict_print(dictionary, prefix=""): 
    for key, val in dictionary.items(): 
        key = key[:MAX_KEY_LEN]
        if isinstance(val, dict): 
            print(f'{prefix}{key}')
            new_prefix = prefix + INDENT
            recursive_dict_print(val, new_prefix)
        else: 
            indent = ''.join([' ' for _ in range(INDENT_SIZE - len(key))])
            print(f'{prefix}{key}:{indent}{val.shape}')


def main(_):
    initialize_compilation_cache()
    devices = jax.devices()
    if FLAGS.o_batch_size > 0: 
        FLAGS.config.batch_size = FLAGS.o_batch_size
    if FLAGS.o_window_size > 0: 
        FLAGS.config.window_size = FLAGS.o_window_size
    if FLAGS.o_img_emb > 0: 
        FLAGS.config.model['image_embedding_size'] = FLAGS.o_img_emb
    if FLAGS.o_num_layers > 0 or FLAGS.o_hidden_layer_width > 0: 
        assert FLAGS.o_num_layers > 0 and FLAGS.o_hidden_layer_width
        FLAGS.config.model['mlp_widths'] = tuple([FLAGS.o_hidden_layer_width for _ in range(FLAGS.o_num_layers)])
 
    logging.info(
        f"""
        Resnet Finetuning Script
        ======================
        Finetuning Dataset: {FLAGS.config.dataset_kwargs.name}
        Data dir: {FLAGS.config.dataset_kwargs.data_dir}
        Task Modality: {FLAGS.config.modality}
        Finetuning Mode: {FLAGS.config.finetuning_mode}

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

    assert (
        FLAGS.config.batch_size % len(devices) == 0
    ), f"Batch size ({FLAGS.config.batch_size}) must be divisible by the number of devices ({len(devices)})"
    assert (
        FLAGS.config.viz_kwargs.eval_batch_size % len(devices) == 0
    ), f"Eval batch size ({FLAGS.config.viz_kwargs.eval_batch_size}) must be divisible by the number of devices ({len(devices)})"

    # create a 1D mesh with a single axis named "batch"
    mesh = Mesh(jax.devices(), axis_names="batch")
    # Our batches will be data-parallel sharded -- each device will get a slice of the batch
    dp_sharding = NamedSharding(mesh, PartitionSpec("batch"))
    # Our model will be replicated across devices (we are only doing data parallelism, not model parallelism)
    replicated_sharding = NamedSharding(mesh, PartitionSpec())

    # prevent tensorflow from using GPU memory since it's only used for data loading
    tf.config.set_visible_devices([], "GPU")

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
    wandb.init(
        config=FLAGS.config.to_dict(),
        id=wandb_id,
        name=name,
        mode="disabled" if FLAGS.debug else None,
        **FLAGS.config.wandb,
    )

            
    #########
    #
    # Setup Data Loader
    #
    #########

    # create text processor
    text_processor = MuseEmbedding()

    def process_batch(batch):
        batch = process_text(batch, text_processor)
        del batch["dataset_name"]
        return batch

    dataset = make_single_dataset(
        FLAGS.config.dataset_kwargs,
        traj_transform_kwargs=FLAGS.config.traj_transform_kwargs,
        frame_transform_kwargs=FLAGS.config.frame_transform_kwargs,
        train=True,
    )
    train_data_iter = (
        dataset.repeat()
        .unbatch()
        .shuffle(FLAGS.config.shuffle_buffer_size)
        .batch(FLAGS.config.batch_size)
        .iterator()
    )
    train_data_iter = map(process_batch, train_data_iter)
    example_batch = next(train_data_iter)


    # print example batch 
    print("############################################")
    print('Example batch:')
    print('\n')
    recursive_dict_print(example_batch)
    print('\n')
    print("############################################")




    
    rng = jax.random.PRNGKey(FLAGS.config.seed)
    rng, init_rng = jax.random.split(rng)
    
    config = FLAGS.config.to_dict() 
    model = ResnetModel.from_config(
        config, 
        example_batch=example_batch, 
        text_processor=text_processor
    )


    #########
    #
    # Setup Optimizer and Train State
    #
    #########


    # make sure keys frozen here 
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
    
    print(train_state.model.config['model'])
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
    wandb.config.update(
        dict(example_batch_spec=example_batch_spec), allow_val_change=True
    )

    #########
    #
    # Define loss, train_step, and eval_step
    #
    #########

    def loss_fn(params, batch, rng, train=True):
        bound_module = model.module.bind({"params": params}, rngs={"dropout": rng})
        pred_action = bound_module(
            batch['observation'],
            batch['task']
        )
        action_loss = jnp.mean(jnp.square(batch['action'] - pred_action))
        action_metrics = { 
            "loss": action_loss, 
            "mse": action_loss
        }


        return action_loss, action_metrics

    # Data parallelism
    # Model is replicated across devices, data is split across devices
    @partial(
        jax.jit,
        in_shardings=[replicated_sharding, dp_sharding],
    )
    def train_step(state: TrainState, batch):
        rng, dropout_rng = jax.random.split(state.rng)
        (loss, info), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            state.model.params, batch, dropout_rng, train=True
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
    # Build validation & visualization callbacks
    #
    #########

    if FLAGS.config.modality == "image_conditioned":
        modes_to_evaluate = ["image_conditioned"]
    elif FLAGS.config.modality == "text_conditioned":
        modes_to_evaluate = ["text_conditioned"]
    elif FLAGS.config.modality == "multimodal":
        modes_to_evaluate = ["image_conditioned", "text_conditioned"]
    else:
        modes_to_evaluate = ["base"]

    dataset_kwargs_list = [FLAGS.config.dataset_kwargs]

    val_callback = ValidationCallback(
        loss_fn=loss_fn,
        process_batch_fn=process_batch,
        text_processor=text_processor,
        val_dataset_kwargs_list=dataset_kwargs_list,
        dataset_kwargs=FLAGS.config,
        modes_to_evaluate=modes_to_evaluate,
        **FLAGS.config.val_kwargs,
    )


    # viz_callback = VisualizationCallback(
    #     text_processor=text_processor,
    #     val_dataset_kwargs_list=dataset_kwargs_list,
    #     dataset_kwargs=FLAGS.config,
    #     modes_to_evaluate=modes_to_evaluate,
    #     **FLAGS.config.viz_kwargs,
    # )

    # gradcam_callback = GradCAMVisualizationCallback(
    #     text_processor=text_processor,
    #     val_dataset_kwargs_list=dataset_kwargs_list, 
    #     dataset_kwargs=FLAGS.config, 
    #     **FLAGS.config.gradcam_kwargs
    # )
    #########
    #
    # Optionally build visualizers for sim env evals
    #
    #########

    # if "rollout_kwargs" in FLAGS.config:
    #     rollout_callback = RolloutVisualizationCallback(
    #         text_processor=text_processor,
    #         history_length=FLAGS.config["window_size"],
    #         model_pred_horizon=config["model"]["heads"]["action"]["kwargs"].get(
    #             "pred_horizon", 1
    #         ),
    #         **FLAGS.config.rollout_kwargs.to_dict(),
    #     )
    # else:
    #     rollout_callback = None

    #########
    #
    # Train loop
    #
    #########

    def wandb_log(info, step):
        wandb.log(flatten_dict(info, sep="/"), step=step)

    if FLAGS.o_steps > 0: 
        FLAGS.config.num_steps = FLAGS.o_steps

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

            # with timer("visualize"):
            #     viz_metrics = viz_callback(train_state, i + 1)
            #     wandb_log(viz_metrics, step=i)

            # with timer('gradcam'):
            #     gradcam_metrics = gradcam_callback(train_state, i + 1) 
            #     wandb_log(gradcam_metrics, step=i)

        #     if rollout_callback is not None:
        #         with timer("rollout"):
        #             rollout_metrics = rollout_callback(train_state, i + 1)
        #             wandb_log(rollout_metrics, step=i)

        if (i + 1) % FLAGS.config.save_interval == 0 and save_dir is not None:
            logging.info("Saving checkpoint...")
            save_callback(train_state, i + 1)


if __name__ == "__main__":
    app.run(main)
