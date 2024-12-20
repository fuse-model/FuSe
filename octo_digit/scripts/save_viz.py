import datetime
from enum import Flag
from functools import partial
import importlib
import os

from absl import app, flags, logging
import flax
from flax.traverse_util import flatten_dict, unflatten_dict
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from ml_collections import config_flags, ConfigDict
from octo.data.dataset import make_single_dataset
from octo.model.components.tokenizers import (
    BinTokenizer,
    ImageTokenizer,
    LowdimObsTokenizer,
    ProjectionTokenizer,
    SiglipTokenizer,
    UnsqueezingImageTokenizer,
)
from octo.model.components.vit_encoders import SmallStem16
from octo.model.octo_model import OctoModel
from octo.utils.jax_utils import initialize_compilation_cache
from octo.utils.spec import ModuleSpec
from octo.utils.train_callbacks import (
    RolloutVisualizationCallback,
    SaveCallback,
    ValidationCallback,
    VisualizationCallback,
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
import optax
import tensorflow as tf
import tqdm
import wandb

try:
    from jax_smi import initialise_tracking  # type: ignore

    initialise_tracking()
except ImportError:
    pass

FLAGS = flags.FLAGS

flags.DEFINE_string("name", "experiment", "Experiment name.")
flags.DEFINE_bool("debug", False, "Debug config (no wandb logging)")

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
INDENT = "".join([" " for _ in range(INDENT_SIZE)])


def recursive_dict_print(dictionary, prefix=""):
    for key, val in dictionary.items():
        key = key[:MAX_KEY_LEN]
        if isinstance(val, dict):
            print(f"{prefix}{key}")
            new_prefix = prefix + INDENT
            recursive_dict_print(val, new_prefix)
        else:
            indent = "".join([" " for _ in range(INDENT_SIZE - len(key))])
            print(f"{prefix}{key}:{indent}{val.shape}")


def main(_):
    initialize_compilation_cache()
    devices = jax.devices()
    logging.info(
        f"""
        Octo Finetuning Script
        ======================
        Pretrained model: {FLAGS.config.pretrained_path}
        Finetuning Dataset: {FLAGS.config.dataset_kwargs.name}
        Data dir: {FLAGS.config.dataset_kwargs.data_dir}
        Task Modality: {FLAGS.config.modality}
        Finetuning Mode: {FLAGS.config.finetuning_mode}

        # Devices: {jax.device_count()}
        Batch size: {FLAGS.config.batch_size} ({FLAGS.config.batch_size // len(devices) } per device)
        # Steps: {FLAGS.config.num_steps}
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
    def wandb_log(info, step):
        wandb.log(flatten_dict(info, sep="/"), step=step)

    timer = Timer()
    gcloud_path = "gs://619c8f721786ba/octo_ckpts/octo/"
    run_args_list = [
        {
            "name": "finetune_vizonly_diffusion_val",
            "timestamp": "20240517_021703",
            "dataset_name": "digit_dataset:8.8.0",
        },
        {
            "name": "finetune_vizonly_mse_val",
            "timestamp": "20240517_022456",
            "dataset_name": "digit_dataset:8.8.0",
        },
        {
            "name": "finetune_vizonly_diffusion_val_small",
            "timestamp": "20240517_022228",
            "dataset_name": "digit_dataset:9.9.0",
        },
        {
            "name": "finetune_vizonly_mse_val_small",
            "timestamp": "20240517_022019",
            "dataset_name": "digit_dataset:9.9.0",
        },
    ]

    for run_args in run_args_list:

        FLAGS.config["dataset_kwargs"]["name"] = run_args["dataset_name"]
        name = format_name_with_config(
            run_args["name"],
            FLAGS.config.to_dict(),
        )
        wandb_id = "{name}_{time}".format(name=name, time=run_args["timestamp"])
        wandb.init(
            config=FLAGS.config.to_dict(),
            id=wandb_id,
            name=name,
            resume="must",
            mode="disabled" if FLAGS.debug else None,
            **FLAGS.config.wandb,
        )

        #########
        #
        # Load Pretrained model + optionally modify config
        #
        #########
        ckpt_path = gcloud_path + run_args["name"] + "_" + run_args["timestamp"]
        pretrained_model_kwargs = {"checkpoint_path": ckpt_path}
        for step in range(5000, 50001, 5000):
            pretrained_model_kwargs["step"] = step
            pretrained_model = OctoModel.load_pretrained(**pretrained_model_kwargs)
            rng = jax.random.PRNGKey(FLAGS.config.seed)
            rng, init_rng = jax.random.split(rng)
            model = pretrained_model

            flat_config = flax.traverse_util.flatten_dict(
                pretrained_model.config, keep_empty_nodes=True
            )

            config = ConfigDict(flax.traverse_util.unflatten_dict(flat_config))
            # config.update(FLAGS.config.get("update_config", ConfigDict()))
            config = config.to_dict()
            # check_config_diff(config, pretrained_model.config)
            #########
            #
            # Setup Data Loader
            #
            #########

            # create text processor
            if config["text_processor"] is None:
                text_processor = None
            else:
                text_processor = ModuleSpec.instantiate(config["text_processor"])()

            def process_batch(batch):
                batch = process_text(batch, text_processor)
                del batch["dataset_name"]
                return batch

            params = model.params
            if FLAGS.config.optimizer.frozen_keys is None:
                FLAGS.config.optimizer.frozen_keys = model.config["optimizer"][
                    "frozen_keys"
                ]

            tx, lr_callable, param_norm_callable = create_optimizer(
                params,
                **FLAGS.config.optimizer.to_dict(),
            )
            train_state = TrainState.create(
                model=model,
                tx=tx,
                rng=rng,
            )

            if FLAGS.config.modality == "image_conditioned":
                modes_to_evaluate = ["image_conditioned"]
            elif FLAGS.config.modality == "text_conditioned":
                modes_to_evaluate = ["text_conditioned"]
            elif FLAGS.config.modality == "multimodal":
                modes_to_evaluate = ["image_conditioned", "text_conditioned"]
            else:
                modes_to_evaluate = ["base"]

            dataset_kwargs_list = [FLAGS.config.dataset_kwargs]

            viz_callback = VisualizationCallback(
                text_processor=text_processor,
                val_dataset_kwargs_list=dataset_kwargs_list,
                dataset_kwargs=FLAGS.config,
                modes_to_evaluate=modes_to_evaluate,
                **FLAGS.config.viz_kwargs,
            )

            with timer("visualize"):
                viz_metrics = viz_callback(train_state, step)
                wandb_log(viz_metrics, step=step)


if __name__ == "__main__":
    app.run(main)
