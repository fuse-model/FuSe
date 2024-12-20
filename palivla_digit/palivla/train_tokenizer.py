from typing import Tuple

from absl import app, flags
from flax import linen as nn
from flax.core.frozen_dict import freeze
import jax
from jax.experimental import multihost_utils
import jax.numpy as jnp
from ml_collections import config_flags
import numpy as np
import orbax.checkpoint as ocp
from palivla.dataset import make_base_dataset, transform_dataset
from palivla.load_model import make_optimizer
from palivla.spec import ModuleSpec, OptimizerSpec
from palivla.train_state import PaliVLATrainState, ShardingMetadata, TrainState
from palivla.train_step import TrainingBatch
from palivla.types import Data, Info
from palivla.utils import host_broadcast_str, key_string
from scalax.sharding import FSDPShardingRule, MeshShardingHelper, PartitionSpec
import tensorflow as tf
import tqdm
import wandb

jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
tf.config.set_visible_devices([], "GPU")


def train_step(model: TrainState, batch: Data) -> Tuple[TrainState, Info]:
    def loss_fn(params):
        return model.apply_fn(params, **batch, method="loss", train=True)

    grads, info = jax.grad(loss_fn, has_aux=True)(model.params)
    model = model.apply_gradients(grads=grads)
    return model, info


def eval_step(model: TrainState, batch: Data) -> Info:
    return model.apply_fn(model.params, **batch, method="loss", train=False)[1]


def main(_):
    try:
        jax.distributed.initialize()
    except:
        print(
            "Distributed initialization failed, probably because we're not using TPUs"
        )

    tf.random.set_seed(jax.process_index())

    config = flags.FLAGS.config

    # Setup mesh and sharding for model and data
    mesh = MeshShardingHelper([-1], ["data"])

    # Replicate the model, it's small enough that this is fine
    model_sharding = PartitionSpec()
    data_sharding = PartitionSpec("data")

    # Make the basic dataset
    # We have to do this first, since we need to know how the dataset is set up before we can construct the model
    train_ds = transform_dataset(
        make_base_dataset(config, train=True),
        None,
        generation=False,
        chunk_relative_actions=config.chunk_relative_actions,
        require_language=False,
    )
    eval_ds = transform_dataset(
        make_base_dataset(config, train=False),
        None,
        generation=False,
        chunk_relative_actions=config.chunk_relative_actions,
        require_language=False,
    )

    batch_shape = {
        "action": jax.ShapeDtypeStruct(
            shape=train_ds.element_spec["action"].shape, dtype=jnp.float32
        ),
    }

    sharding_metadata = ShardingMetadata(
        mesh=mesh,
        model_sharding_rule=model_sharding,
    )

    if config.resume_from_checkpoint_dir is None:
        model = TrainState.create(
            name="action_tokenizer",
            model_spec=ModuleSpec.from_dict(config.tokenizer.to_dict()),
            optimizer_spec=OptimizerSpec.from_dict(
                config.tokenizer_optimizer.to_dict()
            ),
            rng=jax.random.PRNGKey(0),
            batch_spec=batch_shape["action"],
            sharding_metadata=sharding_metadata,
        )
    else:
        restore_checkpoint_manager = ocp.CheckpointManager(
            config.resume_from_checkpoint_dir,
            item_handlers=TrainState.get_checkpoint_handlers("action_tokenizer"),
        )
        model = TrainState.restore(
            name="action_tokenizer",
            checkpoint_manager=restore_checkpoint_manager,
            load_optimizer=True,
            sharding_metadata=sharding_metadata,
            step=config.resume_from_checkpoint_step,
        )

    # Construct the final dataset
    # We need to do this after the model is constructed, since we need to have a tokenizer
    per_host_train_batch_size = config.batch_size // jax.process_count()
    per_host_eval_batch_size = config.eval_batch_size // jax.process_count()

    def make_training_batch(batch):
        return mesh.local_data_to_global_array(
            {
                "action": batch["action"],
            }
        )

    train_it = map(
        make_training_batch,
        train_ds.batch(per_host_train_batch_size).iterator(),
    )
    eval_it = map(
        make_training_batch,
        eval_ds.batch(per_host_eval_batch_size).iterator(),
    )

    # W&B setup
    if jax.process_index() == 0:
        flat_config_dict = {
            key_string(k): v
            for k, v in jax.tree_util.tree_leaves_with_path(config.to_dict())
        }
        wandb.init(project=config.wandb_project)
        wandb.run.name = config.run_name_format.format(
            wandb_run_id=wandb.run.id, **flat_config_dict
        )
        wandb.config.update(config.to_dict())

        run_name = wandb.run.name
    else:
        run_name = None

    if config.save_path is not None:
        run_name = host_broadcast_str(run_name)
        checkpoint_save_path = tf.io.gfile.join(config.save_path, run_name)

        checkpoint_save_manager = ocp.CheckpointManager(
            checkpoint_save_path,
            item_handlers=PaliVLATrainState.get_checkpoint_handlers(),
            options=ocp.CheckpointManagerOptions(),
        )

    wandb_logs = []

    jit_train_step = mesh.sjit(
        train_step,
        in_shardings=(model_sharding, data_sharding),
        out_shardings=(model_sharding, None),
        args_sharding_constraint=(model_sharding, data_sharding),
    )
    jit_eval_step = mesh.sjit(
        eval_step,
        in_shardings=(model_sharding, data_sharding),
        out_shardings=None,
        args_sharding_constraint=(model_sharding, data_sharding),
    )

    # Main training loop
    start_step = model.step.item()
    with tqdm.trange(
        start_step, config.num_steps, desc="Training", dynamic_ncols=True
    ) as pbar:
        for i in pbar:
            batch = next(train_it)
            model, info = jit_train_step(model, batch)

            info = jax.device_get(info)
            wandb_logs.append(info)
            pbar.set_postfix(
                loss=f"{info['mse']:.4f}",
            )

            if (i + 1) % config.log_interval == 0:
                avg_info = jax.tree.map(
                    lambda *xs: np.mean(np.stack(xs), axis=0), *wandb_logs
                )
                if jax.process_index() == 0:
                    wandb.log({"train": avg_info}, step=i)
                wandb_logs = []

            if (i + 1) % config.eval_interval == 0:
                eval_batch = next(eval_it)
                eval_info = jit_eval_step(model, eval_batch)

                if jax.process_index() == 0:
                    wandb.log(
                        {"eval": eval_info},
                        commit=False,
                        step=i,
                    )

            if (i + 1) % config.save_interval == 0:
                if config.save_path is not None:
                    print(f"Saving model to {config.save_path}/{i}")
                    checkpoint_save_manager.save(i + 1, args=model.save_args())


if __name__ == "__main__":
    config_flags.DEFINE_config_file(
        "config", "bridge_config.py", "Path to the config file."
    )
    app.run(main)
