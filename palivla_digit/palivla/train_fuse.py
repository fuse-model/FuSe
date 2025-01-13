from absl import app, flags
from absl import logging as absl_logging
import flax
import jax
import jax.numpy as jnp
from ml_collections import config_flags
import numpy as np
from octo.data.dataset import make_single_dataset
import orbax.checkpoint as ocp
from palivla.dataset import (
    make_base_dataset,
    make_base_dataset_digit,
    make_base_single_dataset,
    transform_dataset,
)
from palivla.load_model import make_optimizer
from palivla.spec import OptimizerSpec
from palivla.train_state import PaliVLATrainState
from palivla.train_step import TrainingBatch
from palivla.utils import host_broadcast_str
from scalax.sharding import FSDPShardingRule, MeshShardingHelper, PartitionSpec
import tensorflow as tf
import tqdm
import wandb

jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
tf.config.set_visible_devices([], "GPU")


def main(_):
    jax.distributed.initialize()
    # Turn off debug logs
    tf.get_logger().setLevel("WARNING")
    absl_logging.set_verbosity(absl_logging.WARNING)

    tf.random.set_seed(jax.process_index())

    config = flags.FLAGS.config

    # Setup mesh and sharding for model and data
    mesh = MeshShardingHelper([-1], ["fsdp"])

    model_sharding = FSDPShardingRule("fsdp", fsdp_axis_size=mesh.mesh.shape["fsdp"])
    data_sharding = PartitionSpec("fsdp")

    # Make the basic dataset
    # We have to do this first, since we need to know how the dataset is set up before we can construct the model
    train_ds = make_base_dataset_digit(**config.dataset_kwargs, train=True)
    eval_ds = make_base_dataset_digit(**config.dataset_kwargs, train=False)
    batch_shape = {
        "text": jax.ShapeDtypeStruct(shape=(1, 10), dtype=jnp.int32),
        "image_primary": jax.ShapeDtypeStruct(shape=(1, 224, 224, 3), dtype=jnp.uint8),
        "image_wrist": jax.ShapeDtypeStruct(shape=(1, 224, 224, 3), dtype=jnp.uint8),
        "image_digit_left": jax.ShapeDtypeStruct(
            shape=(1, 224, 224, 3), dtype=jnp.float32
        ),
        "image_digit_right": jax.ShapeDtypeStruct(
            shape=(1, 224, 224, 3), dtype=jnp.float32
        ),
        "mel_spectro": jax.ShapeDtypeStruct(shape=(1, 224, 224, 3), dtype=jnp.float32),
        "proprio": jax.ShapeDtypeStruct(shape=(1, 6), dtype=jnp.float32),
        "modality_idx": jax.ShapeDtypeStruct(shape=(1, 1), dtype=jnp.int32),
    }
    if (
        config.resume_from_checkpoint_dir is not None
    ):  # load in pretrained model and merge in weights that exist already
        restore_checkpoint_manager = ocp.CheckpointManager(
            config.resume_from_checkpoint_dir,
            item_handlers=PaliVLATrainState.get_checkpoint_handlers(),
        )
        (
            loaded_model,
            pretrained_action_tokenizer_state,
            loaded_language_tokenizer,
        ) = PaliVLATrainState.load_components(
            checkpoint_manager=restore_checkpoint_manager,
            step=config.resume_from_checkpoint_step,
            load_optimizer=True,
            mesh=mesh,
            model_sharding=model_sharding,
            data_sharding=data_sharding,
        )
        pretrained_params = loaded_model.params
    else:
        pretrained_params = None
        pretrained_action_tokenizer_state = None
        loaded_language_tokenizer = None
    action_shape = (train_ds.element_spec["action"]).shape
    action_dim = action_shape[-1]
    action_horizon = action_shape[-2]
    model = PaliVLATrainState.from_components(
        paligemma_weights_path=config.paligemma_weights_path,
        action_tokenizer_weights_path=config.action_tokenizer_path,
        language_tokenizer_path=config.language_tokenizer_path,
        config=config.model_config.to_dict(),
        dataset_statistics=train_ds.dataset_statistics,
        language_tokenizer=None,
        optimizer_spec=OptimizerSpec.create(
            make_optimizer, config.optimizer_kwargs.to_dict()
        ),
        model_sharding=model_sharding,
        data_sharding=data_sharding,
        mesh=mesh,
        seed=config.get("seed", 0),
        param_dtype=jnp.float32,
        batch_shape=batch_shape,
        action_dim=action_dim,
        action_horizon=action_horizon,
        pretrained_params=pretrained_params,
        pretrained_action_tokenizer_state=pretrained_action_tokenizer_state,
        loaded_language_tokenizer=loaded_language_tokenizer,
    )

    # Construct the final dataset
    # We need to do this after the model is constructed, since we need to have a tokenizer
    per_host_train_batch_size = config.batch_size // jax.process_count()
    per_host_eval_batch_size = config.eval_batch_size // jax.process_count()

    def make_training_batch(batch):
        sensors = {
            k: batch["observation"][k]
            for k in batch["observation"]
            if k in model.model_state.model.modality_mappings and k != "text"
        } | {
            "modality_idx": batch["modality_idx"][:, None],
        }
        sensors_mask = {
            k: np.squeeze(batch["observation"]["pad_mask_dict"][k], axis=-1)
            for k in model.model_state.model.modality_mappings
            if k != "text" and k != "modality_idx"
        }  # modality_idx mask added later depending on whether in a fuse step or not

        modal_mask = {
            k: np.squeeze(batch["observation"]["modal_pad_mask_dict"][k], axis=-1)
            for k in model.model_state.model.modality_mappings
            if k != "text" and k != "modality_idx"
        }

        return mesh.local_data_to_global_array(
            TrainingBatch(
                sensors=sensors,
                sensors_mask=sensors_mask,
                modal_mask=modal_mask,
                actions=batch["action"],
                actions_mask=batch["action_pad_mask"],
                tokens=batch["tokens"],
                tokens_ar=batch["mask_ar"],
                tokens_loss=batch.get("mask_loss", None),
                tokens_ar_fuse=batch["mask_ar_fuse"],
                tokens_loss_fuse=batch.get("mask_loss_fuse", None),
                tokens_mask=batch["mask_input"],
                modality_idx=batch["modality_idx"],
                mic_mask=batch["mic_mask"],
            )
        )

    def make_gen_batch(batch):
        sensors = {
            k: batch["observation"][k]
            for k in batch["observation"]
            if k in model.model_state.model.modality_mappings and k != "text"
        } | {
            "modality_idx": batch["modality_idx"][:, None],
        }
        sensors_mask = {
            k: np.squeeze(batch["observation"]["pad_mask_dict"][k], axis=-1)
            for k in model.model_state.model.modality_mappings
            if k != "text" and k != "modality_idx"
        } | {
            "modality_idx": jnp.zeros_like(batch["modality_idx"], dtype=jnp.bool_),
        }
        return mesh.local_data_to_global_array(
            TrainingBatch(
                sensors=sensors,
                sensors_mask=sensors_mask,
                actions=batch["action"],
                actions_mask=batch["action_pad_mask"],
                tokens=batch["tokens"],
                tokens_ar=batch["mask_ar"],
                tokens_loss=batch.get("mask_loss", None),
                tokens_mask=batch["mask_input"],
            )
        )

    train_it = map(
        make_training_batch,
        transform_dataset(
            train_ds,
            model.tokenizer,
            generation=False,
            **config.extra_dataset_transform_kwargs,
        )
        .batch(per_host_train_batch_size)
        .iterator(),
    )

    gen_train_it = map(
        make_gen_batch,
        transform_dataset(
            train_ds,
            model.tokenizer,
            generation=True,
            **config.extra_dataset_transform_kwargs,
        )
        .batch(per_host_train_batch_size)
        .iterator(),
    )

    gen_eval_it = map(
        make_gen_batch,
        transform_dataset(
            eval_ds,
            model.tokenizer,
            generation=True,
            **config.extra_dataset_transform_kwargs,
        )
        .batch(per_host_eval_batch_size)
        .iterator(),
    )

    # W&B setup
    if jax.process_index() == 0:
        wandb_kwargs = {"project": config.wandb_project, "tags": [config.data_mix]}

        if "wandb_run_name" in config.to_dict() and config.wandb_run_name is not None:
            wandb_kwargs["name"] = config.wandb_run_name

        wandb.init(**wandb_kwargs)
        wandb.config.update(config.to_dict())

        run_name = wandb.run.name
    else:
        run_name = None

    run_name = host_broadcast_str(run_name)
    checkpoint_save_path = tf.io.gfile.join(config.save_path, run_name)

    checkpoint_save_manager = ocp.CheckpointManager(
        checkpoint_save_path,
        item_handlers=PaliVLATrainState.get_checkpoint_handlers(),
        options=ocp.CheckpointManagerOptions(),
    )

    wandb_logs = []

    # Main training loop
    start_step = model.model_state.step.item()
    with tqdm.trange(
        start_step, config.num_steps, desc="Training", dynamic_ncols=True
    ) as pbar:
        for i in pbar:
            batch = next(train_it)
            info = model.train_step(batch)

            info = jax.device_get(info)
            wandb_logs.append(info)
            pbar.set_postfix(
                loss=f"{info['loss']:.4f}",
            )

            if (i + 1) % config.log_interval == 0:
                avg_info = jax.tree.map(
                    lambda *xs: np.mean(np.stack(xs), axis=0), *wandb_logs
                )
                if jax.process_index() == 0:
                    wandb.log(avg_info, step=i)
                wandb_logs = []

            if (i + 1) % config.eval_interval == 0:
                eval_info = {}
                eval_batch = next(gen_eval_it)
                eval_info = model.eval_step(
                    eval_batch, "eval/gen_", include_regular_stats=False
                )

                train_batch_for_eval = next(gen_train_it)
                train_info = model.eval_step(
                    train_batch_for_eval, "train/gen_", include_regular_stats=False
                )

                if jax.process_index() == 0:

                    wandb.log(
                        eval_info | train_info,
                        commit=False,
                        step=i,
                    )

            if (i + 1) % config.save_interval == 0:
                if config.save_path is not None:
                    print(f"Saving model to {config.save_path}/{i}")
                    checkpoint_save_manager.save(i + 1, args=model.save_args())
    checkpoint_save_manager.wait_until_finished()


if __name__ == "__main__":
    config_flags.DEFINE_config_file(
        "config", "fuse_config.py", "Path to the config file.", lock_config=False
    )
    app.run(main)
