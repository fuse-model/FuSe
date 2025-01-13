import tensorflow as tf

tf.config.set_visible_devices([], "GPU")

from functools import partial
import json

from absl import app, flags
from flax.training.train_state import TrainState
import jax
from jax.experimental import multihost_utils
import jax.numpy as jnp
from ml_collections import config_flags
import numpy as np
import optax
import orbax.checkpoint as ocp
from palivla.dataset import prepare_image
from palivla.load_model import load_model
from palivla.tokenizer import Tokenizer
from scalax.sharding import FSDPShardingRule, MeshShardingHelper, PartitionSpec
from tensorflow_text import SentencepieceTokenizer


def main(_):
    # jax.distributed.initialize()

    config = flags.FLAGS.config

    with open(config.tokenizer_path, "rb") as f:
        language_tokenizer = SentencepieceTokenizer(f.read())
    tokenizer = Tokenizer.from_tokenizer(language_tokenizer)

    print("Loading model params...")
    model, params, decode = load_model(config, tokenizer)

    # Sharding
    mesh = MeshShardingHelper(
        [config.data_axis_size, config.fsdp_axis_size], ["data", "fsdp"]
    )

    model_sharding = FSDPShardingRule("fsdp", fsdp_axis_size=mesh.mesh.shape["fsdp"])

    optimizer = optax.identity()

    @partial(mesh.sjit, out_shardings=model_sharding)
    def init_fn():
        return TrainState.create(
            apply_fn=model.apply,
            params=model.init(
                jax.random.PRNGKey(0),
                jnp.zeros((1, 224, 224, 3)),
                jnp.zeros((1, 10), dtype=jnp.int32),
                jnp.zeros((1, 10), dtype=jnp.bool_),
            )["params"],
            tx=optimizer,
        )

    train_state = init_fn()

    def _make_restore_args(value, ps: PartitionSpec):
        if isinstance(value, jax.Array):
            return ocp.ArrayRestoreArgs(
                restore_type=jax.Array,
                dtype=value.dtype,
                mesh_axes=ps,
                sharding=value.sharding,
            )
        else:
            raise ValueError(f"Unexpected restore type: {type(value)}")

    params_restore_args = jax.tree_map(
        _make_restore_args, train_state.params, model_sharding.apply(train_state.params)
    )

    train_state = train_state.replace(
        params=ocp.CheckpointManager(
            config.resume_from_checkpoint_dir,
            ocp.PyTreeCheckpointer(),
            options=ocp.CheckpointManagerOptions(),
        ).restore(
            config.resume_from_checkpoint_step,
            items={"params": train_state.params},
            restore_kwargs={
                "restore_args": {
                    "params": params_restore_args,
                    "opt_state": ocp.RestoreArgs(),
                    "step": ocp.RestoreArgs(),
                }
            },
        )[
            "params"
        ]
    )

    # Load dataset statistics
    with tf.io.gfile.GFile(
        f"{config.resume_from_checkpoint_dir}/dataset_statistics.json", "r"
    ) as f:
        dataset_statistics = json.load(f)

    with tf.io.gfile.GFile(
        f"{config.resume_from_checkpoint_dir}/config.json", "r"
    ) as f:
        loaded_config = json.load(f)

    action_mean = np.array(dataset_statistics[flags.FLAGS.dataset_name]["action_mean"])
    action_std = np.array(dataset_statistics[flags.FLAGS.dataset_name]["action_std"])

    # Do inference
    def do_inference(images, instructions):
        batch = {
            "observation": {"image_primary": images},
            "task": {"language_instruction": instructions},
        }
        batch = tokenizer.tokenize_language_instruction(batch)
        batch = prepare_image(batch)
        batch = tokenizer.prepare_tokens_for_generation(batch)
        batch = {
            k: v.numpy() if isinstance(v, tf.Tensor) else v for k, v in batch.items()
        }
        batch = mesh.local_data_to_global_array(
            {
                "image": batch["observation"]["image_primary"][:, 0],
                "text": batch["tokens"],
                "mask_ar": batch["mask_ar"],
                "mask_input": batch["mask_input"],
                "_mask": np.ones(batch["tokens"].shape[0], dtype=np.bool_),
            }
        )

        out_tokens = decode(
            {"params": train_state.params},
            batch,
            model=model,
            devices=jax.devices(),
            max_decode_len=tokenizer.config.num_action_tokens,
            replicate_out=True,
            mesh=mesh.mesh,
        )
        out_tokens = jax.device_get(multihost_utils.process_allgather(out_tokens))

        decoded_actions = tokenizer.bin_detokenize(out_tokens)

        # Re-normalize actions using dataset statistics
        decoded_actions = decoded_actions * action_std + action_mean

        return decoded_actions

    images = tf.zeros((1, 224, 224, 3), dtype=tf.uint8)
    instructions = tf.constant("place the mushroom in the pot")

    decoded_actions = do_inference(images, instructions)
    print(decoded_actions)


if __name__ == "__main__":
    config_flags.DEFINE_config_file(
        "config", "bridge_config.py", "Path to the config file."
    )
    flags.DEFINE_string(
        "dataset_name", "bridge_dataset", "Name of the dataset to use for inference."
    )
    app.run(main)
