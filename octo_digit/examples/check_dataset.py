from PIL import Image

from absl import app, flags, logging
import flax
import jax
import optax
import tensorflow as tf
import tqdm
import pdb
# import wandb

from octo.data.dataset import make_single_dataset
# from octo.data.utils.data_utils import NormalizationType
from octo.model.components.action_heads import L1ActionHead
from octo.model.components.tokenizers import LowdimObsTokenizer
from octo.model.octo_model import OctoModel
from octo.utils.jax_utils import initialize_compilation_cache
from octo.utils.spec import ModuleSpec
from octo.utils.train_utils import (
    freeze_weights,
    merge_params,
    process_text,
    TrainState,
)

FLAGS = flags.FLAGS

flags.DEFINE_string("data_dir", None, "Path to finetuning dataset, in RLDS format.")
flags.DEFINE_string("save_dir", None, "Directory for saving finetuning checkpoints.")
flags.DEFINE_integer("batch_size", 256, "Batch size for finetuning.")


def main(_):
    assert (
        FLAGS.batch_size % jax.device_count() == 0
    ), "Batch size must be divisible by device count."

    initialize_compilation_cache()
    # prevent tensorflow from using GPU memory since it's only used for data loading
    tf.config.set_visible_devices([], "GPU")

    # setup wandb for logging
    # wandb.init(name="finetune_octo_test", project="octo")

    # load pre-trained model 
    print("\n\n\nhere1\n\n\n")
    # breakpoint()
    dataset = make_single_dataset(
        dataset_kwargs=dict(
            name="bridge_dataset",
            data_dir=FLAGS.data_dir,
            image_obs_keys={"primary": "image_0"},
            proprio_obs_key="state",
            language_key="language_instruction",
            # action_proprio_normalization_type=NormalizationType.NORMAL,
            # absolute_action_mask=[False, False, False, False, False, False, True],
            action_normalization_mask = [True, True, True, True, True, True, False] # Add normalization mask
        ),
        traj_transform_kwargs=dict(
            window_size=2,
            future_action_window_size=3, 
        ),
        frame_transform_kwargs=dict(
            resize_size={"primary": (256, 256)},
        ),
        train=True,
    )
    # print("\n\n\nhere2\n\n\n")

    train_data_iter = (
        dataset.repeat()
        .unbatch()
        .shuffle(10000)  # can reduce this if RAM consumption too high
        .batch(FLAGS.batch_size)
        .iterator()
    )
    j = -1
    batch = next(train_data_iter)
    print(batch)
    # while True: 
    #     j += 1
    #     batch = next(train_data_iter)
    #     img_obs = batch['observation']['image_primary']
    #     if not img_obs.any(): 
    #         print(j)
    #         print(img_obs)
    #         continue 
    #     print("success:   ", j)
    #     for k1 in range(img_obs.shape[0]): 
    #         for k2 in range(img_obs.shape[1]): 
    #             img_test = img_obs[k1, k2, ...]
    #             if img_test.any(): 
    #                 im = Image.fromarray(img_test)
    #                 im.save(f"./test_img_{j}_{k1}_{k2}.jpeg")



if __name__ == "__main__":
    app.run(main)
