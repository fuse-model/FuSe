from PIL import Image

from absl import app, flags, logging
import flax
import jax
import optax
import tensorflow as tf
import tqdm
import wandb

from octo.data.dataset import make_single_dataset
from octo.data.utils.data_utils import NormalizationType
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

flags.DEFINE_string(
    "pretrained_path", None, "Path to pre-trained Octo checkpoint directory."
)
flags.DEFINE_string("data_dir", None, "Path to finetuning dataset, in RLDS format.")
flags.DEFINE_string("save_dir", None, "Directory for saving finetuning checkpoints.")
flags.DEFINE_integer("batch_size", 64, "Batch size for finetuning.")

flags.DEFINE_bool(
    "freeze_transformer",
    False,
    "Whether pre-trained transformer weights should be frozen.",
)

flags.DEFINE_integer("num_finetuning_steps", 50000, "Number of finetuning steps.")

flags.DEFINE_integer("save_frequency", 5000, "Frequency to save model (~400MB).")


def main(_):
    assert (
        FLAGS.batch_size % jax.device_count() == 0
    ), "Batch size must be divisible by device count."

    initialize_compilation_cache()
    # prevent tensorflow from using GPU memory since it's only used for data loading
    tf.config.set_visible_devices([], "GPU")

    # setup wandb for logging
    wandb.init(name="finetune_octo_test", project="octo")

    # load pre-trained model
    logging.info("Loading pre-trained model...")
    pretrained_model = OctoModel.load_pretrained(FLAGS.pretrained_path)

    logging.info("Loading finetuning dataset...")
    dataset = make_single_dataset(
        dataset_kwargs=dict(
            name="bridge_dataset",
            data_dir=FLAGS.data_dir,
            image_obs_keys={"primary": "image_0"},
            state_obs_keys=["state"],
            language_key="language_instruction",
            action_proprio_normalization_type=NormalizationType.NORMAL,
            absolute_action_mask=[False, False, False, False, False, False, True],
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
    train_data_iter = (
        dataset.repeat()
        .unbatch()
        .shuffle(10000)  # can reduce this if RAM consumption too high
        .batch(FLAGS.batch_size)
        .iterator()
    )

    # run text tokenizer over batch (this needs to happen before training / sharding) + delete unused keys
    text_processor = pretrained_model.text_processor

    def process_batch(batch):
        batch = process_text(batch, text_processor)
        del batch["dataset_name"]
        return batch

    train_data_iter = map(process_batch, train_data_iter)
    example_batch = next(train_data_iter)

    # load pre-training config and modify --> remove wrist cam
    config = pretrained_model.config
    del config["model"]["observation_tokenizers"]["wrist"]

    model = OctoModel.from_config(
        config,
        example_batch,
        text_processor,
        verbose=False,
        dataset_statistics=dataset.dataset_statistics,
    )
    merged_params = merge_params(model.params, pretrained_model.params)
    # can perform any additional parameter surgery here...
    # ...
    model = model.replace(params=merged_params)
    del pretrained_model

    # create optimizer & train_state, optionally freeze keys for pre-trained transformer
    # train_state bundles parameters & optimizers
    learning_rate = optax.join_schedules(
        [optax.linear_schedule(0, 3e-5, 100), optax.constant_schedule(3e-5)], [100]
    )
    tx = optax.adamw(learning_rate)
    frozen_keys = model.config["optimizer"]["frozen_keys"]
    if FLAGS.freeze_transformer:
        frozen_keys.append("BlockTransformer_0")
    tx = freeze_weights(tx, model.params, frozen_keys)
    train_state = TrainState.create(
        rng=jax.random.PRNGKey(1234),
        model=model,
        tx=tx,
    )

    # define loss function and train step
    def loss_fn(params, batch, rng, train=True):
        bound_module = model.module.bind({"params": params}, rngs={"dropout": rng})
        transformer_embeddings = bound_module.octo_transformer(
            batch["observation"],
            batch["task"],
            batch["observation"]["pad_mask"],
            train=train,
        )
        action_loss, action_metrics = bound_module.heads["action"].loss(
            transformer_embeddings,  # Action head knows to pull out the action readout_key
            batch["action"],
            pad_mask=batch["observation"]["pad_mask"],
            train=train,
        )
        return action_loss, action_metrics

    @jax.jit
    def train_step(state, batch):
        rng, dropout_rng = jax.random.split(state.rng)
        (loss, info), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            state.model.params, batch, dropout_rng, train=True
        )
        new_state = state.apply_gradients(grads=grads, rng=rng)
        return new_state, info

    # run finetuning loop
    logging.info("Starting finetuning...")
    num_steps = FLAGS.num_finetuning_steps
    for i in tqdm.tqdm(range(num_steps), total=num_steps, dynamic_ncols=True):
        j = -1
        while True: 
            j += 1
            batch = next(train_data_iter)
            img_obs = batch['observation']['image_primary']
            if not img_obs.any(): 
                print(j)
                continue 
            print("success:   ", j)
            for k1 in range(img_obs.shape[0]): 
                for k2 in range(img_obs.shape[1]): 
                    img_test = img_obs[k1, k2, ...]
                    if img_test.any(): 
                        im = Image.fromarray(img_test)
                        im.save(f"./test_img_{j}_{k1}_{k2}.jpeg")
        
        print(batch)
        print(batch.keys())
        for key, val in batch.items(): 
            if isinstance(val, dict): 
                for k, v in val.items(): 
                    if isinstance(v, dict): 
                        for k2, v2 in v.items(): 
                            print(f"{key}     {k}      {k2}     {v2.shape}")
                    else: 
                        print(f"{key}      {k}     {v.shape}")
            else:   
                print(f"{key}    {val.shape}")
        found_img = False 
        img_obs = batch['observation']['image_primary']
        # for i in range(img_obs.shape[0]):
        #     for j in range(img_obs.shape[1]): 
                # img_test = batch['observation']['image_primary'][i, j, ...]
                # im = Image.fromarray(img_test)
                # im.save(f"./test_img_{i}_{j}.jpeg")
                # if img_test.any():
                #     print(i, j)
        # while True: 
            
        
        # exit(0)
        train_state, update_info = train_step(train_state, batch)
        if (i + 1) % 100 == 0:
            update_info = jax.device_get(update_info)
            wandb.log(
                flax.traverse_util.flatten_dict({"training": update_info}, sep="/"),
                step=i,
            )
        if (i + 1) % FLAGS.save_frequency == 0:
            # save checkpoint
            train_state.model.save_pretrained(step=i, checkpoint_path=FLAGS.save_dir)


if __name__ == "__main__":
    app.run(main)
