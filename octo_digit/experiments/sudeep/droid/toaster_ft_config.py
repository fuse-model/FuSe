from ml_collections import ConfigDict
from ml_collections.config_dict import FieldReference, placeholder

from octo.utils.spec import ModuleSpec
from octo.model.components.action_heads import DDPMActionHead


# add octo folder to PYTHONPATH for importing standardization function
import octo, os, sys
sys.path.append(os.path.join(os.path.dirname(octo.__file__), "../"))


def get_config(config_string="4,0,rel_act"):
    pred_horizon, grad_accum, act_type = config_string.split(',')

    # hard-code some constants to match the original configs
    task = 'language_conditioned'
    mode = 'full'

    assert task in ["image_conditioned", "language_conditioned", "multimodal"]
    assert mode in ["full", "head_only", "head_mlp_only"]

    # Fill this in for your own dataset!

    # There should be two image keys
    # first image key should be the third-person view (None if not used)
    # and second image key should be the wrist view (None if not used)

    if act_type == "abs_act":
        n_act_dims = 10
        standardize_fn = "experiments.sudeep.droid.standardization_transforms:droid_dataset_transform"
    elif act_type == "rel_act":
        n_act_dims = 10
        standardize_fn = "experiments.sudeep.droid.standardization_transforms:droid_rel_dataset_transform"
    else:
        raise ValueError

    FINETUNING_KWARGS = {
        "name": "r2_d2_toaster3_cmu_rgb",
        "data_dir": "/scratch2/sudeep/tfds", #"gs://rail-orca-central2",
        "image_obs_keys": {"primary": "exterior_image_2_left", "wrist": None},
        "proprio_obs_key": "proprio",
        "language_key": "language_instruction",
        # We don't normalize the gripper
        "action_normalization_mask": [True] * (n_act_dims-1) + [False],
        # standardize_fn is dynamically loaded from a file
        "standardize_fn": ModuleSpec.create(standardize_fn),
        # If the default data loading speed is too slow, try these:
        "num_parallel_reads": 8,  # for reading from disk / GCS
        "num_parallel_calls": 16,  # for initial dataset construction
    }

    if mode == "full":
        frozen_keys = None
    elif mode == "head_only":
        frozen_keys = ("octo_transformer.*",)
    elif mode == "head_mlp_only":
        frozen_keys = (
            "octo_transformer.*",
            "heads_*.map_head.probe",
            "heads_*.map_head.MultiHeadDotProductAttention_0.*",
        )
    elif mode == "frozen_transformer":
        frozen_keys = ("octo_transformer.BlockTransformer_0.*",)
    else:
        raise ValueError("Invalid mode")

    max_steps = 500000
    grad_accum = int(grad_accum)
    if grad_accum:
        batch_size = int(256 // grad_accum)
        max_steps = int(max_steps * grad_accum)
    else:
        grad_accum = None
        batch_size = 256

    max_steps = FieldReference(max_steps)
    window_size = FieldReference(default=1)

    config = dict(
        pretrained_path=placeholder(str),
        pretrained_step=placeholder(int),
        batch_size=batch_size,
        shuffle_buffer_size=50000,
        num_steps=max_steps,
        log_interval=100,
        eval_interval=1e9,  # disable eval due to bugs
        save_interval=20000,
        save_dir=placeholder(str),
        seed=42,
        wandb=dict(
            project="octo_finetune", group=placeholder(str), entity=placeholder(str)
        ),
        dataset_kwargs=FINETUNING_KWARGS,
        modality=task,
        finetuning_mode=mode,
        window_size=window_size,
        optimizer=dict( # copied from diffusion policy
            learning_rate=dict(
                name="cosine",
                init_value=0.0,
                peak_value=1e-4,
                warmup_steps=500,
                decay_steps=max_steps,
                end_value=0.0,
            ),
            weight_decay=1e-6,
            b1=0.95,
            b2=0.999,
            eps=1e-8,
            clip_gradient=1.0,
            frozen_keys=frozen_keys,
            grad_accumulation_steps=grad_accum,  # if you are using grad accumulation, you need to adjust max_steps accordingly
        ),
        val_kwargs=dict(
            val_shuffle_buffer_size=1000,
            num_val_batches=16,
        ),
        viz_kwargs=dict(
            eval_batch_size=128,
            trajs_for_metrics=100,
            trajs_for_viz=8,
            samples_per_state=8,
        ),
    )

    if task == "image_conditioned":
        goal_relabeling_strategy = "uniform"
        keep_image_prob = 1.0
    elif task == "language_conditioned":
        goal_relabeling_strategy = "uniform"
        keep_image_prob = 0.0
    elif task == "multimodal":
        goal_relabeling_strategy = "uniform"
        keep_image_prob = 0.5
    else:
        raise ValueError("Invalid modality")

    pred_horizon = int(pred_horizon)

    traj_transform_kwargs = dict(
        window_size=window_size,
        action_horizon=pred_horizon,
        max_action_dim=10,
        goal_relabeling_strategy=goal_relabeling_strategy,
        task_augment_strategy="delete_task_conditioning",
        task_augment_kwargs=dict(
            keep_image_prob=keep_image_prob,
        ),
        # If the default data loading speed is too slow, try these:
        # num_parallel_calls=16,  # for less CPU-intensive ops
    )
    workspace_augment_kwargs = dict(
        random_resized_crop=dict(scale=[0.8, 1.0], ratio=[0.9, 1.1]),
        random_brightness=[0.1],
        random_contrast=[0.9, 1.1],
        random_saturation=[0.9, 1.1],
        random_hue=[0.05],
        augment_order=[
            "random_resized_crop",
            "random_brightness",
            "random_contrast",
            "random_saturation",
            "random_hue",
        ],
    )
    wrist_augment_kwargs = dict(
        random_brightness=[0.1],
        random_contrast=[0.9, 1.1],
        random_saturation=[0.9, 1.1],
        random_hue=[0.05],
        augment_order=[
            "random_brightness",
            "random_contrast",
            "random_saturation",
            "random_hue",
        ],
    )
    frame_transform_kwargs = dict(
        resize_size={
            "primary": (256, 256),  # workspace (3rd person) camera is at 256x256
            "wrist": (128, 128),  # wrist camera is at 128x128
        },
        image_augment_kwargs={
            "primary": workspace_augment_kwargs,
            "wrist": wrist_augment_kwargs,
        },
    )
    # If the default data loading speed is too slow, try these:
    config[
        "frame_transform_threads"
    ] = 16  # for the most CPU-intensive ops (decoding, resizing, augmenting)

    config["traj_transform_kwargs"] = traj_transform_kwargs
    config["frame_transform_kwargs"] = frame_transform_kwargs

    action_head = ModuleSpec.create(
        DDPMActionHead,
        action_dim=n_act_dims,
        action_horizon=pred_horizon,
        readout_key="readout_action",
        use_map=False,
        flatten_tokens=False,
        max_action=1,
        timesteps=100,
    )
    config["update_config"] = dict(
        model=dict(
            heads=dict(
                action=action_head
            ),
        )
    )
    return ConfigDict(config)
