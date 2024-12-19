from copy import deepcopy
import imp
import os

from ml_collections import ConfigDict, FieldReference

get_base_config = imp.load_source(
    "config", "scripts/configs/config.py"
).get_config

from octo.model.components.action_heads import DiffusionActionHead
from octo.model.components.tokenizers import ImageTokenizer
from octo.model.components.vit_encoders import SmallStem16
from octo.utils.spec import ModuleSpec
from experiments.homer.sim.wrappers.mujoco import MujocoManipWrapper


def update_config(config, **kwargs):
    updates = ConfigDict(kwargs)
    new_config = deepcopy(config)
    new_config.update(updates)
    return new_config


def get_config(config_string=None):
    config = get_base_config(config_string)

    action_dim = FieldReference(7)

    config["model"]["observation_tokenizers"] = {
        "primary": ModuleSpec.create(
            ImageTokenizer,
            obs_stack_keys=["image_primary"],
            task_stack_keys=["image_primary"],
            encoder=ModuleSpec.create(SmallStem16),
        ),
    }
    config["model"]["task_tokenizers"] = {}
    config["model"]["readouts"] = {"action": 1}
    config["model"]["heads"]["action"] = ModuleSpec.create(
        DiffusionActionHead,
        readout_key="readout_action",
        use_map=False,
        action_horizon=4,
        action_dim=action_dim,
        n_diffusion_samples=1,
    )

    # We augment differently for the primary and wrist cameras
    primary_augment_kwargs = dict(
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

    # ML-collections complains if the type of an existing field changes
    # so we delete and re-add the field

    del config["dataset_kwargs"]["frame_transform_kwargs"]["resize_size"]
    del config["dataset_kwargs"]["frame_transform_kwargs"]["image_augment_kwargs"]
    del config["text_processor"]
    del config["pretrained_loaders"]
    del config["dataset_kwargs"]["traj_transform_kwargs"]["task_augment_kwargs"]

    config["dataset_kwargs"]["frame_transform_kwargs"]["resize_size"] = {
        "primary": (128, 128),  # workspace camera is at 128x128
    }
    config["dataset_kwargs"]["frame_transform_kwargs"]["image_augment_kwargs"] = {
        "primary": primary_augment_kwargs,
    }

    config["val_kwargs"]["modes_to_evaluate"] = ("image_conditioned",)

    base_rollout_vis_kwargs = dict(
        env_name="franka-shoe-pick-place",
        max_episode_length=50,
        exec_horizon=4,
        history_length=2,
        vis_fps=10,
        video_subsample_rate=1,
    )

    config = update_config(
        config,
        num_steps=300000,
        window_size=2,
        optimizer=dict(),
        dataset_kwargs=dict(
            oxe_kwargs=dict(
                data_mix=[("mujoco_manip", 1.0)],
                load_camera_views=("primary",),
                load_depth=False,
                force_recompute_dataset_statistics=False,
            ),
            traj_transform_kwargs=dict(
                action_horizon=4,
                max_action_dim=action_dim,
                task_augment_strategy="delete_task_conditioning",
                task_augment_kwargs=dict(
                    keep_image_prob=1.0,
                ),
            ),
            batch_size=512,
            shuffle_buffer_size=500000,
            balance_weights=True,
        ),
        rollout_kwargs=dict(
            trajs_for_rollouts=20,
            modes_to_evaluate=("image_conditioned",),
            dataset_name="mujoco_manip",
            visualizer_kwargs_list=[
                dict(
                    **base_rollout_vis_kwargs,
                    name="default",
                    env_kwargs=dict(
                        goals_path="gs://rail-tpus-homer-v4/mujoco_rlds/mujoco_manip/eval_goals.npy",
                    ),
                ),
            ],
        ),
        text_processor=None,
        pretrained_loaders=(),
        eval_datasets=["mujoco_manip"],
    )

    return config
