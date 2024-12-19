from copy import deepcopy
import imp
import os

from ml_collections import ConfigDict

get_base_config = imp.load_source(
    "config", os.path.join(os.path.dirname(__file__), "config.py")
).get_config


# add octo folder to PYTHONPATH for importing standardization function
import octo, os, sys
sys.path.append(os.path.join(os.path.dirname(octo.__file__), "../"))


from octo.model.components.tokenizers import ImageTokenizer, LanguageTokenizer
from octo.model.components.vit_encoders import SmallStem16
from octo.utils.spec import ModuleSpec
from octo.model.components.action_heads import UNetActionHead


def update_config(config, **kwargs):
    updates = ConfigDict(kwargs)
    new_config = deepcopy(config)
    new_config.update(updates)
    return new_config


def get_config(config_string="vit_b"):
    config = get_base_config(config_string)

    action_dim = 10
    pred_horizon=4
    act_type="abs_act"
    window_size=1
    modality='text'

    config["window_size"] = window_size
    config["num_steps"] = 500000
    config["model"]["observation_tokenizers"] = {
        "primary": ModuleSpec.create(
            ImageTokenizer,
            obs_stack_keys=["image_primary"],
            task_stack_keys=["image_primary"],
            encoder=ModuleSpec.create(SmallStem16),
        ),
    }
    config["model"]["task_tokenizers"] = {
        "language": ModuleSpec.create(
            LanguageTokenizer,
            encoder="t5-base",
            finetune_encoder=False,
        ),
    }
    config["model"]["repeat_task_tokens"] = True
    config["model"]["readouts"] = {"action": 1}
    config["model"]["heads"]["action"] = ModuleSpec.create(
        UNetActionHead,
        action_dim=action_dim,
        action_horizon=pred_horizon,
        readout_key="readout_action",
        n_diffusion_samples=1,
        use_map=False,
        flatten_tokens=False,
        max_action=1,
        diffusion_steps=100,
        down_dims=[128, 256, 512]
    )

    # ML-collections complains if the type of an existing field changes
    # so we delete and re-add the field

    del config['dataset_kwargs']
    config['dataset_kwargs'] = get_dataset_config(modality, window_size, pred_horizon, act_type)
    config['eval_datasets'] = ("r2_d2_toaster3_cmu_rgb",)
    config["eval_interval"] = 50000000
    config["viz_interval"] = 50000000
    config["savei_interval"]= 100000

    return config


def get_dataset_config(modality="text", window_size=1, pred_horizon=4, act_type='rel_act'):
    if act_type == "abs_act":
        n_act_dims = 10
        standardize_fn = "experiments.sudeep.droid.standardization_transforms:droid_dataset_transform"
    elif act_type == "rel_act":
        n_act_dims = 10
        standardize_fn = "experiments.sudeep.droid.standardization_transforms:droid_rel_dataset_transform"
    else:
        raise ValueError

    if modality == "multimodal":
        task_augmentation = dict(
            goal_relabeling_strategy = "uniform",
            task_augment_strategy="delete_task_conditioning",
            task_augment_kwargs=dict(
                keep_image_prob=0.5,
            ),
        )
    elif modality == "image_conditioned":
        task_augmentation = dict(
            goal_relabeling_strategy = "uniform",
            task_augment_strategy="delete_task_conditioning",
            task_augment_kwargs=dict(
                keep_image_prob=1.0,
            ),
        )
    elif modality == "text":
        task_augmentation = dict(
            goal_relabeling_strategy = "uniform",
            task_augment_strategy="delete_task_conditioning",
            task_augment_kwargs=dict(
                keep_image_prob=0.0,
            ),
        )
    else:
        raise ValueError(f"Unknown modality {modality}")

    return dict(
            dataset_kwargs_list=[dict(
                        name="r2_d2_toaster3_cmu_rgb",
                        data_dir="gs://rail-orca-central2/",
                        image_obs_keys= {"primary": "exterior_image_2_left"},
                        proprio_obs_key="proprio",
                        action_normalization_mask=[True] * (n_act_dims - 1) + [False],
                        standardize_fn=ModuleSpec.create(standardize_fn),
                        language_key= "language_instruction",

            )],
            sample_weights=[1.0],
            traj_transform_kwargs=dict(
                window_size=window_size,
                action_horizon=pred_horizon,
                # subsample_length=100,
                **task_augmentation,
            ),
            frame_transform_kwargs=dict(
                resize_size=(256, 256),
                image_augment_kwargs=dict(
                    random_resized_crop=dict(scale=[0.8, 1.0], ratio=[0.9, 1.1]),
                    random_brightness=[0.2],
                    random_contrast=[0.8, 1.2],
                    random_saturation=[0.8, 1.2],
                    random_hue=[0.1],
                    augment_order=[
                        "random_resized_crop",
                        "random_brightness",
                        "random_contrast",
                        "random_saturation",
                        "random_hue",
                    ],
                ),
            ),
            batch_size=128,
            shuffle_buffer_size=50000,
            balance_weights=True,
            traj_transform_threads=48,  # shared between all datasets
            traj_read_threads=48,  # shared between all datasets
        )
