import copy
from copy import deepcopy
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from scripts.configs.config import get_config as get_base_config
from ml_collections import ConfigDict
from ml_collections.config_dict import placeholder

from octo.data.utils.text_processing import HFTokenizer
from octo.model.components.tokenizers import ImageTokenizer, LanguageTokenizer, LowdimObsTokenizer
from octo.model.components.vit_encoders import SmallStem16
from octo.model.components.action_heads import DiffusionActionHead
from octo.utils.spec import ModuleSpec
from octo.utils.train_utils import hf_weights_loader
from octo.data.traj_transforms import zero_out_future_proprio


def update_config(config, **kwargs):
    updates = ConfigDict(kwargs)
    new_config = deepcopy(config)
    new_config.update(updates)
    return new_config


def get_config(config_string="vit_s,no_filter,base_act,no_state,8"):
    config_string, filter, act_frame, state_cond, chunk_length = config_string.split(",")
    base_config = get_base_config(config_string)

    # Can't delete with update_config
    del base_config["model"]["observation_tokenizers"]
    # Field reference can't be updated with update_config
    base_config["window_size"] = 2
    base_config["num_steps"] = 3000000

    #
    # Changes to the model:
    #

    encoder = ModuleSpec.create(SmallStem16)

    base_config["model"]["observation_tokenizers"] = {
        "wrist": ModuleSpec.create(
            ImageTokenizer,
            obs_stack_keys=["image_wrist"],
            task_stack_keys=["image_wrist"],
            task_film_keys=[],
            encoder=encoder,
        ),
        "primary": ModuleSpec.create(
            ImageTokenizer,
            obs_stack_keys=["image_primary"],
            task_stack_keys=["image_primary"],
            task_film_keys=[],
            encoder=encoder,
        ),
        # "secondary": ModuleSpec.create(
        #     ImageTokenizer,
        #     obs_stack_keys=["image_secondary"],
        #     task_stack_keys=["image_secondary"],
        #     task_film_keys=[],
        #     encoder=encoder,
        # ),
    }
    # base_config["model"]["task_tokenizers"] = {
    #     "language": ModuleSpec.create(
    #         LanguageTokenizer,
    #         encoder="t5-base",
    #         finetune_encoder=False,
    #     ),
    # }
    base_config["model"]["readouts"] = {"action": 1}
    base_config["model"]["heads"] = dict(
        action=ModuleSpec.create(
            DiffusionActionHead,
            action_horizon=int(chunk_length),
            action_dim=10,
            readout_key="readout_action",
            hidden_dim=1024,
            diffusion_steps=100,
        ),
    )

    if state_cond == "with_state":
        base_config["model"]["observation_tokenizers"].update({
            "proprio": ModuleSpec.create(
                LowdimObsTokenizer,
                obs_keys=["proprio"],
                n_bins=256,
            ),
        })
        # remove proprio after first step to avoid causal confusion
        base_config["dataset_kwargs"]["traj_transform_kwargs"]["post_chunk_transforms"] = [
            ModuleSpec.create(
                zero_out_future_proprio,
            ),
        ]


    #
    # Changes to data-loading
    #

    # different augmentations for wrist and workspace
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
    exo_augment_kwargs = dict(
        random_resized_crop=dict(scale=[0.8, 1.0], ratio=[0.9, 1.1]),
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

    del base_config["dataset_kwargs"]["frame_transform_kwargs"]["resize_size"]
    del base_config["dataset_kwargs"]["frame_transform_kwargs"]["image_augment_kwargs"]

    base_config["dataset_kwargs"]["frame_transform_kwargs"]["resize_size"] = {
        "wrist": (128, 128),
        "primary": (256, 256),
        # "secondary": (256, 256),
    }
    base_config["dataset_kwargs"]["frame_transform_kwargs"]["image_augment_kwargs"] = {
        "wrist": wrist_augment_kwargs,
        "primary": exo_augment_kwargs,
        # "secondary": exo_augment_kwargs,
    }
    # randomly drop out exterior camera, always keep wrist image for wrist act
    base_config["dataset_kwargs"]["frame_transform_kwargs"]["image_dropout_prob"] = 0.5

    if act_frame == "base_act":
        standardize_fn = ModuleSpec.create(
            "experiments.karl.droid.standardization_transforms:droid_baseact_transform"
        )
    elif act_frame == "wrist_act":
        standardize_fn = ModuleSpec.create(
            "experiments.karl.droid.standardization_transforms:droid_wristact_transform"
        )
        base_config["dataset_kwargs"]["frame_transform_kwargs"]["image_dropout_keep_key"] = "image_wrist"
    elif act_frame == "wrist_act_cumulative":
        standardize_fn = ModuleSpec.create(
            "experiments.karl.droid.standardization_transforms:droid_cumulative_wristact_transform",
            action_horizon=int(chunk_length),
        )
        base_config["dataset_kwargs"]["frame_transform_kwargs"]["image_dropout_keep_key"] = "image_wrist"
    else:
        raise ValueError(f"Action frame {act_frame} not supported.")

    del base_config["dataset_kwargs"]["oxe_kwargs"]
    base_dataset_kwargs = dict(
        image_obs_keys=dict(
            wrist="wrist_image_left",
            primary="exterior_image_1_left",
            # secondary="exterior_image_2_left",
        ),
        proprio_obs_key="proprio",
        language_key="language_instruction*",
        standardize_fn=standardize_fn,
        data_dir="gs://rail-orca-central2",
        ignore_errors=True,
    )

    if filter == "no_filter":
        filter_fcns = []
    elif filter == "success":
        filter_fcns = [
            ModuleSpec.create(
                "experiments.karl.droid.droid_filter_functions:filter_success"
            ),
        ]
    elif filter == "t50":
        filter_fcns = [
            ModuleSpec.create(
                "experiments.karl.droid.droid_filter_functions:filter_task_50"
            ),
        ]
    elif filter == "t100":
        filter_fcns = [
            ModuleSpec.create(
                "experiments.karl.droid.droid_filter_functions:filter_task_100"
            ),
        ]
    elif filter == "t200":
        filter_fcns = [
            ModuleSpec.create(
                "experiments.karl.droid.droid_filter_functions:filter_task_200"
            ),
        ]
    elif filter == "skill8":
        filter_fcns = [
            ModuleSpec.create(
                "experiments.karl.droid.droid_filter_functions:filter_skill_8"
            ),
        ]
    elif filter == "view10k":
        filter_fcns = [
            ModuleSpec.create(
                "experiments.karl.droid.droid_filter_functions:filter_viewpoint_10k"
            ),
        ]
    else:
        raise ValueError(f"Filter {filter} not supported.")

    # delete language conditioning
    base_config["text_encoder"] = None
    base_config["dataset_kwargs"]["traj_transform_kwargs"]["task_augment_strategy"] = None
    del base_config["dataset_kwargs"]["traj_transform_kwargs"]["task_augment_kwargs"]

    config = update_config(
        base_config,
        save_dir="gs://karl-central-2",
        eval_interval=100000,
        save_interval=10000,
        viz_interval=100000,
        log_interval=5000,
        # optimizer=dict(
            # frozen_keys=("*hf_model*",),
            # grad_accumulation_steps=2,
        # ),
        dataset_kwargs=dict(
            dataset_kwargs_list=[
                dict(
                    name="r2_d2",
                    filter_functions=filter_fcns,
                    **base_dataset_kwargs
                ),
            ],
            sample_weights=[1],
            batch_size=256,
            shuffle_buffer_size=200000,
            balance_weights=True,
            traj_transform_kwargs=dict(
                action_horizon=int(chunk_length),
                goal_relabeling_strategy="uniform",
                goal_relabeling_kwargs=dict(
                    max_goal_distance=50,
                )
            ),
        ),
        # text_processor=ModuleSpec.create(
        #     HFTokenizer,
        #     tokenizer_name="t5-base",
        #     encode_with_model=False,
        #     tokenizer_kwargs={
        #         "max_length": 16,
        #         "padding": "max_length",
        #         "truncation": True,
        #         "return_tensors": "np",
        #     },
        # ),
        # pretrained_loaders=(
        #     ModuleSpec.create(
        #         hf_weights_loader,
        #         hf_model="t5-base",
        #     ),
        # ),
        eval_datasets=("r2_d2",),
    )

    return config
