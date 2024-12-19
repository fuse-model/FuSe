from copy import deepcopy
import imp
import os

from ml_collections import ConfigDict, FieldReference

get_base_config = imp.load_source(
    "config", os.path.join(os.path.dirname(__file__), "config.py")
).get_config

from octo.data.utils.text_processing import HFTokenizer
from octo.model.components.action_heads import DiffusionActionHead
from octo.model.components.tokenizers import ImageTokenizer, LanguageTokenizer
from octo.model.components.vit_encoders import SmallStem16
from octo.utils.spec import ModuleSpec
from octo.utils.train_utils import hf_weights_loader

from typing import Union

from ml_collections import ConfigDict
from ml_collections.config_dict import FieldReference, placeholder

from octo.data.utils.text_processing import MuseEmbedding
from octo.model.components.bcz_action_heads import MSEBczActionHead, HuberBczActionHead
from octo.model.components.tokenizers import ImageTokenizer
from octo.model.components.transformer import common_transformer_sizes
from octo.model.components.vit_encoders import ResNet26FILM
from octo.utils.spec import ModuleSpec
from octo.model.components.bcz_language_reconstruction_heads import BczCosineFeaturesLanguageHead


def get_model_config():
    return dict(
        observation_tokenizers=dict(
            primary=ModuleSpec.create(
                ImageTokenizer,
                obs_stack_keys=["image_primary"],
                task_film_keys=["language_instruction"],
                encoder=ModuleSpec.create(ResNet26FILM, use_film=True),
            ),
            wrist=ModuleSpec.create(
                ImageTokenizer,
                obs_stack_keys=["image_wrist"],
                task_film_keys=["language_instruction"],
                encoder=ModuleSpec.create(ResNet26FILM, use_film=True),
            ),
        ),
        # task_tokenizers=dict(),
        heads=dict(
            action=ModuleSpec.create(
                HuberBczActionHead,
                action_horizon=4,
                action_dim=7,
            ),
            language=ModuleSpec.create(
                BczCosineFeaturesLanguageHead,
                language_dim=512
            )
        ),
        max_horizon=10,
    )



def get_config():
    NAME='42.0.0'
    img_obs_keys = {
        'primary': 'image_0',
        'wrist': 'image_1'
    }
    LANGUAGE_KEY = 'all_lang_list'
    ANNOTATION_MANAGER_KWARGS = {
        'modality_keep_probabilities': { 
            'visual': 0.7,
            'tactile': 0.7,
            'audio': 0.7,
        },
        'remove_keys': ['', 'audio'], 
        'simple_ratio': 1.0,
        'uniform_modality_dropout': True,
        'reconstruction_loss_keys': ['simple'] # ['simple', 'visual,tactile'] # ['simple']#  ['simple', 'visual,tactile', 'visual,tactile,audio']
    }
    FINETUNING_KWARGS = {
        "name": f"digit_dataset:{NAME}",  
        "data_dir": "gs://619c8f721786ba/",
        "image_obs_keys": img_obs_keys,
        "digit_obs_keys": {}, # TODO: remove this, treating digits just as images
        "proprio_obs_key": None, # "state",
        "sensor_obs_keys": {},
        "language_key": LANGUAGE_KEY,
        "annotation_manager_kwargs": ANNOTATION_MANAGER_KWARGS, 
        # We want to avoid normalizing the gripper
        "action_normalization_mask": [True, True, True, True, True, True, False],
        # standardize_fn is dynamically loaded from a file
        # for example: "experiments/kevin/custom_standardization_transforms.py:aloha_dataset_transform"
        "standardize_fn": ModuleSpec.create(
            "octo.data.oxe.oxe_standardization_transforms:bridge_dataset_transform",
        ),
        
        # If the default data loading speed is too slow, try these:
        # "num_parallel_reads": 8,  # for reading from disk / GCS
        # "num_parallel_calls": 16,  # for initial dataset construction
    }
    
    window_size = FieldReference(default=2)
    goal_relabeling_strategy = None
    keep_image_prob = 0.0
    traj_transform_kwargs = dict(
        window_size=window_size,
        action_horizon=4,
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
            "wrist": (128, 128),   # wrist camera is at 128x128
            # "digit_left": (256, 256), #(128, 128),
            # "digit_right": (256, 256),  # (128, 128)
        },
        image_augment_kwargs = { 
            "primary": workspace_augment_kwargs, 
            "wrist": wrist_augment_kwargs, 
            # "digit_left": digit_augment_kwargs,
            # "digit_right": digit_augment_kwargs,
        }, 
        # background_subtraction_map = BACKGROUND_SUBTRACT_MAP, 
    )
    return ConfigDict(
        dict(
            #wandb_resume_id='joshuajones/octo/bcz_20k_20240805_223159',
            wandb_resume_id='joshuajones/octo/bcz_20k_20240805_223201',
            is_bcz=True, 
            batch_size=128,
            modality='language_conditioned',
            modalities=['cams'], 
            seed=42,
            num_steps=300_000,
            save_dir="gs://619c8f721786ba/octo_ckpts/",
            model=get_model_config(),
            window_size=window_size,
            dataset_kwargs=FINETUNING_KWARGS,
            optimizer=dict(
                learning_rate=dict(
                    name="rsqrt",
                    init_value=0.0,
                    peak_value=3e-4,
                    warmup_steps=2000,
                    timescale=10000,
                ),
                weight_decay=0.1,
                clip_gradient=1.0,
                frozen_keys=tuple(),
            ),
            prefetch_num_batches=0,
            start_step=None,
            log_interval=100,
            eval_interval=5000,
            viz_interval=20000,
            save_interval=5000,
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
            resume_path=placeholder(str),
            text_processor=ModuleSpec.create(MuseEmbedding),
            pretrained_loaders=tuple(),
            wandb=dict(
                project="octo",
                group=placeholder(str),
                entity=placeholder(str),
            ),
            traj_transform_kwargs=traj_transform_kwargs,
            frame_transform_kwargs=frame_transform_kwargs,
            frame_transform_threads=16,
            shuffle_buffer_size=10000,  # shared between all datasets
        )
    )


def get_dataset_config(window_size=1):
    task_augmentation = dict(
        task_augment_strategy="delete_task_conditioning",
        task_augment_kwargs=dict(
            keep_image_prob=0.5,
        ),
    )

    return {
        # oxe_kwargs will generate dataset_kwargs_list and sampling weights
        "oxe_kwargs": dict(
            data_mix=placeholder(Union[str, list]),
            data_dir=placeholder(str),
            load_camera_views=("primary", "wrist"),
            load_depth=False,
        ),
        "traj_transform_kwargs": dict(
            window_size=window_size,
            action_horizon=1,
            goal_relabeling_strategy="uniform",
            subsample_length=100,
            **task_augmentation,
        ),
        "frame_transform_kwargs": dict(
            resize_size=(256, 256),
            image_dropout_prob=0.0,
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
            num_parallel_calls=200,
        ),
        "traj_transform_threads": 48,  # shared between all datasets
        "traj_read_threads": 48,  # shared between all datasets
        "shuffle_buffer_size": 100000,  # shared between all datasets
        "batch_size": 1024,
        "balance_weights": True,
    }





def update_config(config, **kwargs):
    updates = ConfigDict(kwargs)
    new_config = deepcopy(config)
    new_config.update(updates)
    return new_config


# def get_config(config_string=None):
    # config = get_base_config(config_string)

    # action_dim = FieldReference(7)

    # config["model"]["observation_tokenizers"] = {
    #     "primary": ModuleSpec.create(
    #         ImageTokenizer,
    #         obs_stack_keys=["image_primary"],
    #         task_stack_keys=["image_primary"],
    #         encoder=ModuleSpec.create(SmallStem16),
    #     ),
    #     "wrist": ModuleSpec.create(
    #         ImageTokenizer,
    #         obs_stack_keys=["image_wrist"],
    #         task_stack_keys=["image_wrist"],
    #         encoder=ModuleSpec.create(SmallStem16),
    #     ),
    # }
    # config["model"]["task_tokenizers"] = {
    #     "language": ModuleSpec.create(
    #         LanguageTokenizer,
    #         encoder="t5-base",
    #         finetune_encoder=False,
    #     ),
    # }
    # config["model"]["repeat_task_tokens"] = True
    # config["model"]["readouts"] = {"action": 1}
    # config["model"]["heads"]["action"] = ModuleSpec.create(
    #     DiffusionActionHead,
    #     readout_key="readout_action",
    #     use_map=False,
    #     action_horizon=4,
    #     action_dim=action_dim,
    #     n_diffusion_samples=1,
    #     dropout_rate=0.0,
    # )

    # # We augment differently for the primary and wrist cameras
    # primary_augment_kwargs = dict(
    #     random_resized_crop=dict(scale=[0.8, 1.0], ratio=[0.9, 1.1]),
    #     random_brightness=[0.1],
    #     random_contrast=[0.9, 1.1],
    #     random_saturation=[0.9, 1.1],
    #     random_hue=[0.05],
    #     augment_order=[
    #         "random_resized_crop",
    #         "random_brightness",
    #         "random_contrast",
    #         "random_saturation",
    #         "random_hue",
    #     ],
    # )
    # wrist_augment_kwargs = dict(
    #     random_brightness=[0.1],
    #     random_contrast=[0.9, 1.1],
    #     random_saturation=[0.9, 1.1],
    #     random_hue=[0.05],
    #     augment_order=[
    #         "random_brightness",
    #         "random_contrast",
    #         "random_saturation",
    #         "random_hue",
    #     ],
    # )

    # # ML-collections complains if the type of an existing field changes
    # # so we delete and re-add the field

    # del config["dataset_kwargs"]["frame_transform_kwargs"]["resize_size"]
    # del config["dataset_kwargs"]["frame_transform_kwargs"]["image_augment_kwargs"]

    # config["dataset_kwargs"]["frame_transform_kwargs"]["resize_size"] = {
    #     "primary": (256, 256),  # workspace camera is at 256x256
    #     "wrist": (128, 128),  # wrist camera is at 128x128
    # }
    # config["dataset_kwargs"]["frame_transform_kwargs"]["image_augment_kwargs"] = {
    #     "primary": primary_augment_kwargs,
    #     "wrist": wrist_augment_kwargs,
    # }

    # config = update_config(
    #     config,
    #     num_steps=300000,
    #     window_size=2,
    #     optimizer=dict(
    #         frozen_keys=("*hf_model*",),
    #     ),
    #     dataset_kwargs=dict(
    #         oxe_kwargs=dict(
    #             data_mix="oxe_magic_soup",
    #             data_dir="gs://rail-orca-central2/resize_256_256",
    #             load_camera_views=("primary", "wrist"),
    #             load_depth=False,
    #             force_recompute_dataset_statistics=False,
    #         ),
    #         traj_transform_kwargs=dict(
    #             action_horizon=4,
    #             max_action_dim=action_dim,
    #             task_augment_strategy="delete_and_rephrase",
    #             task_augment_kwargs=dict(
    #                 pickle_file_path="gs://rail-datasets-europe-west4/oxe/resize_256_256/paraphrases_oxe.pkl",
    #                 rephrase_prob=0.5,
    #             ),
    #         ),
    #         frame_transform_kwargs=dict(
    #             image_dropout_prob=0.5,
    #         ),
    #         batch_size=128,
    #         shuffle_buffer_size=500000,
    #         balance_weights=True,
    #     ),
    #     text_processor=ModuleSpec.create(
    #         HFTokenizer,
    #         tokenizer_name="t5-base",
    #         encode_with_model=False,
    #         tokenizer_kwargs={
    #             "max_length": 16,
    #             "padding": "max_length",
    #             "truncation": True,
    #             "return_tensors": "np",
    #         },
    #     ),
    #     pretrained_loaders=(
    #         ModuleSpec.create(
    #             hf_weights_loader,
    #             hf_model="t5-base",
    #         ),
    #     ),
    #     eval_datasets=["bridge_dataset"],
    # )

    # return config
