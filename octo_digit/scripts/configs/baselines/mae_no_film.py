from copy import deepcopy
import imp
import os

from ml_collections import ConfigDict, FieldReference

# get_base_config = imp.load_source(
#     "config", os.path.join(os.path.dirname(__file__), "config.py")
# ).get_config

from octo.data.utils.text_processing import HFTokenizer
from octo.model.components.action_heads import DiffusionActionHead
from octo.model.components.tokenizers import ImageTokenizer, LanguageTokenizer, IdentityObsTokenizer
from octo.model.components.vit_encoders import SmallStem16
from octo.utils.spec import ModuleSpec
from octo.utils.train_utils import hf_weights_loader
from octo.utils.train_utils import resnet_26_loader

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
            viztac=ModuleSpec.create(
                IdentityObsTokenizer,
                obs_keys=['uniform']
            ),
            lang=ModuleSpec.create(
                LanguageTokenizer,
                encoder=None,
                repeat_tokens_window=2,
                proper_pad_mask=False,
            ),
            spectro=ModuleSpec.create(
                ImageTokenizer,
                obs_stack_keys=['mel_spectro'],
                task_film_keys=["language_instruction"],
                encoder=ModuleSpec.create(ResNet26FILM, use_film=True),
                repeat_channel_dim=True,
            )
        ),
        heads=dict(
            action=ModuleSpec.create(
                HuberBczActionHead,
                action_horizon=4,
                action_dim=7,
            ),
        ),
        max_horizon=10,
    )


def get_config():
    LANG_HEAD = None
    GEN_HEAD = None
    DS_NAME='54.0.0'
    img_obs_keys = {
        # 'primary': 'image_0', 
        # 'wrist': 'image_1',
        # 'digit_left': 'digit_0',
        # 'digit_left_background': 'digit_0_background',
        # 'digit_right': 'digit_1',
        # 'digit_right_background': 'digit_1_background'
    }
    BACKGROUND_SUBTRACT_MAP = {
            # 'image_digit_left': 'image_digit_left_background', 
            # 'image_digit_right': 'image_digit_right_background',
    }
    LANGUAGE_KEY = 'rephrase_batch_full'
    PRETRAINED_LOADERS = [ModuleSpec.create(resnet_26_loader, restore_path='gs://sudeep_r2d2_experiments/R26_S_32-i21k-300ep-lr_0.001-aug_light1-wd_0.1-do_0.0-sd_0.0.npz')] 
    sensor_obs_keys = {
        'uniform': 'uniform',
        'mel_spectro': 'mel_spectro',
        'mic_mask': 'has_mic',
    }
    REMOVE_TOKENIZERS = []

    ANNOTATION_MANAGER_KWARGS = {
        'force_uniform_overall': True,
        'reconstruction_loss_keys': [','.join(string_tuple) for string_tuple in [('visual',), ('tactile',), ('audio',), ('visual', 'tactile'), ('visual', 'audio'), ('tactile', 'audio'), ('visual', 'tactile', 'audio')]]
    } 
    
    DIGIT_SIZE = (256, 256) 
    FINETUNING_KWARGS = {
        "name": f"digit_dataset:{DS_NAME}",  
        "data_dir": "gs://oier-europe-bucket",
        "image_obs_keys": img_obs_keys,
        "proprio_obs_key": None, # "state",
        "sensor_obs_keys": sensor_obs_keys,
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
    digit_augment_kwargs = dict(
        augment_order=[],
    )
    frame_transform_kwargs = dict(
        resize_size={
            "primary": (256, 256),  # workspace (3rd person) camera is at 256x256
            "wrist": (128, 128),   # wrist camera is at 128x128
            "digit_left": DIGIT_SIZE,
            "digit_right": DIGIT_SIZE, 
        },
        image_augment_kwargs = { 
            "primary": workspace_augment_kwargs, 
            "wrist": wrist_augment_kwargs, 
            # "digit_left": digit_augment_kwargs,
            # "digit_right": digit_augment_kwargs,
        }, 
        background_subtraction_map = BACKGROUND_SUBTRACT_MAP, 
    )
    for key, val in BACKGROUND_SUBTRACT_MAP.items(): 
        if key.startswith('image_'): 
            key = key[len('image_'):]
        if val.startswith('image_'): 
            val = val[len('image_'):]
        frame_transform_kwargs['resize_size'][val] = frame_transform_kwargs['resize_size'][key]
    

    return ConfigDict(
        dict(
            # wandb_resume_id='joshuajones/octo/bcz_20k_20240805_223159',
            # wandb_resume_id='joshuajones/octo/bcz_20k_20240805_223201',
            reconstruction_loss_weight=0, 
            pretrained_loaders=PRETRAINED_LOADERS, 
            lang_head=LANG_HEAD,
            gen_head=GEN_HEAD,
            is_bcz=True, 
            batch_size=1024,
            modality='language_conditioned',
            modalities=['cams'], 
            seed=42,
            num_steps=50000,
            save_dir="gs://619c8f721786ba/octo_ckpts/",
            model=get_model_config(),
            window_size=window_size,
            dataset_kwargs=FINETUNING_KWARGS,
            optimizer=dict(
                learning_rate=dict(
                    name="cosine",
                    init_value=0.0,
                    peak_value=3e-4,
                    warmup_steps=2000,
                    decay_steps=50000,
                    end_value=0.0,
                ),
                weight_decay=0.1,
                clip_gradient=1.0,
                frozen_keys=tuple(),
            ),
            prefetch_num_batches=0,
            start_step=None,
            log_interval=100,
            eval_interval=500000,
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