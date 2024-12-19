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
from octo.model.components.language_reconstruction_heads import CLIPContrastiveHead

from typing import Union

from ml_collections import ConfigDict
from ml_collections.config_dict import FieldReference, placeholder

from octo.data.utils.text_processing import MuseEmbedding
from octo.model.components.action_heads import MSEActionHead
from octo.model.components.tokenizers import ImageTokenizer
from octo.model.components.transformer import common_transformer_sizes
from octo.model.components.vit_encoders import ResNet26FILM
from octo.utils.spec import ModuleSpec
from octo.model.components.bcz_language_reconstruction_heads import BczCosineFeaturesLanguageHead


def get_model_config():
    size, kwargs = common_transformer_sizes('vit_s')
    return dict(
        observation_tokenizers=dict(
            primary=ModuleSpec.create(
                ImageTokenizer,
                obs_stack_keys=["image_primary"],
                task_stack_keys=[],
                encoder=ModuleSpec.create(SmallStem16, use_film=False),
            ),
            wrist=ModuleSpec.create(
                ImageTokenizer,
                obs_stack_keys=["image_wrist"],
                task_stack_keys=[],
                encoder=ModuleSpec.create(SmallStem16, use_film=False),
            ),
            digits=ModuleSpec.create(
                    ImageTokenizer,
                    obs_stack_keys=['image_digit_left', 'image_digit_right'],
                    task_stack_keys=[],
                    encoder=ModuleSpec.create(SmallStem16, use_film=False),
                )
        ),
        task_tokenizers=dict(
            language=ModuleSpec.create(
                LanguageTokenizer,
                encoder="t5-base",
                finetune_encoder=False,
            ),
        ),
        heads=dict(
            action= ModuleSpec.create(
                MSEActionHead,
                readout_key="readout_action",
                use_map = True, # should this be disabled? 
                action_horizon=4,
                action_dim=7
            ),
            language_all_lang_0= ModuleSpec.create(
                CLIPContrastiveHead, 
                readout_key="readout_language",
                use_map = True
            ) 
        ),
        max_horizon=10,
        readouts = {
            'action': 1, 
            'language': 16
        }, 
        token_embedding_size=size, 
        transformer_kwargs=kwargs,
        use_correct_attention=True,
    )



def get_config():
    NAME='42.0.0'
    img_obs_keys = {
        'primary': 'image_0',
        'wrist': 'image_1',
        'digit_left': 'digit_0', 
        'digit_right': 'digit_1',
        'digit_left_background': 'digit_0_background',
        'digit_right_background': 'digit_1_background',
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
    digit_augment_kwargs = dict(
        random_resized_crop=dict(scale=[0.8, 1.0], ratio=[0.9, 1.1]),
        augment_order=[
            "random_resized_crop"
        ],
    )
    BACKGROUND_SUBTRACT_MAP = {}
    BACKGROUND_SUBTRACT_MAP['image_digit_left'] = 'image_digit_left_background'
    BACKGROUND_SUBTRACT_MAP['image_digit_right'] = 'image_digit_right_background'
    frame_transform_kwargs = dict(
        resize_size={
            "primary": (256, 256),  # workspace (3rd person) camera is at 256x256
            "wrist": (128, 128),   # wrist camera is at 128x128
            "digit_left": (128, 128), #(128, 128),
            "digit_right": (128, 128),  # (128, 128)
            'digit_left_background': (128, 128), 
            'digit_right_background': (128, 128) 
        },
        image_augment_kwargs = { 
            "primary": workspace_augment_kwargs, 
            "wrist": wrist_augment_kwargs, 
            "digit_left": digit_augment_kwargs,
            "digit_right": digit_augment_kwargs,
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
            # wandb_resume_id='joshuajones/octo/scratch_cont_frozen_20240806_010949',
            # wandb_resume_step=184_200,
            reconstruction_loss_weight=3,
            batch_size=128,
            modality='language_conditioned',
            modalities=['cams', 'digits'], 
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
                frozen_keys=('*hf_model*',),
            ),
            prefetch_num_batches=0,
            start_step=0,
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
            text_processor=ModuleSpec.create(
                HFTokenizer,
                tokenizer_name="t5-base",
                encode_with_model=False,
                tokenizer_kwargs={
                    "max_length": 16,
                    "padding": "max_length",
                    "truncation": True,
                    "return_tensors": "np",
                },
            ),
            pretrained_loaders=(
                ModuleSpec.create(
                    hf_weights_loader,
                    hf_model="t5-base",
                ),
            ),
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