from ml_collections import ConfigDict
from ml_collections.config_dict import FieldReference, placeholder
from octo.model.components.action_heads import MSEActionHead
from octo.model.components.tokenizers import BinTokenizer, LowdimObsTokenizer, ImageTokenizer, UnsqueezingImageTokenizer, ProjectionTokenizer, SiglipTokenizer
from octo.model.components.vit_encoders import SmallStem16, ResNet26FILM

from octo.utils.train_utils import resnet_26_loader, tvl_loader
from octo.utils.spec import ModuleSpec
from typing import Iterable
from octo.model.components.language_reconstruction_heads import BCZLanguageHead, CLIPContrastiveHead
from octo.data.utils.data_utils import AnnotationSelectionManager
from octo.model.components.tvl_vit import tvlViT
from octo.data.utils.text_processing import HFTokenizer

<<<<<<< HEAD:scripts/configs/josh_finetune_config.py
def get_config(config_string=None):
    # config_string = "full,language_conditioned"
    config_string = "full,multimodal"
    mode, task = config_string.split(",")
=======

def get_config(config_str=None):
    # mode, task = "full,multimodal".split(',') 
    mode, task = "full,language_conditioned".split(",")
>>>>>>> 49944f3034e9cfb649fadc02c7c0f5d6777510ac:scripts/configs/rephrase_experiments/josh_multi_base.py
    assert task in ["image_conditioned", "language_conditioned", "multimodal"]
    assert mode in ["full", "head_only", "head_mlp_only"]

    # Fill this in for your own dataset!

    # There should be two image keys
    # first image key should be the third-person view (None if not used)
    # and second image key should be the wrist view (None if not used)
<<<<<<< HEAD:scripts/configs/josh_finetune_config.py
    CAMS_ONLY = True 

    FINETUNING_KWARGS = {
        "name": "digit_dataset:8.8.0",
        "data_dir": "gs://619c8f721786ba/",
        "image_obs_keys": {
            "primary": "image_0",
            "wrist": "image_1",
=======

    FROZEN_KEYS = ['*hf_model*']
    LANGUAGE_LENGTH = 24
    TEXT_PROCESSOR = ModuleSpec.create(
        HFTokenizer,
        tokenizer_name="t5-base",
        encode_with_model=False,
        tokenizer_kwargs={
            "max_length": LANGUAGE_LENGTH,
            "padding": "max_length",
            "truncation": True,
            "return_tensors": "np",
>>>>>>> 49944f3034e9cfb649fadc02c7c0f5d6777510ac:scripts/configs/rephrase_experiments/josh_multi_base.py
        },
    )
    UNIFIED_LANG = True # if False, batches during CLIP loss consist of single-type annotations
    REPEAT_TASK_TOKENS = True 
    PRETRAINED_LOADERS = [] 
    DS_NUM = '48.0.0'
    MODALITIES = {'cam_primary', 'cam_wrist'}
    LANGUAGE_KEY = 'all_lang_list' 
    img_obs_keys = {
        'primary': 'image_0', 
        'wrist': 'image_1',
    }

    ANNOTATION_MANAGER_KWARGS = {
        'force_uniform_overall': True,
    } 
    

    FINETUNING_KWARGS = {
        "name": f"digit_dataset:{DS_NUM}",  
        "data_dir": "gs://619c8f721786ba/",
        "image_obs_keys": img_obs_keys,
        "proprio_obs_key": None,
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

    NEW_OBS_TOKENIZERS = {} 
    

    NEW_ACTION_HEAD = ModuleSpec.create(
        MSEActionHead,
        readout_key="readout_action",
        use_map = True, # should this be disabled? 
        action_horizon=4,
        action_dim=7
    )
    
    LANGUAGE_RECONSTRUCTION_HEAD = None

    max_steps = FieldReference(50000)
    window_size = FieldReference(default=2)

    config = dict(
<<<<<<< HEAD:scripts/configs/josh_finetune_config.py
        pretrained_path="hf://rail-berkeley/octo-small",
        batch_size=256,
=======
        text_processor=TEXT_PROCESSOR,
        unified_lang=UNIFIED_LANG, 
        pretrained_loaders=PRETRAINED_LOADERS, 
        modalities=MODALITIES,
        pretrained_path="hf://rail-berkeley/octo-small-1.5", 
        batch_size=128,
>>>>>>> 49944f3034e9cfb649fadc02c7c0f5d6777510ac:scripts/configs/rephrase_experiments/josh_multi_base.py
        shuffle_buffer_size=10000,
        num_steps=max_steps,
        log_interval=100,
        eval_interval=500000,
        save_interval=5000,
        save_dir="gs://619c8f721786ba/octo_ckpts/",
	    seed=42,
        wandb=dict(
            project="octo", group=placeholder(str), entity=placeholder(str)
        ),
        dataset_kwargs=FINETUNING_KWARGS,
        modality=task,
        finetuning_mode=mode,
        window_size=window_size,
        optimizer=dict(
            learning_rate=dict(
                name="cosine",
                init_value=0.0,
                peak_value=3e-4,
                warmup_steps=2000,
                decay_steps=max_steps,
                end_value=0.0,
            ),
            weight_decay=0.01,
            clip_gradient=1.0,
            frozen_keys=FROZEN_KEYS,
            grad_accumulation_steps=None,  # if you are using grad accumulation, you need to adjust max_steps accordingly
        ),
        val_kwargs=dict(
            val_shuffle_buffer_size=1000,
            num_val_batches=16,
        ),
        viz_kwargs=dict(
            eval_batch_size=64,
            trajs_for_metrics=100,
            trajs_for_viz=8,
            samples_per_state=8,
        ),
        gradcam_kwargs=dict( 
            eval_batch_size=4, 
            shuffle_buffer_size=1000, 
            train=False, 
            gradcam_kwargs_list=(
                    ('obs_primary', {'psuedo_loss_type': 'loss'}),
                    ('obs_wrist', {'psuedo_loss_type': 'loss'})
            )
        )
    )

    if task == "image_conditioned":
        goal_relabeling_strategy = "uniform"
        keep_image_prob = 1.0
    elif task == "language_conditioned":
        goal_relabeling_strategy = None
        keep_image_prob = 0.0
    elif task == "multimodal":
        goal_relabeling_strategy = "uniform"
        keep_image_prob = 0.5
    else:
        raise ValueError("Invalid modality")


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
    frame_transform_kwargs = dict(
        resize_size={
            "primary": (256, 256),  # workspace (3rd person) camera is at 256x256
            "wrist": (128, 128),   # wrist camera is at 128x128
        },
        image_augment_kwargs = { 
            "primary": workspace_augment_kwargs, 
            "wrist": wrist_augment_kwargs, 
            "digit_left": digit_augment_kwargs,
            "digit_right": digit_augment_kwargs,
        }, 
        background_subtraction_map={},
    )
    # If the default data loading speed is too slow, try these:
    config[
        "frame_transform_threads"
    ] = 16  # for the most CPU-intensive ops (decoding, resizing, augmenting)

    config["traj_transform_kwargs"] = traj_transform_kwargs
    config["frame_transform_kwargs"] = frame_transform_kwargs
    config["new_obs_tokenizers"] = NEW_OBS_TOKENIZERS

    config['update_config'] = { 
        "model":  {
            "repeat_task_tokens": REPEAT_TASK_TOKENS,
        }
    }
    if NEW_ACTION_HEAD is not None: 
        config['update_config']['model']['heads'] = { 
            'action': ConfigDict(NEW_ACTION_HEAD)
        } 
    config['update_config']['model']['observation_tokenizers'] = NEW_OBS_TOKENIZERS

    if LANGUAGE_RECONSTRUCTION_HEAD is not None: 
        config['update_config']['model']['readouts'] = {'language': LANGUAGE_LENGTH}
    config['reconstruction_loss_weight'] = 3.
    config['lang_head'] = LANGUAGE_RECONSTRUCTION_HEAD



    config['update_config'] = ConfigDict(config['update_config'])
    config['pop_keys'] = [
        ('model', 'heads', 'action', 'kwargs', 'n_diffusion_samples'),
        ('model', 'heads', 'action', 'kwargs', 'dropout_rate')
    ]

    return ConfigDict(config)
