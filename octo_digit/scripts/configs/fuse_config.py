from ml_collections import ConfigDict
from ml_collections.config_dict import FieldReference, placeholder
from octo.model.components.action_heads import MSEActionHead
from octo.model.components.tokenizers import ImageTokenizer, ImageTokenizerConcatTokens
from octo.model.components.vit_encoders import ResNet26FILM
from octo.utils.spec import ModuleSpec
from octo.model.components.language_reconstruction_heads import CLIPContrastiveHead, SingleHeadContinuousGenerationHead
from octo.data.utils.text_processing import HFTokenizer
from octo.utils.train_utils import tvl_loader
from octo.model.components.tvl_vit import tvlViT

def get_dataset_kwargs():
    return {
        "name": placeholder(str),  
        "data_dir": placeholder(str),
        "image_obs_keys": {
            'primary': 'image_0', 
            'wrist': 'image_1',
            'digit_left': 'digit_0',
            'digit_left_background': 'digit_0_background',
            'digit_right': 'digit_1',
            'digit_right_background': 'digit_1_background'
        },
        "proprio_obs_key": None,
        "sensor_obs_keys": {
            'mel_spectro': 'mel_spectro',
        },
        "language_key": "language_instruction",
        # We want to avoid normalizing the gripper
        "action_normalization_mask": [True, True, True, True, True, True, False],
        # standardize_fn is dynamically loaded from a file
        "standardize_fn": ModuleSpec.create(
            "octo.data.oxe.oxe_standardization_transforms:bridge_dataset_transform",
        ),
        
        # If the default data loading speed is too slow, try these:
        # "num_parallel_reads": 8,  # for reading from disk / GCS
        # "num_parallel_calls": 16,  # for initial dataset construction
    }

def get_config():
    mode, task = "full,language_conditioned".split(",")
    assert task in ["image_conditioned", "language_conditioned", "multimodal"]
    assert mode in ["full", "head_only", "head_mlp_only"]
    max_steps = FieldReference(50000)
    window_size = FieldReference(default=2)

    config = dict(
        text_processor=ModuleSpec.create(
            HFTokenizer,
            tokenizer_name="t5-base",
            encode_with_model=False,
            tokenizer_kwargs={
                "max_length": 24,
                "padding": "max_length",
                "truncation": True,
                "return_tensors": "np",
            },
        ), 
        pretrained_loaders=[
            ModuleSpec.create(
                tvl_loader,
                restore_path=placeholder(str), 
                verbose=True
            ),
        ],
        pretrained_path="hf://rail-berkeley/octo-small-1.5", 
        batch_size=1024,
        shuffle_buffer_size=10000,
        num_steps=max_steps,
        log_interval=100,
        eval_interval=500000,
        save_interval=5000,
        lang_interval=10000,
        save_dir=placeholder(str),
	    seed=42,
        wandb=dict(
            project="octo", group=placeholder(str), entity=placeholder(str)
        ),
        dataset_kwargs=get_dataset_kwargs(),
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
            frozen_keys=["*hf_model*",],
            grad_accumulation_steps=None,  # if you are using grad accumulation, you need to adjust max_steps accordingly
        ),
        val_kwargs=dict(
            val_shuffle_buffer_size=1000,
            num_val_batches=16,
        ),
        gen_kwargs=dict(
            val_shuffle_buffer_size=1000,
            num_val_batches=16,
        ),
        mask_invalid_language=False,
    )


    traj_transform_kwargs = dict(
        window_size=window_size,
        action_horizon=4,
        goal_relabeling_strategy=None,
        fuse_augment_strategy="fuse_augmentation",
        fuse_augment_kwargs=dict(
            rephrase_prob=0.5,
            modal_file_path=placeholder(str),
            rephrase_file_path=placeholder(str),
        ),

        task_augment_strategy="delete_task_conditioning",
        task_augment_kwargs=dict(
            keep_image_prob=0.0,
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
            "digit_left": (224, 224),
            "digit_right": (224, 224),
            "digit_left_background": (224, 224),
            "digit_right_background": (224, 224),
        },
        image_augment_kwargs = { 
            "primary": workspace_augment_kwargs, 
            "wrist": wrist_augment_kwargs, 
        }, 
        background_subtraction_map = {
            'image_digit_left': 'image_digit_left_background', 
            'image_digit_right': 'image_digit_right_background',
        },
    )

    # If the default data loading speed is too slow, try these:
    config[
        "frame_transform_threads"
    ] = 16  # for the most CPU-intensive ops (decoding, resizing, augmenting)
    config["traj_transform_kwargs"] = traj_transform_kwargs
    config["frame_transform_kwargs"] = frame_transform_kwargs

    config['update_config'] = ConfigDict({
        'model': {
            'repeat_task_tokens': True,
            'heads': {
                'action': ModuleSpec.create(
                    MSEActionHead,
                    readout_key="readout_action",
                    use_map = True, # should this be disabled? 
                    action_horizon=4,
                    action_dim=7
                ),
                'clip': ModuleSpec.create(
                    CLIPContrastiveHead,
                    readout_key="readout_language",
                    use_map = True,
                    n_lang_tokens=24,
                ),
                'gen': ModuleSpec.create(
                    SingleHeadContinuousGenerationHead,
                    n_lang_tokens=24,  
                ),
            },
            'readouts': {
                'language': 24,
            },
            'observation_tokenizers': {
                'mel_spectro':  ModuleSpec.create(
                    ImageTokenizer,
                    obs_stack_keys=['mel_spectro'],
                    task_stack_keys=[],
                    encoder=ModuleSpec.create(ResNet26FILM, use_film=False),
                    add_channel_dim=True,
                ), 
                "digits":  ModuleSpec.create(
                    ImageTokenizerConcatTokens,
                    obs_stack_keys=["image_digit_left", "image_digit_right"],
                    task_stack_keys=[],
                    encoder=ModuleSpec.create(tvlViT, img_size=(224, 224)),
                ),
            }

        }
    })
    
    config['config_delete_keys'] = {
        'model': {
            'heads': {
                'action': {
                    'kwargs': {
                        'n_diffusion_samples': True,
                        'dropout_rate': True
                    }
                }
            }
        }
    }

    return ConfigDict(config)
