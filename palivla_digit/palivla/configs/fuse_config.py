from ml_collections.config_dict import placeholder, ConfigDict, FieldReference

from palivla.model import get_default_config
from palivla.spec import ModuleSpec
from palivla.utils import freeze_structure
from flax.core import FrozenDict
import jax.numpy as jnp

placeholder(int)._value


def get_config():
    num_train_steps = FieldReference(40_000, int)

    from big_vision.models.ppp.gemma import get_config
    variant = "gemma_2b"
    llm_embdim = get_config(variant).width
    model_config = get_default_config()

    model_config['prompt_autoregressive'] = False

    model_config["modality_mappings"] = {
        "image_primary": "img",
        "image_wrist": "img",
        "image_digit_left": "digit_left",
        "image_digit_right": "digit_right",
        "mel_spectro": "spectro",
        "text": "llm",
        "modality_idx": "modality_embedder"
    }
    
    model_config["target_key_order"] = (
        "image_primary", "image_wrist", "image_digit_left", "image_digit_right", "mel_spectro", "modality_idx"
    )
    
    model_config['encoder_specs']['digit_left'] = {
        "__ctor": "palivla.tactile_encoder_pooled.tvlViT",
        "config": {
            "num_classes": llm_embdim,
            "pool_type": "cross_attn",
            "load_fn": "palivla.utils.load_tvl_weights",
            "load_kwargs": FrozenDict({
                "pretrained_path": placeholder(str), 
            }),
        },
       
    }
    
    model_config['encoder_specs']['digit_right'] = {
        "__ctor": "palivla.tactile_encoder_pooled.tvlViT",
        "config": {
            "num_classes": llm_embdim,
            "pool_type": "cross_attn",
            "load_fn": "palivla.utils.load_tvl_weights",
            "load_kwargs": FrozenDict({
                "pretrained_path": placeholder(str), 
            }),
        },
    }
    
    model_config['encoder_specs']['spectro'] = {
        "__ctor": "octo.model.components.vit_encoders.ResNet26FILM",
        "config": {
            "use_film": False,
            "num_classes": llm_embdim,
            "flatten_result": True,
            "normalize_input": False,
        }
    }
    
    model_config['encoder_specs']['modality_embedder'] = {
        "__ctor": "octo.palivla.modality_embedder.ModalityEmbedder",
        "config": {
            "num_embeddings": 9,
            "embedding_dim": llm_embdim,
            "dtype_str": "float32",
        }
    }


    dataset_kwargs = {
        "dataset_kwargs_list": [{
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
            "proprio_obs_key": "proprio",
            "sensor_obs_keys": {
                'mel_spectro': 'mel_spectro',
            },
            "language_key": "language_instruction",
            # We want to avoid normalizing the gripper
            # "action_normalization_mask": [True, True, True, True, True, True, False],
            # standardize_fn is dynamically loaded from a file
            # for example: "experiments/kevin/custom_standardization_transforms.py:aloha_dataset_transform"
            "standardize_fn": {'module':"octo.data.oxe.oxe_standardization_transforms", 
                "name": "bridge_dataset_transform",
                "args": [],
                "kwargs": {},
            },

            
            # If the default data loading speed is too slow, try these:
            # "num_parallel_reads": 8,  # for reading from disk / GCS
            # "num_parallel_calls": 16,  # for initial dataset construction
        }],
        "traj_transform_kwargs": dict(
            window_size=1,
            action_horizon=1,
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
        ),
        
        "frame_transform_kwargs": {
            "image_augment_kwargs": {
                "primary": dict(
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
                ),
                    "wrist": dict(
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
                ),
            },
            "resize_size": {
                'primary': [224, 224], 
                'wrist': [224, 224],
                'digit_left': [224, 224],
                'digit_left_background': [224, 224],
                'digit_right': [224, 224],
                'digit_right_background': [224, 224]
            },
            "background_subtraction_map": {
                'image_digit_left': 'image_digit_left_background', 
                'image_digit_right': 'image_digit_right_background',
            }
        },
        "balance_weights": True,
        "shuffle_buffer_size": 50_000,
        "traj_transform_threads": 24,
        "traj_read_threads": 24,
    }

    return ConfigDict(
        {
            "wandb_project": "palivla_fuse",
            "paligemma_weights_path": "models/paligemma-3b-mix-224.f16.npz",
            "language_tokenizer_path": "models/paligemma_tokenizer.model",
            "action_tokenizer_path": placeholder(str),
            "model_load_fn": "big_vision.models.proj.paligemma.paligemma.load",
            "tokenizer_path": "models/paligemma_tokenizer.model",
            "model_path": "models/paligemma",
            "save_path": "models/paligemma_saved_model",
            "batch_size": 1024,
            "eval_batch_size": 1024,
            "num_steps": num_train_steps,
            "eval_interval": 1000,
            "save_interval": 2500,
            "log_interval": 1,
            "data_axis_size": 1,
            "fsdp_axis_size": -1,
            "resume_from_checkpoint_dir": placeholder(str), # or remove to start from scratch
            "resume_from_checkpoint_step": placeholder(int),
            "model_config": model_config,
            "dataset_kwargs": dataset_kwargs,
            "extra_dataset_transform_kwargs": {
                "multimodal_rephrasings": True,
                "chunk_relative_actions": False,
                "proprio_dropout_prob": 0.0,
                "require_language": False,
            },
            "optimizer_kwargs": {
                "optimizer": "adamw",
                "num_steps": num_train_steps,
                "llm_optimizer_kwargs": {
                    "init_learning_rate": 0,
                    "learning_rate": 1e-5,
                    "warmup_steps": 2000,
                    "weight_decay": 5e-6,
                    "grad_norm_clip": 10.0,
                    "b1": 0.9,
                    "b2": 0.99,
                },
                "embed_optimizer_kwargs": {
                    "init_learning_rate": 0,
                    "learning_rate": 1e-5,
                    "warmup_steps": 1000,
                    "weight_decay": 0.0,
                    "grad_norm_clip": 10.0,
                    "b1": 0.9,
                    "b2": 0.99,
                },
                "img_optimizer_kwargs": {
                    "init_learning_rate": 0,
                    "learning_rate": 1e-5,
                    "warmup_steps": 2000,
                    "weight_decay": 5e-6,
                    "grad_norm_clip": 10.0,
                    "b1": 0.9,
                    "b2": 0.99,
                },
            },
        }
    )
