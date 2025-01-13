from ml_collections.config_dict import placeholder, ConfigDict, FieldReference

from palivla.model import get_default_config
from palivla.spec import ModuleSpec

placeholder(int)._value


def get_config():
    data_mix = FieldReference("oxe_magic_soup", str)
    num_train_steps = FieldReference(400000, int)

    # config_str_list = config_str.split("_")

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
    }
    
    model_config["target_key_order"] = (
        "image_primary", "image_wrist", "image_digit_left", "image_digit_right", "mel_spectro",
    )
    
    model_config['encoder_specs']['digit_left'] = {
        "__ctor": "palivla.tactile_encoder.tvlViT",
        "config": {
            "num_classes": llm_embdim,
            
            "load_fn": "palivla.utils.load_tvl_weights",
            "load_kwargs": {
                "pretrained_path": "gs://619c8f721786ba/ported_weights/tvl/tvl_vitbgs_params_jax.npz",
            },
        },
    }
    
    model_config['encoder_specs']['digit_right'] = {
        "__ctor": "palivla.tactile_encoder.tvlViT",
        "config": {
            "num_classes": llm_embdim,
            
            "load_fn": "palivla.utils.load_tvl_weights",
            "load_kwargs": {
                "pretrained_path": "gs://619c8f721786ba/ported_weights/tvl/tvl_vitbgs_params_jax.npz",
            },
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
    
    img_obs_keys = {
        'primary': 'image_0', 
        'wrist': 'image_1',
        'digit_left': 'digit_0',
        'digit_left_background': 'digit_0_background',
        'digit_right': 'digit_1',
        'digit_right_background': 'digit_1_background'
    }
    sensor_obs_keys = {
        'mel_spectro': 'mel_spectro',
        'mic_mask': 'has_mic',
    }
    ANNOTATION_MANAGER_KWARGS = {
        'force_uniform_overall': True,
        'reconstruction_loss_keys': [','.join(string_tuple) for string_tuple in [('visual',), ('tactile',), ('audio',), ('visual', 'tactile'), ('visual', 'audio'), ('tactile', 'audio'), ('visual', 'tactile', 'audio')]]
    } 
    
    DS_NUM = "99.0.0"
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
    traj_transform_kwargs = {
                        "window_size": 1,
                        "action_horizon": 1,
                    }
    frame_transform_kwargs = {
                "image_augment_kwargs": {
                    "primary": workspace_augment_kwargs,
                    "wrist": wrist_augment_kwargs,
                },
                "resize_size": {
                },
                "image_dropout_prob": 0.5,
            }
    dataset_kwargs = {
        "dataset_kwargs_list": [{
            "name": f"digit_dataset:{DS_NUM}",  
            "data_dir": "gs://619c8f721786ba",
            "image_obs_keys": img_obs_keys,
            "proprio_obs_key": "proprio",
            "sensor_obs_keys": sensor_obs_keys,
            "language_key": "rephrase_batch_full",
            # We want to avoid normalizing the gripper
            # "action_normalization_mask": [True, True, True, True, True, True, False],
            # standardize_fn is dynamically loaded from a file
            # for example: "experiments/kevin/custom_standardization_transforms.py:aloha_dataset_transform"
            "standardize_fn": {'module':"octo.data.oxe.oxe_standardization_transforms", 
                "name": "bridge_dataset_transform",
                "args": [],
                "kwargs": {},
            },
            'num_gpt_gen_arg': 20, 
            "annotation_manager_kwargs": ANNOTATION_MANAGER_KWARGS,
            
            # If the default data loading speed is too slow, try these:
            # "num_parallel_reads": 8,  # for reading from disk / GCS
            # "num_parallel_calls": 16,  # for initial dataset construction
        }],
         "traj_transform_kwargs": traj_transform_kwargs,
        "frame_transform_kwargs": frame_transform_kwargs,
        "balance_weights": True,
        "shuffle_buffer_size": 50000,
        "traj_transform_threads": 48,
        "traj_read_threads": 48,
    }
    for k in img_obs_keys:
        frame_transform_kwargs['resize_size'][k] = [224, 224]

    

    return ConfigDict(
        {
            "data_mix": data_mix,
            "wandb_project": "palivla",
            "paligemma_weights_path": "models/paligemma-3b-mix-224.f16.npz",
            "language_tokenizer_path": "models/paligemma_tokenizer.model",
            "action_tokenizer_path": placeholder(str),
            "model_load_fn": "big_vision.models.proj.paligemma.paligemma.load",
            "tokenizer_path": "models/paligemma_tokenizer.model",
            "model_path": "models/paligemma",
            "save_path": "models/paligemma_saved_model",
            "batch_size": 16,
            "eval_batch_size": 16,
            "num_steps": num_train_steps,
            "eval_interval": 100,
            "save_interval": 10000000,
            "log_interval": 1,
            "data_axis_size": 1,
            "fsdp_axis_size": -1,
            "resume_from_checkpoint_dir": placeholder(str),
            "resume_from_checkpoint_step": placeholder(int),
            "model_config": model_config,
            "dataset_kwargs": dataset_kwargs,
            "extra_dataset_transform_kwargs": {
                "multimodal_rephrasings": True,
                "chunk_relative_actions": False,
                "multimodal_rephrasing_kwargs": {
                    "rephrase_prob": 0.5,
                    "num_gpt_gen": 20,
                    "num_modalities": 9,
                },
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
