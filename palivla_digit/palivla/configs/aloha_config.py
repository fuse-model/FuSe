from ml_collections.config_dict import ConfigDict, FieldReference, placeholder
from octo.data.utils.data_utils import NormalizationType
from palivla.spec import ModuleSpec

placeholder(int)._value


def get_config():
    from palivla.model import get_default_config

    num_train_steps = FieldReference(100000, int)

    from big_vision.models.ppp.gemma import get_config

    variant = "gemma_2b"
    llm_embdim = get_config(variant).width

    model_config = get_default_config()
    model_config["encoder_specs"]["proprio"] = ModuleSpec.from_name(
        "flax.linen.Dense",
        {"features": llm_embdim},
    )
    model_config["modality_mappings"] = {
        "proprio": "proprio",
        "image_primary": "image_primary",
        "image_secondary": "image_secondary",
        "image_wrist_left": "image_wrist_left",
        "image_wrist_right": "image_wrist_right",
        "text": "llm",
    }

    return ConfigDict(
        {
            "wandb_project": "palivla-aloha",
            "paligemma_weights_path": placeholder(str),
            "language_tokenizer_path": placeholder(str),
            "action_tokenizer_path": placeholder(str),
            "save_path": placeholder(str),
            "batch_size": 192,
            "eval_batch_size": 128,
            "num_steps": num_train_steps,
            "eval_interval": 100,
            "save_interval": 1000,
            "log_interval": 1,
            "data_axis_size": 1,
            "fsdp_axis_size": -1,
            "proprio_dim": 14,
            "num_proprio_tokens": 4,
            "resume_from_checkpoint_dir": placeholder(str),
            "resume_from_checkpoint_step": placeholder(int),
            "image_keys": [
                "image_primary",
                "image_secondary",
                "image_wrist_left",
                "image_wrist_right",
            ],
            "chunk_relative_actions": True,
            "model_config": {
                "palivla_model_config": model_config,
                "prompt_autoregressive": False,
            },
            "dataset_kwargs": {
                "oxe_kwargs": {
                    "data_mix": "aloha",
                    "data_dir": "/data/rlds",
                    "load_camera_views": [
                        "primary",
                        "secondary",
                        "wrist_left",
                        "wrist_right",
                    ],
                    "load_depth": False,
                    "load_proprio": True,
                    "load_language": False,  # Because we don't want to overwrite the language key
                    "force_recompute_dataset_statistics": False,
                    "action_proprio_normalization_type": NormalizationType.NONE,
                },
                "traj_transform_kwargs": {
                    "window_size": 1,
                    "action_horizon": 64,
                },
                "frame_transform_kwargs": {
                    "image_augment_kwargs": {},
                    "resize_size": {
                        "primary": [224, 224],
                        "secondary": [224, 224],
                        "wrist_left": [224, 224],
                        "wrist_right": [224, 224],
                    },
                    "num_parallel_calls": 64,
                },
                "balance_weights": True,
                "shuffle_buffer_size": 50000,
                "traj_transform_threads": 16,
                "traj_read_threads": 16,
            },
            "optimizer_kwargs": {
                "optimizer": "adamw",
                "num_steps": num_train_steps,
                "llm_optimizer_kwargs": {
                    "init_learning_rate": 0,
                    "learning_rate": 5e-5,
                    "warmup_steps": 500,
                    "weight_decay": 5e-6,
                    "grad_norm_clip": 10.0,
                },
                "embed_optimizer_kwargs": {
                    "init_learning_rate": 0,
                    "learning_rate": 5e-5,
                    "warmup_steps": 100,
                    "weight_decay": 0.0,
                    "grad_norm_clip": 10.0,
                },
                "img_optimizer_kwargs": {
                    "init_learning_rate": 0,
                    "learning_rate": 5e-5,
                    "warmup_steps": 500,
                    "weight_decay": 5e-6,
                    "grad_norm_clip": 10.0,
                },
            },
        }
    )
