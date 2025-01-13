from octo.data.utils.data_utils import NormalizationType
from ml_collections.config_dict import placeholder, ConfigDict, FieldReference

from palivla.model import get_default_config

placeholder(int)._value

def get_config():
    num_train_steps = FieldReference(100000, int)

    return ConfigDict(
        {
            "wandb_project": "palivla",
            "paligemma_weights_path": placeholder(str),
            "language_tokenizer_path": placeholder(str),
            "action_tokenizer_path": placeholder(str),
            "model_load_fn": "big_vision.models.proj.paligemma.paligemma.load",
            "tokenizer_path": "models/paligemma_tokenizer.model",
            "model_path": "models/paligemma",
            "save_path": placeholder(str),
            "batch_size": 192,
            "eval_batch_size": 128,
            "num_steps": num_train_steps,
            "eval_interval": 100,
            "save_interval": 1000,
            "log_interval": 1,
            "data_axis_size": 1,
            "fsdp_axis_size": -1,
            "resume_from_checkpoint_dir": placeholder(str),
            "resume_from_checkpoint_step": placeholder(int),
            "model_config": get_default_config(),
            "dataset_kwargs": {
                "oxe_kwargs": {
                    "data_mix": "bridge",
                    "data_dir": "/data/rlds/",
                    "load_camera_views": ["primary"],
                    "load_depth": False,
                    "load_proprio": True,
                    "load_language": True,
                    "force_recompute_dataset_statistics": False,
                    "action_proprio_normalization_type": NormalizationType.NORMAL,
                },
                "traj_transform_kwargs": {
                    "window_size": 1,
                    "action_horizon": 1,
                },
                "frame_transform_kwargs": {
                    "image_augment_kwargs": {},
                    "resize_size": {"primary": [224, 224]},
                },
                "balance_weights": True,
                "shuffle_buffer_size": 50000,
                "traj_transform_threads": 16,
                "traj_read_threads": 16,
            },
            "extra_dataset_transform_kwargs": {
                "multimodal_rephrasings": False,
                "chunk_relative_actions": False,
                "multimodal_rephrasing_kwargs": {},
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
            }
        }
    )
