from octo.data.utils.data_utils import NormalizationType
from ml_collections.config_dict import placeholder, ConfigDict, FieldReference

from palivla.spec import ModuleSpec

placeholder(int)._value

def get_config(config_str: str):
    from palivla.model import get_default_config
    num_train_steps = FieldReference(500000, int)
    chunk_size = FieldReference(64, int)
    data_dim = 14

    num_tokens = FieldReference(16, int)

    if config_str == "attention":
        tokenizer_config = {
            "__ctor": "palivla.learned_tokenizer.FsqAttentionTokenizer",
            "config": {
                "embed_dim": 256,
                "data_dim": data_dim,
                "data_horizon": chunk_size,
                "num_tokens": num_tokens,
                "num_layers": 4,
                "target_codebook_size": 1024,
                "causal": False,
                "use_state_conditioning": False,
                "min_action_value": None,
                "max_action_value": None,
            }
        }
    elif config_str == "conv":
        tokenizer_config = {
            "__ctor": "palivla.learned_tokenizer.LfqResnetTokenizer",
            "config": {
                "stages": [3, 3],
                "stage_filters": [256, 256],
                "target_codebook_size": 1024,
                "data_dim": data_dim,
                "data_horizon": chunk_size,
                "l1_loss_weight": 0.0,
                "l2_loss_weight": 1.0,
                "commit_loss_weight": 0.0,
                "entropy_loss_weight": 0.0,
                "use_state_conditioning": False,
            }
        }
    elif config_str == "bin":
        tokenizer_config = {
            "__ctor": "palivla.tokenizer.BinTokenizer",
            "config": {
                "num_bins": num_tokens,
                "data_dim": data_dim,
                "data_horizon": chunk_size,
                "min_action_value": -2.0,
                "max_action_value": 2.0,
            }
        }
        num_train_steps.set(0)
    else:
        raise ValueError(f"Unknown tokenizer config: {config_str}")

    return ConfigDict(
        {
            "wandb_project": "palivla-aloha-tokenizer",
            "run_name_format": f"{config_str}-t{{tokenizer/config/num_tokens}}-b{{batch_size}}-lr{{tokenizer_optimizer/config/learning_rate}}-{{wandb_run_id}}",
            "save_path": placeholder(str),
            "batch_size": 512,
            "eval_batch_size": 128,
            "num_steps": num_train_steps,
            "eval_interval": 500,
            "save_interval": 5000,
            "log_interval": 100,
            "data_axis_size": 1,
            "fsdp_axis_size": -1,
            "resume_from_checkpoint_dir": placeholder(str),
            "resume_from_checkpoint_step": placeholder(int),
            "chunk_relative_actions": True,
            "dataset_kwargs": {
                "oxe_kwargs": {
                    "data_mix": "aloha",
                    "data_dir": "/data/rlds",
                    "load_camera_views": ["primary", "secondary", "wrist_left", "wrist_right"],
                    "load_depth": False,
                    "load_proprio": True,
                    "load_language": False, # Because we don't want to overwrite the language key
                    "force_recompute_dataset_statistics": False,
                    "action_proprio_normalization_type": NormalizationType.NONE,
                },
                "traj_transform_kwargs": {
                    "window_size": 1,
                    "action_horizon": chunk_size,
                },
                "frame_transform_kwargs": {
                    "image_augment_kwargs": {},
                    "resize_size": {"primary": [224, 224], "secondary": [224, 224], "wrist_left": [224, 224], "wrist_right": [224, 224]},
                    "num_parallel_calls": 64,
                },
                "balance_weights": True,
                "shuffle_buffer_size": 50000,
                "traj_transform_threads": 16,
                "traj_read_threads": 16,
            },
            "tokenizer": tokenizer_config,
            "tokenizer_optimizer": {
                "__ctor": "palivla.load_model.adamw_cosine_warmup",
                "config": {
                    "learning_rate": 5e-5,
                    "weight_decay": 0.0,
                    "warmup_steps": 1000,
                    "total_steps": num_train_steps,
                    "global_norm": 10.0,
                    "b1": 0.8,
                    "b2": 0.95,
                },
            },
            # Placeholders, for the pod config
            "paligemma_weights_path": placeholder(str),
            "language_tokenizer_path": placeholder(str),
        }
    )
