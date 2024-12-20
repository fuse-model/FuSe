from ml_collections.config_dict import ConfigDict, FieldReference, placeholder
from palivla.model import get_default_config
from palivla.spec import ModuleSpec

placeholder(int)._value


def get_config(config_str: str):
    data_mix = FieldReference("oxe_magic_soup", str)
    num_train_steps = FieldReference(400000, int)

    config_str_list = config_str.split("_")

    from big_vision.models.ppp.gemma import get_config

    variant = "gemma_2b"
    llm_embdim = get_config(variant).width
    model_config = get_default_config()

    use_wrist = "wrist" in config_str_list
    use_proprio = "proprio" in config_str_list

    model_config["modality_mappings"] = {
        "image_primary": "img",
        "text": "llm",
    }
    model_config["target_key_order"] = ("image_primary",)

    if use_wrist:
        model_config["modality_mappings"]["image_wrist"] = "img"
        model_config["target_key_order"] = model_config["target_key_order"] + (
            "image_wrist",
        )

    if use_proprio:
        model_config["modality_mappings"]["proprio"] = "proprio"
        model_config["target_key_order"] = model_config["target_key_order"] + (
            "proprio",
        )
        model_config["encoder_specs"]["proprio"] = {
            "__ctor": "flax.linen.Dense",
            "config": {"features": llm_embdim},
        }

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

    return ConfigDict(
        {
            "data_mix": data_mix,
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
            "model_config": model_config,
            "dataset_kwargs": {
                "oxe_kwargs": {
                    "data_mix": data_mix,
                    "data_dir": "/data/rlds/",
                    "load_camera_views": ["primary", "wrist"],
                    "load_depth": False,
                    "load_proprio": True,
                    "load_language": True,
                    "force_recompute_dataset_statistics": False,
                    "action_proprio_normalization_type": "scale",
                },
                "traj_transform_kwargs": {
                    "window_size": 1,
                    "action_horizon": 1,
                },
                "frame_transform_kwargs": {
                    "image_augment_kwargs": {
                        "primary": workspace_augment_kwargs,
                        "wrist": wrist_augment_kwargs,
                    },
                    "resize_size": {"primary": [224, 224], "wrist": [224, 224]},
                    "image_dropout_prob": 0.5,
                },
                "balance_weights": True,
                "shuffle_buffer_size": 50000,
                "traj_transform_threads": 48,
                "traj_read_threads": 48,
            },
            "extra_dataset_transform_kwargs": {
                "multimodal_rephrasings": False,
                "chunk_relative_actions": False,
                "multimodal_rephrasing_kwargs": {},
                "proprio_dropout_prob": 0.5,
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
