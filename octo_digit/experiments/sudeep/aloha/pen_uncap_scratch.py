from copy import deepcopy
import functools

from octo.utils.spec import ModuleSpec
from octo.data.utils.text_processing import MuseEmbedding
from octo.model.components.tokenizers import ImageTokenizer
from octo.model.components.vit_encoders import ResNet26FILM
from octo.model.components.action_heads import UNetDDPMActionHead
from ml_collections import ConfigDict
from ml_collections.config_dict import FieldReference, placeholder


# add octo folder to PYTHONPATH for importing standardization function
import octo, os, sys
sys.path.append(os.path.join(os.path.dirname(octo.__file__), "../"))


def update_config(config, **kwargs):
    updates = ConfigDict(kwargs)
    new_config = deepcopy(config)
    new_config.update(updates)
    return new_config


def wrap(f):
    """Simple wrapper to enable passing config strings to `get_config`

    Usage:

    python train.py --config=config.py:vit_s,multimodal
    python train.py --config=config.py:transformer_size=vit_s
    """

    @functools.wraps(f)
    def wrapped_f(config_string=None):
        if config_string is None:
            return f()
        elements = config_string.split(",")
        args, kwargs = [], {}
        for e in elements:
            if "=" in e:
                k, v = e.split("=")
                kwargs[k] = v
            else:
                args.append(e)
        return f(*args, **kwargs)

    return wrapped_f


@wrap
def get_config(
    act_type="abs_act",
    modality='text',
    pred_horizon=64,
):
    pred_horizon=int(pred_horizon)
    print("Creating config with: ", locals())

    num_steps = FieldReference(default=int(500000))
    window_size = FieldReference(default=1)
    return ConfigDict(
        dict(
            seed=42,
            num_steps=num_steps,
            save_dir=placeholder(str), #"/scratch2/sudeep/plate_scratch",
            model=get_model_config(pred_horizon),
            window_size=window_size,
            dataset_kwargs=get_dataset_config(modality, window_size, pred_horizon, act_type),
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
                frozen_keys=tuple(),
            ),
            prefetch_num_batches=0,
            start_step=placeholder(int),
            log_interval=100,
            eval_interval=50000000,
            viz_interval=50000000,
            save_interval=100000,
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
            text_processor_kwargs=dict(),
            pretrained_loaders=tuple(),
            pretrained_loader_kwargs=tuple(),
            wandb=dict(
                project="octo",
                group=placeholder(str),
                entity=placeholder(str),
            ),
            wandb_resume_id=placeholder(str),
            eval_datasets=("aloha_pen_uncap_diverse_dataset"),
        )
    )


def get_dataset_config(modality="text", window_size=1, pred_horizon=64, act_type='rel_act'):
    if act_type == "abs_act":
        n_act_dims = 14
        standardize_fn = "experiments.sudeep.aloha.standardization_transforms:aloha_dataset_transform"
    else:
        raise ValueError

    if modality == "multimodal":
        task_augmentation = dict(
            goal_relabeling_strategy = "uniform",
            task_augment_strategy="delete_task_conditioning",
            task_augment_kwargs=dict(
                keep_image_prob=0.5,
            ),
        )
    elif modality == "image_conditioned":
        task_augmentation = dict(
            goal_relabeling_strategy = "uniform",
            task_augment_strategy="delete_task_conditioning",
            task_augment_kwargs=dict(
                keep_image_prob=1.0,
            ),
        )
    elif modality == "text":
        task_augmentation = dict(
            goal_relabeling_strategy = "uniform",
            task_augment_strategy="delete_task_conditioning",
            task_augment_kwargs=dict(
                keep_image_prob=0.0,
            ),
        )
    else:
        raise ValueError(f"Unknown modality {modality}")

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
            "wrist": (128, 128),  # wrist camera is at 128x128
        },
        image_augment_kwargs={
            "primary": wrist_augment_kwargs,
            "wrist": wrist_augment_kwargs,
        },
    )

    return dict(
            dataset_kwargs_list=[dict(
                        name="aloha_pen_uncap_diverse_dataset",
                        data_dir="/scratch2/sudeep/tfds", #"gs://rail-orca-central2",
                        image_obs_keys= {"primary": "cam_right_wrist", "wrist": "cam_left_wrist"},
                        proprio_obs_key="proprio",
                        action_normalization_mask=[True] * n_act_dims,
                        standardize_fn=ModuleSpec.create(standardize_fn),
                        language_key= "language_instruction",

            )],
            sample_weights=[1.0],
            traj_transform_kwargs=dict(
                window_size=window_size,
                action_horizon=pred_horizon,
                **task_augmentation,
            ),
            frame_transform_kwargs=frame_transform_kwargs,
            batch_size=256,
            shuffle_buffer_size=50000,
            balance_weights=True,
            traj_transform_threads=16,  # shared between all datasets
            traj_read_threads=16,  # shared between all datasets
        )


def get_transformer_kwargs():
    default_params = {
        "attention_dropout_rate": 0.0,
        "add_position_embedding": False,
    }

    TRANSFORMER_SIZES = {
        "vanilla": dict(
            num_layers=2,
            mlp_dim=512,
            num_attention_heads=4,
            dropout_rate=0.1,
        )
    }

    TOKEN_DIMS = {
        "vanilla": 256,
    }
    return dict(
        token_embedding_size=TOKEN_DIMS["vanilla"],
        transformer_kwargs={
            **default_params,
            **TRANSFORMER_SIZES["vanilla"],
        },
    )

def get_model_config(pred_horizon):
    n_act_dims = 14
    encoder = ModuleSpec.create(ResNet26FILM)

    action_head = ModuleSpec.create(
        UNetDDPMActionHead,
        action_dim=n_act_dims,
        action_horizon=pred_horizon,
        readout_key="obs",
        use_map=False,
        flatten_tokens=False,
        max_action=1,
        timesteps=100,
    )

    return {
        **get_transformer_kwargs(),
        "max_horizon": 100,
        "readouts": dict(),
        "heads": dict(
            action=action_head,
        ),
        "observation_tokenizers": {
        "primary": ModuleSpec.create(
               ImageTokenizer,
               obs_stack_keys=["image_primary"],
               task_stack_keys=["image_primary"],
               task_film_keys=["language_instruction"],
               encoder=encoder,
           ),
        "wrist": ModuleSpec.create(
               ImageTokenizer,
               obs_stack_keys=["image_wrist"],
               task_stack_keys=["image_wrist"],
               task_film_keys=["language_instruction"],
               encoder=encoder,
           ),
        },
        "task_tokenizers": dict(),
    }
