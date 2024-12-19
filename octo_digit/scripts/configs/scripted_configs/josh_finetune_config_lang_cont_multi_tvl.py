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
from octo.model.components.tvl_vit import JaxViT


def get_config(config_str=None):
    # mode, task = "full,multimodal".split(',') 
    mode, task = "full,language_conditioned".split(",")
    assert task in ["image_conditioned", "language_conditioned", "multimodal"]
    assert mode in ["full", "head_only", "head_mlp_only"]

    # Fill this in for your own dataset!

    # There should be two image keys
    # first image key should be the third-person view (None if not used)
    # and second image key should be the wrist view (None if not used)
    TVL_NORMALIZE = False
    REPEAT_TASK_TOKENS = True 
    DO_BACKGROUND_SUBTRACTION = True 
    BACKGROUND_SUBTRACT_MAP = {} 
    PRETRAINED_LOADERS = [ModuleSpec.create(tvl_loader)] 
    VALID_MODALITY_NAMES = {'cams', 'cam_primary', 'cam_wrist', 'mic', 'digits', 'digit_left', 'digit_right', 'digit_embeddings', 'siglip'}
    config_input = config_str.split(';')
    ds_name = config_input[0]

    modalities = set()
    if len(config_input) == 1: 
        modality_str = 'cams'
    else: 
        modality_str = config_input[1]

    for modality in modality_str.split(','): 
        if modality: 
            modality = modality.strip() 
            if modality not in VALID_MODALITY_NAMES: 
                raise ValueError(f'Input was {config_str}. Input is expected to be "[ds_name];[modality1],[modality2],..." string, '
                                f'where each modality is in {VALID_MODALITY_NAMES}. Received erroneous modality {modality}.')
            else: 
                modalities.add(modality)

    MODALITY_KEEP_PROBABILITIES = { 
        'visual': 0.7,
        'tactile': 0.7,
        'audio': 0.7,
    }
    ANNOTATION_REMOVE_KEYS = {''}

    if len(config_input) < 3: 
        LANGUAGE_KEY = 'all_lang_list'
        # LANGUAGE_KEY = 'multimodal_annotations'
        # LANGUAGE_KEY = 'language_instruction'
    else: 
        LANGUAGE_KEY = config_input[2]

    if 'cams' in modalities: 
        modalities = modalities | {'cam_primary', 'cam_wrist'}
    if 'digits' in modalities: 
        modalities = modalities | {'digit_left', 'digit_right'}

    img_obs_keys = {}
    if 'cam_primary' in modalities:
        img_obs_keys['primary'] = 'image_0'
    
    if 'cam_wrist' in modalities:
        img_obs_keys['wrist'] = 'image_1'


    DIGITS_TO_USE = set()
    if 'digit_left' in modalities:
        DIGITS_TO_USE.add('left')
    
    if 'digit_right' in modalities: 
        DIGITS_TO_USE.add('right')


    if ds_name == "" or ds_name is None or ds_name == "None": 
        NAME = "14.0.0"
    else: 
        NAME = ds_name 

    ANNOTATION_MANAGER_KWARGS = {
        'modality_keep_probabilities': MODALITY_KEEP_PROBABILITIES,
        'remove_keys': ANNOTATION_REMOVE_KEYS, 
        'simple_ratio': 0.14,
        'uniform_modality_dropout': True,
        'reconstruction_loss_keys': ['simple', 'visual,tactile'] # ['simple']#  ['simple', 'visual,tactile', 'visual,tactile,audio']
    } 
    # simple, empty, visual, tactile, audio, visual,tactile
    

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

    NEW_OBS_TOKENIZERS = {} 
    
    if False: 
        PRETRAINED_LOADERS = []
        # ModuleSpec.create(
        #     resnet_26_loader, 
        #     restore_path='gs://sudeep_r2d2_experiments/R26_S_32-i21k-300ep-lr_0.001-aug_light1-wd_0.1-do_0.0-sd_0.0.npz',
        #     exclude_tokenizers=('.*primary', '.*wrist')
        # )
    
        digit_names = {}
        if "left" in DIGITS_TO_USE: 
            digit_names["digit_left"] = "digit_0"
            if DO_BACKGROUND_SUBTRACTION: 
                BACKGROUND_SUBTRACT_MAP['image_digit_left'] = 'image_digit_left_background'
                digit_names['digit_left_background'] = 'digit_0_background'
        if "right" in DIGITS_TO_USE: 
            digit_names["digit_right"] = "digit_1"
            if DO_BACKGROUND_SUBTRACTION: 
                BACKGROUND_SUBTRACT_MAP['image_digit_right'] = 'image_digit_right_background'
                digit_names['digit_right_background'] = 'digit_1_background'

        FINETUNING_KWARGS["image_obs_keys"].update(digit_names)
        stack_keys = [f'image_digit_{key}' for key in DIGITS_TO_USE]
        if stack_keys: 
            digit_tokenizer = {"digits": 
                ModuleSpec.create(
                    ImageTokenizer,
                    obs_stack_keys=stack_keys,
                    task_stack_keys=[],
                    encoder=ModuleSpec.create(SmallStem16, use_film=False),
                )}
            NEW_OBS_TOKENIZERS.update(digit_tokenizer)

    if "mic" in modalities: 
        FINETUNING_KWARGS["sensor_obs_keys"]['spectro'] = 'mel_spectro'
        mic_tokenizer =  {
            "spectrogram": ModuleSpec.create( 
                            UnsqueezingImageTokenizer,
                            obs_stack_keys = ["spectro"], 
                            task_stack_keys=[], 
                            encoder=ModuleSpec.create(SmallStem16, use_film=False),
                    )
        }
        NEW_OBS_TOKENIZERS.update(mic_tokenizer)

    if "imu" in modalities: 
        # "imu": ModuleSpec.create( 
        #             ProjectionTokenizer,
        #             num_output_tokens=7,
        #             n_bins=256,
        #             obs_keys=["imu"], 
        #         ),
        raise NotImplementedError
    DIGITS_TO_USE = ['left', 'right']
    if True: 
        digit_names = {}
        TVL_NORMALIZE = True
        DO_BACKGROUND_SUBTRACTION = True
        if "left" in DIGITS_TO_USE: 
            digit_names["digit_left"] = "digit_0"
            if DO_BACKGROUND_SUBTRACTION: 
                BACKGROUND_SUBTRACT_MAP['image_digit_left'] = 'image_digit_left_background'
                digit_names['digit_left_background'] = 'digit_0_background'
        if "right" in DIGITS_TO_USE: 
            digit_names["digit_right"] = "digit_1"
            if DO_BACKGROUND_SUBTRACTION: 
                BACKGROUND_SUBTRACT_MAP['image_digit_right'] = 'image_digit_right_background'
                digit_names['digit_right_background'] = 'digit_1_background'
        
        FINETUNING_KWARGS["image_obs_keys"].update(digit_names)
        if 'left' in DIGITS_TO_USE: 
            tvl_tokenizer = {"tvl_left": 
                ModuleSpec.create(
                    ImageTokenizer,
                    obs_stack_keys=['image_digit_left',],
                    task_stack_keys=[],
                    encoder=ModuleSpec.create(JaxViT, img_size=(224, 224), num_classes=512),
                    strict_key_match=True,
                )
                }
            NEW_OBS_TOKENIZERS.update(tvl_tokenizer)
            
            tvl_tokenizer = {"tvl_right": 
                ModuleSpec.create(
                    ImageTokenizer,
                    obs_stack_keys=['image_digit_right'],
                    task_stack_keys=[],
                    encoder=ModuleSpec.create(JaxViT, img_size=(224, 224), num_classes=512),
                    strict_key_match=True
                )
                }
            NEW_OBS_TOKENIZERS.update(tvl_tokenizer)
        # stack_keys = [f'image_digit_{key}' for key in DIGITS_TO_USE]
        # if stack_keys: 
        #     tvl_tokenizer = {"tvl_left": 
        #         ModuleSpec.create(
        #             ImageTokenizer,
        #             obs_stack_keys=stack_keys,
        #             task_stack_keys=[],
        #             encoder=ModuleSpec.create(JaxViT, img_size=(224, 224), num_classes=512)
        #         )
        #         }
        #     NEW_OBS_TOKENIZERS.update(tvl_tokenizer)
    
    
    if 'siglip' in modalities: 
        SIGLIP_KEYS = {'digit_left': 'digit_0'}
        FINETUNING_KWARGS['image_obs_keys'].update(SIGLIP_KEYS)
        siglip_tokenizer = {
            "siglip": {
                    "freeze": False, # True, 
                    "config": { 
                        'encoder_path':'/home/sjosh/nfs/octo_digit/siglip.npz:img', # '/home/joshwajones/octo/siglip.npz:img',
                        'image_model': 'vit', 
                        'image': dict(variant='B/16', pool_type='map'),
                        'obs_stack_keys': [f'image_{key}' for key in SIGLIP_KEYS]
                    }
                }
        }
        NEW_OBS_TOKENIZERS.update(siglip_tokenizer)
        raise NotImplementedError

    # NEW_ACTION_HEAD = None 
    NEW_ACTION_HEAD = ModuleSpec.create(
        MSEActionHead,
        readout_key="readout_action",
        use_map = True, # should this be disabled? 
        action_horizon=4,
        action_dim=7
    )

    if mode == "full":
        frozen_keys = None
    elif mode == "head_only":
        frozen_keys = ("octo_transformer.*",)
    elif mode == "head_mlp_only":
        frozen_keys = (
            "octo_transformer.*",
            "heads_*.map_head.probe",
            "heads_*.map_head.MultiHeadDotProductAttention_0.*",
        )
    elif mode == "frozen_transformer":
        frozen_keys = ("octo_transformer.BlockTransformer_0.*",)
    else:
        raise ValueError("Invalid mode")

    if REPEAT_TASK_TOKENS: 
        pretrained_path = "hf://rail-berkeley/octo-small-1.5"
    else: 
        pretrained_path = "hf://rail-berkeley/octo-small-1.5"

    max_steps = FieldReference(50000)
    window_size = FieldReference(default=2)

    config = dict(
        pretrained_loaders=PRETRAINED_LOADERS, 
        modalities=modalities,
        pretrained_path=pretrained_path, 
        batch_size=512,
        shuffle_buffer_size=10000,
        num_steps=max_steps,
        log_interval=100,
        eval_interval=5000,
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
            frozen_keys=frozen_keys,
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

    if "siglip" in NEW_OBS_TOKENIZERS: 
        should_freeze, siglip_cfg = NEW_OBS_TOKENIZERS["siglip"]["freeze"], NEW_OBS_TOKENIZERS["siglip"]["config"]
        config["siglip_config"] = siglip_cfg
        NEW_OBS_TOKENIZERS["siglip"] = ModuleSpec.create( 
                    SiglipTokenizer,
                    image=siglip_cfg["image"],
                    image_model=siglip_cfg["image_model"],
                    encoder_path=siglip_cfg["encoder_path"],
                    n_bins=256,
                    obs_keys=['siglip'],
                    obs_stack_keys=siglip_cfg['obs_stack_keys']
        )

        if should_freeze: 
            prev_frozen = frozen_keys if frozen_keys else ()
            config["optimizer"]["frozen_keys"] = prev_frozen + ("octo_transformer.observation_tokenizers_siglip.*", "*hf_model*")


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
    # DIGIT_SIZE = (128, 128)
    DIGIT_SIZE = (224, 224)
    frame_transform_kwargs = dict(
        resize_size={
            "primary": (256, 256),  # workspace (3rd person) camera is at 256x256
            "wrist": (128, 128),   # wrist camera is at 128x128
            "digit_left": DIGIT_SIZE, #(128, 128),
            "digit_right": DIGIT_SIZE,  # (128, 128)
        },
        image_augment_kwargs = { 
            "primary": workspace_augment_kwargs, 
            "wrist": wrist_augment_kwargs, 
            "digit_left": digit_augment_kwargs,
            "digit_right": digit_augment_kwargs,
        }, 
        background_subtraction_map = BACKGROUND_SUBTRACT_MAP,
        # normalize_obs_keys=[f'image_digit_{name}' for name in DIGITS_TO_USE] if TVL_NORMALIZE else tuple()
    )
    for key, val in BACKGROUND_SUBTRACT_MAP.items(): 
        if key.startswith('image_'): 
            key = key[len('image_'):]
        if val.startswith('image_'): 
            val = val[len('image_'):]
        frame_transform_kwargs['resize_size'][val] = frame_transform_kwargs['resize_size'][key]
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


    # LANGUAGE_RECONSTRUCTION_HEAD = None 
    LANGUAGE_RECONSTRUCTION_HEAD = ModuleSpec.create(
        CLIPContrastiveHead,
        readout_key="readout_language",
        use_map = True
    )

    if LANGUAGE_RECONSTRUCTION_HEAD is not None: 
        config['update_config']['model']['readouts'] = {'language': 16}
    config['reconstruction_loss_weight'] = 3.
    config['lang_head'] = LANGUAGE_RECONSTRUCTION_HEAD



    config['update_config'] = ConfigDict(config['update_config'])
    config['pop_keys'] = [
        ('model', 'heads', 'action', 'kwargs', 'n_diffusion_samples'),
        ('model', 'heads', 'action', 'kwargs', 'dropout_rate')
    ]

    return ConfigDict(config)
