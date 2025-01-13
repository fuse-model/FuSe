from ml_collections import ConfigDict
from absl import flags

from octo.data.utils.text_processing import HFTokenizer
from octo.utils.spec import ModuleSpec

def get_compositional_config():
    BOTTOM = 0.05
    TOP = 0.10
    TOP_COOLDOWN = 5
    return ConfigDict({
        "bottom": BOTTOM,
        "top": TOP,
        "top_cooldown": TOP_COOLDOWN,
        "modality_to_decode": "audio",
    })

def get_config():
    STEP_DURATION_MESSAGE = """
    Bridge data was collected with non-blocking control and a step duration of 0.2s.
    However, we relabel the actions to make it look like the data was collected with
    blocking control and we evaluate with blocking control.
    We also use a step duration of 0.4s to reduce the jerkiness of the policy.
    Be sure to change the step duration back to 0.2 if evaluating with non-blocking control.
    """
    STICKY_GRIPPER_NUM_STEPS = 1
    ENV_PARAMS = {
        "camera_topics": [
            {"name": "/blue/image_raw"}, 
            {"name": "/wrist/image_raw", "is_python_node": True}
        ],
        "digit_topics": [
            {"name": '/digit_left/image_raw', "width": 320, "height":240, "is_python_node": True},
            {"name": '/digit_right/image_raw', "width": 320, "height":240, "is_python_node": True}
        ],
        "override_workspace_boundaries": [[0.1, -0.15, -0.01, -1.57, 0], [0.45, 0.25, 0.25, 1.57, 0]],
        "move_duration": 0.2,
        'mic_topic': '/mic/mic_raw',
        # 'imu_topic': '/imu/data', # uncomment this if you want to stream IMU data from the server
    }

    # received image sizes from server
    RECEIVED_IMAGE_SIZES = { 
        'image_0': (256, 256), 
        'image_1': (128, 128),
        'digit': (224, 224),
        'background': (224, 224), 
        'mel_spectro': (128, 128)
    }

    RESIZE_MAP = { 
        'image_0': (256, 256), 
        'image_1': (128, 128),

        'digit_l': (224, 224),
        'digit_r': (224, 224),
        'background_l': (224, 224), 
        'background_r': (224, 224), 
    }

    # renaming map
    OBS_KEY_MAP = { 
        "image": { 
            "primary": "image_0", 
            "wrist": "image_1", 
            "digit_left": "digit_l", 
            "digit_right": "digit_r", 
            'digit_left_background': 'background_l', 
            'digit_right_background': 'background_r'
        }, 
    }

    CALCULATED_FIELDS = [
        "spectro",
    ]

    return ConfigDict({
        "env_params": ENV_PARAMS,
        "step_duration_message": STEP_DURATION_MESSAGE,
        "sticky_gripper_num_steps": STICKY_GRIPPER_NUM_STEPS,
        "received_image_sizes": RECEIVED_IMAGE_SIZES,
        "resize_map": RESIZE_MAP,
        "obs_key_map": OBS_KEY_MAP,
        "calculated_fields": CALCULATED_FIELDS,
        "compositional": get_compositional_config(),
        "text_processor_spec": ModuleSpec.create(
            HFTokenizer,
            tokenizer_name="t5-base",
            encode_with_model=False,
            tokenizer_kwargs={
                "max_length": 24,
                "padding": "max_length",
                "truncation": True,
                "return_tensors": "np",
            },
        )
    }, 
    convert_dict=False)