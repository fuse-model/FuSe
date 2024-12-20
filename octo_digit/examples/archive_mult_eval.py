"""
This script shows how we evaluated a finetuned Octo model on a real WidowX robot. While the exact specifics may not
be applicable to your use case, this script serves as a didactic example of how to use Octo in a real-world setting.

If you wish, you may reproduce these results by [reproducing the robot setup](https://rail-berkeley.github.io/bridgedata/)
and installing [the robot controller](https://github.com/rail-berkeley/bridge_data_robot)
"""

from collections import OrderedDict
from ctypes import Union
from dataclasses import dataclass
from datetime import datetime
from distutils.sysconfig import PREFIX
from email.policy import default
from functools import partial
from itertools import combinations, filterfalse
import os
import pickle
import time
from typing import Iterable
from xmlrpc.client import FastMarshaller

from absl import app, flags, logging
import click
import cv2
from envs.widowx_env import convert_obs, state_to_eep, wait_for_obs, WidowXGym
import imageio
import jax
import jax.numpy as jnp
import numpy as np
from octo.data.utils.text_processing import MuseEmbedding
from octo.model.octo_model import OctoModel
from octo.model.resnet_model import ResnetModel
from octo.utils.gym_wrappers import (
    HistoryWrapper,
    ObsProcessingWrapper,
    ResizeImageWrapper,
    ResizeImageWrapperDict,
    RHCWrapper,
    TemporalEnsembleWrapper,
)
from PIL import Image
import widowx_envs.utils.transformation_utils as tr
from widowx_envs.widowx_env_service import WidowXClient, WidowXConfigs, WidowXStatus

np.set_printoptions(suppress=True)

logging.set_verbosity(logging.WARNING)


FLAGS = flags.FLAGS


PREFIX_COMMANDS = {"visual": "looks", "tactile": "feels", "audio": "sounds"}
REMOVE_KEYS = {"", "audio"}

# TODO(CARLO): fill in checkpoint and step here

PATHS = [
    "all_multi_resnet_32_20240627_075737",
    # "cams_resnet_32_20240627_122302"
]
flags.DEFINE_string("checkpoint_weights_path", "NULL", "Path to checkpoint")

NAME_TO_WINDOW_SIZE = {
    "all_multi_resnet_32_20240627_075737": 2,
    "cams_resnet_32_20240627_122302": 2,
}

SIMPLE_CMD = "Grab and raise the red block."
COMPLEX_CMD = "Grab the object that and looks red, rectangle and feels wooden, matte."


@dataclass
class MultimodalCommand:
    annotation_dict: dict[str, str]
    prefix: str = "Grab the object that"

    def __post_init__(self):
        sorted_keys = list(self.annotation_dict.keys())
        all_combos = tuple()
        for i in range(len(sorted_keys) + 1):
            all_combos += tuple(combinations(sorted_keys, i))
        string_combos = [",".join(tup) for tup in all_combos]

        def construct_annotation(string_key):
            annotation = [self.prefix]
            is_first = True
            for modality in string_key.split(","):
                connector = "" if is_first else " and"
                annotation.append(
                    f"{connector} {PREFIX_COMMANDS[modality]} {self.annotation_dict[modality]}"
                )
                is_first = False
            annotation.append(".")
            return "".join(annotation)

        self.all_annotations = [
            construct_annotation(key) for key in string_combos if key not in REMOVE_KEYS
        ]
        self.index = 0

    def curr_annotation(self):
        return self.all_annotations[self.index]

    def next_annotation(self):
        self.index = (self.index + 1) % len(self.all_annotations)
        return self.curr_annotation()

    def reset(self):
        self.index = len(self.all_annotations) - 1


MULTI_CMD = MultimodalCommand(
    annotation_dict={
        "visual": "silver, large",
        "tactile": "metallic, sharp",
        "audio": "jangly",
    }
)

NAME_TO_COMMAND = {
    "standard_largebatch_20240605_015537": SIMPLE_CMD,
    "resnet26_20240613_035123": SIMPLE_CMD,
    "combined_dataset_20240606_223700": SIMPLE_CMD,
    "cams_only_23_20240614_014341": SIMPLE_CMD,
    "cams_mic_23_20240614_014343": SIMPLE_CMD,
    "cams_mic_digit_23_20240614_014346": SIMPLE_CMD,
    "cams_digit_23_20240614_014348": SIMPLE_CMD,
    "cams_25_20240617_042143": COMPLEX_CMD,
    "all_modalities_25_20240617_101010": COMPLEX_CMD,
    "overhead_only_23_20240620_191557": SIMPLE_CMD,
    "overhead_digits_23_20240620_191602": SIMPLE_CMD,
    "map_23_20240621_070600": SIMPLE_CMD,
    "cams_digleftembedding_23_20240621_073044": SIMPLE_CMD,
    "cams_mic_digitleft_23_20240620_191548": SIMPLE_CMD,
    "cams_multi_27_20240624_190301": MULTI_CMD,
    "all_multi_27_20240624_190329": MULTI_CMD,
    "all_multi_resnet_32_20240627_075737": MULTI_CMD,
    "cams_resnet_32_20240627_122302": MULTI_CMD,
}


flags.DEFINE_bool("is_resnet", False, "if ckpt is a resnet-only model")

flags.DEFINE_integer("checkpoint_step", 50000, "Checkpoint step")

dell2_direct = "192.168.99.10"
dell2_ether = "128.32.175.252"
dell2 = dell2_ether
# custom to bridge_data_robot
flags.DEFINE_string("ip", dell2, "IP address of the robot")
flags.DEFINE_integer("port", 5556, "Port of the robot")
flags.DEFINE_spaceseplist("goal_eep", [0.3, 0.0, 0.15], "Goal position")
flags.DEFINE_spaceseplist(
    "initial_eep", [0.11796844, -0.01554691, 0.23344009], "Initial position"
)  # neutral

flags.DEFINE_bool("blocking", False, "Use the blocking controller")


flags.DEFINE_integer(
    "im_size",
    256,
    "Image size",
    # required=True
)

flags.DEFINE_string("video_save_path", "./videos", "Path to save video")
flags.DEFINE_integer("num_timesteps", 80, "num timesteps")
flags.DEFINE_integer("window_size", 2, "Observation history length")
flags.DEFINE_integer("exec_horizon", 1, "Length of action sequence to execute")


# show image flag
flags.DEFINE_bool("show_image", False, "Show image")

flags.DEFINE_string(
    "ckpt_dir",
    os.path.join(os.environ["HOME"], "tpu_octo_ckpts"),
    "directory containing the checkpoints",
)

##############################################################################

STEP_DURATION_MESSAGE = """
Bridge data was collected with non-blocking control and a step duration of 0.2s.
However, we relabel the actions to make it look like the data was collected with
blocking control and we evaluate with blocking control.
We also use a step duration of 0.4s to reduce the jerkiness of the policy.
Be sure to change the step duration back to 0.2 if evaluating with non-blocking control.
"""
STEP_DURATION = 0.2
STICKY_GRIPPER_NUM_STEPS = 1
WORKSPACE_BOUNDS = [[0.1, -0.15, -0.01, -1.57, 0], [0.45, 0.25, 0.25, 1.57, 0]]
CAMERA_TOPICS = [
    {"name": "/blue/image_raw"},
    {"name": "/wrist/image_raw", "is_python_node": True},
]
DIGIT_TOPICS = [
    {
        "name": "/digit_left/image_raw",
        "width": 320,
        "height": 240,
        "is_python_node": True,
    },
    {
        "name": "/digit_right/image_raw",
        "width": 320,
        "height": 240,
        "is_python_node": True,
    },
]


ENV_PARAMS = {
    "camera_topics": CAMERA_TOPICS,
    "digit_topics": DIGIT_TOPICS,
    "override_workspace_boundaries": WORKSPACE_BOUNDS,
    "move_duration": STEP_DURATION,
    # 'imu_topic': '/imu/imu_raw',
    "mic_topic": "/mic/mic_raw",
}

import tensorflow as tf

##############################################################################
# new key (octo) : old key (bridge)
OBS_KEY_MAP = {
    "image": {
        "primary": "image_0",
        "wrist": "image_1",
        "digit_left": "digit_l",
        "digit_right": "digit_r",
    },
    "background_l": "background_l",
    "background_r": "background_r",
    # "spectro": None,
    # "spectro": "mel_spectro",
    # "imu": "imu"
}
CALCULATED_FIELDS = [
    "spectro",
    # "digit_embeddings"
]

# TODO(CARLO): Try with this False (probably better) and True
FLIP_CHANNELS = False


def initialize_widowx_env(FLAGS, env_params):
    connection_success = False
    while not connection_success:
        try:
            widowx_client = WidowXClient(host=FLAGS.ip, port=FLAGS.port)
            connection_success = True
        except Exception as e:
            print(f"RECEIVED EXCEPTION:     {e}")
            print("Retrying environment initialization...")

    widowx_client.init(env_params, image_size=FLAGS.im_size)
    env = WidowXGym(
        widowx_client, FLAGS.im_size, FLAGS.blocking, STICKY_GRIPPER_NUM_STEPS
    )

    # wrap the robot environment
    resize_map = {
        "image_0": (256, 256),
        "image_1": (128, 128),
        "digit_l": (256, 256),
        "digit_r": (256, 256),
        "background_l": (256, 256),
        "background_r": (256, 256),
    }
    env = ResizeImageWrapperDict(env, resize_map)
    env = ObsProcessingWrapper(
        env, OBS_KEY_MAP, CALCULATED_FIELDS, flip_channels=FLIP_CHANNELS
    )
    pre_history_wrap = env

    return pre_history_wrap, env


def click_exit_wrapper(click_func, click_kwargs, is_exit_checker=False):
    got_response = False
    should_quit_bool = False
    while not got_response:
        try:
            response = click_func(**click_kwargs)
            got_response = True
        except click.exceptions.Abort as e:
            response = None
            should_quit = click_exit_wrapper(
                click.confirm,
                {
                    "text": "Quit?",
                    "default": False,
                },
                is_exit_checker=True,
            )
            should_quit_bool = should_quit[0] or should_quit[1]

            if should_quit_bool:
                got_response = True
            if is_exit_checker and should_quit[0] is not None:
                got_response = True
                response = should_quit[0]

    return response, should_quit_bool


def main(_):
    models = []
    model_idx = 0
    for name in PATHS:
        FLAGS.checkpoint_weights_path = name
        if not name.startswith("/"):
            FLAGS.checkpoint_weights_path = os.path.join(FLAGS.ckpt_dir, name)

        # set up the widowx client

        # if not FLAGS.blocking:
        #     assert STEP_DURATION == 0.2, STEP_DURATION_MESSAGE

        # load models
        loaded_dataset_stats = None
        load_kwargs = {
            "checkpoint_path": FLAGS.checkpoint_weights_path,
            "step": FLAGS.checkpoint_step,
        }
        if FLAGS.is_resnet:
            MODELTYPE = ResnetModel
            load_kwargs["text_processor"] = MuseEmbedding()
            with open("./dataset_stats.pkl", "rb") as file:
                loaded_dataset_stats = pickle.load(file)
            load_kwargs["dataset_statistics"] = loaded_dataset_stats

        else:
            MODELTYPE = OctoModel
            if "overhead" in name:
                load_kwargs["skip_wrist"] = True

        print(f"Loading model   {name}...")
        model = MODELTYPE.load_pretrained(**load_kwargs)
        print("Model loaded!\n")
        models.append((name, model))

    # def recursive_dict_print(dic, sep=""):
    #     for key, val in dic.items():
    #         print(key)
    #         if isinstance(val, dict):
    #             recursive_dict_print(val, sep + "      ")
    #         elif isinstance(val, np.ndarray):
    #             try:
    #                 temp = val
    #                 while len(temp) > 1:
    #                     temp = temp[0]
    #                 print(" ", temp[0])
    #             except:
    #                 print(val)
    #         else:
    #             print(" ", val)
    MAX_KEY_LEN = 20
    INDENT_SIZE = MAX_KEY_LEN + 4
    INDENT = "".join([" " for _ in range(INDENT_SIZE)])

    def recursive_dict_print(dictionary, prefix=""):
        for key, val in dictionary.items():
            key = key[:MAX_KEY_LEN]
            if isinstance(val, dict):
                print(f"{prefix}{key}")
                new_prefix = prefix + INDENT
                recursive_dict_print(val, new_prefix)
            else:
                indent = "".join([" " for _ in range(INDENT_SIZE - len(key))])
                print(f"{prefix}{key}:{indent}{val.shape} {val.dtype}")

    def sample_actions(
        pretrained_model: MODELTYPE,
        observations,
        tasks,
        rng,
    ):

        # add batch dim to observations
        observations = jax.tree_map(lambda x: x[None], observations)
        actions = pretrained_model.sample_actions(
            observations,
            tasks,
            unnormalization_statistics=model.dataset_statistics["action"],
            rng=rng,
        )
        # remove batch dim
        return actions[0]

    policy_fns = [
        partial(sample_actions, model, rng=jax.random.PRNGKey(0)) for _, model in models
    ]

    info_models = [
        (name, model, policy_fn) for (name, model), policy_fn in zip(models, policy_fns)
    ]

    reset_environment = True
    while reset_environment:
        reset_environment = False
        goal_image = jnp.zeros((FLAGS.im_size, FLAGS.im_size, 3), dtype=np.uint8)
        goal_instruction = ""

        if FLAGS.initial_eep is not None:
            assert isinstance(FLAGS.initial_eep, list)
            initial_eep = [float(e) for e in FLAGS.initial_eep]
            start_state = np.concatenate([initial_eep, [0, 0, 0, 1]])
        else:
            start_state = None

        default_env_params = WidowXConfigs.DefaultEnvParams

        env_params = default_env_params.copy()
        env_params.update(ENV_PARAMS)
        env_params["start_state"] = list(start_state)

        pre_history_wrap, env = initialize_widowx_env(FLAGS, env_params)

        images = []
        wrist_images = []

        reinit = True
        # goal sampling loop
        text = "DUMMY: REPLACE!"
        while True:
            try:
                modality = "l"

                if modality == "g":
                    raise NotImplementedError

                elif modality == "l":
                    print("Current instruction:  ", goal_instruction)
                    response, should_exit = click_exit_wrapper(
                        click.confirm, {"text": "Change model?", "default": False}
                    )
                    if should_exit:
                        break
                    if response:
                        print("Available models:    ")
                        for i, (name, _, _) in enumerate(info_models):
                            print(f"{i}     {name}")
                        response = input("Choose model index:   ")
                        if response.lower() == "n" or response == "":
                            model_idx = (model_idx + 1) % len(info_models)
                        else:
                            try:
                                model_idx = int(response)
                            except ValueError as e:
                                print(
                                    f"Received error {e} as a result of entering {response}. Incrementing index by one..."
                                )
                                model_idx = (model_idx + 1) % len(info_models)

                        response, should_exit = click_exit_wrapper(
                            click.confirm,
                            {"text": "Reset environment?", "default": False},
                        )
                        if should_exit:
                            break

                        if response:
                            pre_history_wrap, env = initialize_widowx_env(
                                FLAGS, env_params
                            )
                            reinit = True

                        curr_name, model, policy_fn = info_models[model_idx]
                        example_batch = model.example_batch
                        recursive_dict_print(example_batch)
                        # from flax.traverse_util import flatten_dict
                        # flattened = flatten_dict(model.params)
                        # for key, param in flattened.items():
                        #     print(key, param.shape)
                        new_window_wize = NAME_TO_WINDOW_SIZE[curr_name]
                        if new_window_wize != FLAGS.window_size or reinit:
                            FLAGS.window_size = new_window_wize
                            env = HistoryWrapper(pre_history_wrap, FLAGS.window_size)
                            env = TemporalEnsembleWrapper(env, FLAGS.exec_horizon)
                            reinit = False

                        command: Union[str, MultimodalCommand] = NAME_TO_COMMAND[
                            curr_name
                        ]
                        if isinstance(command, str):
                            text = command
                        else:
                            command.reset()
                            text = command.curr_annotation()

                        print(f"Using model with name {curr_name}")
                    # if click.confirm("Take a new instruction?   ", default=False):
                    #     text = input("Instruction?   ")
                    # Format task for the model
                    need_to_exit = False
                    if isinstance(command, MultimodalCommand):
                        response, should_exit = click_exit_wrapper(
                            click.confirm,
                            {
                                "text": "Proceed to next annotation type?",
                                "default": True,
                            },
                        )
                        if should_exit:
                            break
                        if response is True:
                            text = command.next_annotation()
                        else:
                            while True:
                                try:
                                    command_idx = input(
                                        "Get index (0=viz, 1=tac, 2=viz/tac, 3=viz/aud, 4=tac/aud, 5=viz/tac/aud, or empty input for no change):     "
                                    )
                                    if command_idx == "":
                                        text = command.curr_annotation()
                                        break
                                    command.index = int(command_idx)
                                    text = command.curr_annotation()
                                    break
                                except Exception as e:
                                    print(f"Received exception {e}")
                                    need_to_exit = click_exit_wrapper(
                                        click.confirm,
                                        {"text": "exit?  ", "default": False},
                                    )
                                    if need_to_exit:
                                        break

                    if need_to_exit:
                        break
                    task = model.create_tasks(texts=[text])
                    # For logging purposes
                    goal_instruction = text
                    goal_image = jnp.zeros_like(goal_image)
                else:
                    raise NotImplementedError()

                # reset env
                images = []
                wrist_images = []
                obs, _ = env.reset()

                time.sleep(2.0)
                print(
                    f"Model:   {curr_name}\nWindow size:   {FLAGS.window_size}\nInstruction:    {text}"
                )
                input("Press [Enter] to start.")

                # do rollout
                last_tstep = time.time()

                goals = []
                t = 0
                pred_acs = []
                WINDOW_SIZE = FLAGS.window_size
                pad_mask = np.array([[True for _ in range(WINDOW_SIZE)]])[0]
                while t < FLAGS.num_timesteps:
                    if time.time() > last_tstep + STEP_DURATION:
                        last_tstep = time.time()
                        step_info = {
                            "step": t,
                        }
                        print("\n#################################\n")
                        # recursive_dict_print(obs)
                        # for key, val in obs.items():
                        #     if isinstance(val, np.ndarray):
                        #         print(key, val.shape)
                        # save images
                        images.append(obs["image_primary"][-1])
                        wrist_images.append(obs["image_wrist"][-1])
                        goals.append(goal_image)
                        obs["image_digit_left"] = np.array(
                            obs["image_digit_left"], np.int16
                        )
                        obs["image_digit_right"] = np.array(
                            obs["image_digit_left"], np.int16
                        )
                        obs["image_digit_left_background"] = obs.pop("background_l")
                        obs["image_digit_right_background"] = obs.pop("background_r")
                        # obs = model.example_batch['observation']
                        remove_keys = [
                            "task_completed",
                            "timestep",
                            "pad_mask_dict",
                            "timestep_pad_mask",
                        ]
                        # for key in remove_keys:
                        #     obs.pop(key, None)
                        # for key, val in obs.items():
                        #     print(val.shape)
                        #     obs[key] = val[0]
                        #     print(obs[key].shape)
                        # print('\n#################################\n')
                        # recursive_dict_print(obs)
                        true_obs = {}
                        for key, val in model.example_batch["observation"].items():
                            if key in remove_keys:
                                continue
                            true_obs[key] = val[0]
                        # old_obs = obs
                        # obs = true_obs
                        print("\n\n####################################")
                        recursive_dict_print(obs)
                        print("####################################")
                        recursive_dict_print(true_obs)
                        print("####################################\n\n")

                        if FLAGS.show_image:
                            bgr_img = cv2.cvtColor(
                                obs["image_primary"][-1], cv2.COLOR_RGB2BGR
                            )
                            cv2.imshow("img_view", bgr_img)
                            cv2.waitKey(20)

                        obs["timestep_pad_mask"] = pad_mask
                        forward_pass_time = time.time()
                        pred_action = np.array(policy_fn(obs, task), dtype=np.float64)
                        pred_acs.append(pred_action[0].copy())
                        action = pred_action
                        step_info["forward_pass_time"] = time.time() - forward_pass_time

                        # perform environment step
                        start_time = time.time()
                        try:
                            obs, _, _, truncated, _ = env.step(action)
                        except KeyError:
                            reset_environment = True
                        step_info["step_time"] = time.time() - start_time
                        step_info["total_time"] = time.time() - last_tstep
                        info_str = ""
                        for key, val in step_info.items():
                            info_str += f"{key}:      {val}\n"
                        print(info_str)

                        t += 1

                        if truncated:
                            break
            except KeyboardInterrupt:
                pass

            # save video
            try:
                print(f"Model {curr_name} using command {goal_instruction}")
                if FLAGS.video_save_path is not None:
                    if wrist_images and images:
                        ckpt_name = curr_name
                        curr_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                        save_dir_path = os.path.join(
                            FLAGS.video_save_path, ckpt_name, curr_time
                        )
                        os.makedirs(save_dir_path)
                        video_path = os.path.join(save_dir_path, "video.mp4")
                        for i, im in enumerate(wrist_images):
                            wrist_images[i] = cv2.resize(im, (256, 256))

                        video = np.concatenate(
                            [np.stack(wrist_images), np.stack(images)], axis=1
                        )
                        # video=np.stack(images)
                        imageio.mimsave(
                            video_path, video, fps=1.0 / STEP_DURATION * 1.25
                        )
                        command_save_path = os.path.join(save_dir_path, "lang.txt")

                        info_string = f"{goal_instruction}\n\n{FLAGS.checkpoint_weights_path}:{FLAGS.checkpoint_step}"
                        with open(command_save_path, "wb") as file:
                            file.write(str.encode(info_string, "utf-8"))
            except KeyboardInterrupt:
                pass


if __name__ == "__main__":
    app.run(main)
