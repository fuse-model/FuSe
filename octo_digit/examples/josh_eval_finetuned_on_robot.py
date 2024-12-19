"""
This script shows how we evaluated a finetuned Octo model on a real WidowX robot. While the exact specifics may not
be applicable to your use case, this script serves as a didactic example of how to use Octo in a real-world setting.

If you wish, you may reproduce these results by [reproducing the robot setup](https://rail-berkeley.github.io/bridgedata/)
and installing [the robot controller](https://github.com/rail-berkeley/bridge_data_robot)
"""

import pickle
from datetime import datetime
from functools import partial
import os
import time

from absl import app, flags, logging
import click
import cv2
from envs.widowx_env import convert_obs, state_to_eep, wait_for_obs, WidowXGym
import imageio
import jax
import jax.numpy as jnp
import numpy as np
from widowx_envs.widowx_env_service import WidowXClient, WidowXConfigs, WidowXStatus
from octo.data.utils.text_processing import MuseEmbedding

from octo.model.octo_model import OctoModel
from octo.model.resnet_model import ResnetModel
from octo.utils.gym_wrappers import HistoryWrapper, ObsProcessingWrapper, RHCWrapper, ResizeImageWrapper, ResizeImageWrapperDict, TemporalEnsembleWrapper
import widowx_envs.utils.transformation_utils as tr
from PIL import Image

np.set_printoptions(suppress=True)

logging.set_verbosity(logging.WARNING)

FLAGS = flags.FLAGS


#TODO(CARLO): fill in checkpoint and step here 
flags.DEFINE_string(
    "checkpoint_weights_path",

    # "standard_20240605_020204",
    # "standard_largebatch_20240605_015537",
    # "smallstem_32_full_largebatch_20240605_015512",
    # "smallstem_32_full_20240605_015049",
    # "smallstem_32_b2_20240610_212524",
    # "resnet26_20240613_035123",

    # "combined_dataset_20240606_223700",
    # "combined_dataset_s16_b128_20240606_224813",
    # "combined_dataset_s32_b128_20240606_225641",


    # "filtered_dataset_s16_b256_20240606_225715",
    # "filtered_dataset_s16_b128_20240606_225248",
    # "filtered_dataset_s32_b128_20240606_230028",


    # "cams_only_w2_20240611_194935",
    # "digits_mic_20240611_225735",
    # "digits_20240611_225915",
    # "mic_20240611_225957",

    # "best_8k_20240613_033249",

    # "cams_only_23_20240614_014341",
    # "cams_mic_23_20240614_014343",
    # "cams_mic_digit_23_20240614_014346",
    # "cams_digit_23_20240614_014348",

    # "cams_25_20240617_042143",
    # "all_modalities_25_20240617_101010",


    "Path to checkpoint"
)

NAME_TO_WINDOW_SIZE = { 
    "standard_20240605_020204": 4,
    "standard_largebatch_20240605_015537": 3,
    "smallstem_32_full_largebatch_20240605_015512": 10,
    "smallstem_32_full_20240605_015049": 16,
    "smallstem_32_b2_20240610_212524": 2,
    "resnet26_20240613_035123": 3,

    "combined_dataset_20240606_223700": 3,
    "combined_dataset_s16_b128_20240606_224813": 4,
    "combined_dataset_s32_b128_20240606_225641": 15,


    "filtered_dataset_s16_b256_20240606_225715": 3,
    "filtered_dataset_s16_b128_20240606_225248": 4,
    "filtered_dataset_s32_b128_20240606_230028": 15,


    "cams_only_w2_20240611_194935": 2,
    "digits_mic_20240611_225735": 2,
    "digits_20240611_225915": 2,
    "mic_20240611_225957": 2,

    "best_8k_20240613_033249": 3,

    "cams_only_23_20240614_014341": 2,
    "cams_mic_23_20240614_014343": 2,
    "cams_mic_digit_23_20240614_014346": 2,
    "cams_digit_23_20240614_014348": 2,

    "cams_25_20240617_042143": 2,
    "all_modalities_25_20240617_101010": 2,
}



flags.DEFINE_bool(
    "is_resnet", 
    False, 
    "if ckpt is a resnet-only model"
)

flags.DEFINE_integer(
    "checkpoint_step", 
    50000, 
    "Checkpoint step"
)

dell2_direct = "192.168.99.10"
dell2_ether = "128.32.175.252"
dell2 = dell2_ether
# custom to bridge_data_robot
flags.DEFINE_string("ip", 
dell2, 
"IP address of the robot")
flags.DEFINE_integer("port", 5556, "Port of the robot")
flags.DEFINE_spaceseplist("goal_eep", [0.3, 0.0, 0.15], "Goal position")
flags.DEFINE_spaceseplist("initial_eep",[0.11796844, -0.01554691,  0.23344009], "Initial position") # neutral 

flags.DEFINE_bool("blocking", False, "Use the blocking controller")


flags.DEFINE_integer(
    "im_size", 
    256, 
    "Image size", 
    #required=True
)

flags.DEFINE_string("video_save_path", "./videos", "Path to save video")
flags.DEFINE_integer("num_timesteps", 80, "num timesteps")
flags.DEFINE_integer("window_size", 2, "Observation history length")
flags.DEFINE_integer("exec_horizon", 1, "Length of action sequence to execute")


# show image flag
flags.DEFINE_bool("show_image", False, "Show image")

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
    {"name": "/wrist/image_raw", "is_python_node": True}
]
DIGIT_TOPICS =  [
    {"name": '/digit_left/image_raw', "width": 320, "height":240, "is_python_node": True},
    {"name": '/digit_right/image_raw', "width": 320, "height":240, "is_python_node": True}
]


ENV_PARAMS = {
    "camera_topics": CAMERA_TOPICS,
    "digit_topics": DIGIT_TOPICS, 
    "override_workspace_boundaries": WORKSPACE_BOUNDS,
    "move_duration": STEP_DURATION,
    # 'imu_topic': '/imu/imu_raw',
    'mic_topic': '/mic/mic_raw',
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
    # "spectro": None, 
    "spectro": "mel_spectro", 
    # "imu": "imu"
}
CALCULATED_FIELDS = [
    # "spectro", 
    # "digit_embeddings"
    
]

#TODO(CARLO): Try with this False (probably better) and True
FLIP_CHANNELS = False



def main(_):
    name = FLAGS.checkpoint_weights_path
    if "/" not in FLAGS.checkpoint_weights_path: 
        FLAGS.checkpoint_weights_path = os.path.join('/home/joshwajones/tpu_octo_ckpts', FLAGS.checkpoint_weights_path)
    
    FLAGS.window_size = NAME_TO_WINDOW_SIZE[name]  
    
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
        load_kwargs["text_processor"] =  MuseEmbedding()
        with open('./dataset_stats.pkl', 'rb') as file: 
            loaded_dataset_stats = pickle.load(file)
        load_kwargs['dataset_statistics'] = loaded_dataset_stats

    else: 
        MODELTYPE = OctoModel

    print("Loading model...")
    model = MODELTYPE.load_pretrained(
        **load_kwargs
    )
    print("Model loaded!")


    def recursive_dict_print(dic, sep=""): 
        for key, val in dic.items(): 
            print(key)
            if isinstance(val, dict): 
                recursive_dict_print(val, sep + "      ")
            elif isinstance(val, np.ndarray): 
                try: 
                    temp = val 
                    while len(temp) > 1: 
                        temp = temp[0]
                    print(" ", temp[0])
                except: 
                    print(val)
            else: 
                print(" ", val)


    dataset_statistics =  loaded_dataset_stats if FLAGS.is_resnet else model.dataset_statistics 
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
            unnormalization_statistics=dataset_statistics["action"],
            rng=rng,
        )
        # remove batch dim
        return actions[0]

    policy_fn = partial(
        sample_actions,
        model,
        rng=jax.random.PRNGKey(0)
    )

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
            'image_0': (256, 256), 
            'image_1': (128, 128),
            'digit_l': (256, 256),
            'digit_r': (256, 256),
        }
        env = ResizeImageWrapperDict(env, resize_map)
        env = ObsProcessingWrapper(env, OBS_KEY_MAP, CALCULATED_FIELDS, flip_channels=FLIP_CHANNELS)
        env = HistoryWrapper(env, FLAGS.window_size)
        env = TemporalEnsembleWrapper(env, FLAGS.exec_horizon)
        #env = RHCWrapper(env, FLAGS.exec_horizon)


        images = []
        wrist_images = []


        # goal sampling loop
        while True:
            try: 
                modality = "l"

                if modality == "g":
                    # if click.confirm("Take a new goal?", default=True):
                    #     assert isinstance(FLAGS.goal_eep, list)
                    #     _eep = [float(e) for e in FLAGS.goal_eep]
                    #     goal_eep = state_to_eep(_eep, 0)
                    #     widowx_client.move_gripper(1.0)  # open gripper

                    #     move_status = None
                    #     while move_status != WidowXStatus.SUCCESS:
                    #         move_status = widowx_client.move(goal_eep, duration=1.5)

                    #     input("Press [Enter] when ready for taking the goal image. ")
                    #     obs = wait_for_obs(widowx_client)
                    #     obs = convert_obs(obs, FLAGS.im_size)
                        # goal = jax.tree_map(lambda x: x[None], obs)
                    # goal_step = 25
                    # goal_dir = '/home/joshwajones/dataset_processing/curr_dataset/2024-04-28_02-08-56/raw/traj_group0/traj5'
                    # goal_im_0_path = os.path.join(goal_dir, 'images0', f'im_{goal_step}.jpg')
                    # img0 = np.asarray(Image.open(goal_im_0_path)) 
                    # img0 = tf.image.resize(img0, (256, 256), method="lanczos3", antialias=True)
                    # img0 = tf.cast(tf.clip_by_value(tf.round(img0), 0, 255), np.uint8)
                    # img0 = np.asarray(img0, dtype=np.uint8)
                    # img0 = np.zeros_like(img0)


                    # goal_im_1_path = os.path.join(goal_dir, 'images1', f'im_{goal_step}.jpg')
                    # img1 = np.asarray(Image.open(goal_im_1_path)) 
                    # img1 = tf.image.resize(img1, (128, 128), method="lanczos3", antialias=True)
                    # img1 = tf.cast(tf.clip_by_value(tf.round(img1), 0, 255), np.uint8)
                    # img1 = np.asarray(img1, dtype=np.uint8)


                    # obs = {"image_primary": img0, "image_wrist": img1}
                    # goal = jax.tree_map(lambda x: x[None], obs)
                    # # goal = convert_obs(goal_obs)
                    
                    # lang_path = os.path.join(goal_dir, 'lang.txt')
                    # with open(lang_path, 'rb') as file: 
                    #     text = file.read().decode('utf-8')
                    # # Format task for the model
                    # task = model.create_tasks(goals=goal)
                    # # For logging purposes
                    
                    # goal_image = goal["image_primary"][0]
                    # goal_instruction = "Grab the object that looks yellow, small, round."
                    raise NotImplementedError

                elif modality == "l":
                    print("Current instruction:  ", goal_instruction)
                    if click.confirm("Take a new instruction?   ", default=True):
                        text = input("Instruction?   ")
                    # Format task for the model
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
                        print(obs.keys())
                        last_tstep = time.time()

                        # save images
                        images.append(obs["image_primary"][-1])
                        wrist_images.append(obs["image_wrist"][-1])
                        goals.append(goal_image)

                        if FLAGS.show_image:
                            bgr_img = cv2.cvtColor(obs["image_primary"][-1], cv2.COLOR_RGB2BGR)
                            cv2.imshow("img_view", bgr_img)
                            cv2.waitKey(20)

                        obs['timestep_pad_mask'] = pad_mask
                        forward_pass_time = time.time()
                        pred_action = np.array(policy_fn(obs, task), dtype=np.float64)
                        pred_acs.append(pred_action[0].copy())
                        action = pred_action
                        print("forward pass time: ", time.time() - forward_pass_time)

                        # perform environment step
                        start_time = time.time()
                        try: 
                            obs, _, _, truncated, _ = env.step(action)
                        except KeyError: 
                            reset_environment = True
                        print("step time: ", time.time() - start_time)

                        t += 1

                        if truncated:
                            break
            except KeyboardInterrupt: 
                pass 

            # save video
            try: 
                if FLAGS.video_save_path is not None:
                    if wrist_images and images: 
                        ckpt_name = FLAGS.checkpoint_weights_path[FLAGS.checkpoint_weights_path.rfind('/') + 1:]
                        curr_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                        save_dir_path = os.path.join(FLAGS.video_save_path, ckpt_name, curr_time)
                        os.makedirs(save_dir_path)
                        video_path = os.path.join(save_dir_path, 'video.mp4')
                        for i, im in enumerate(wrist_images): 
                            wrist_images[i] = cv2.resize(im, (256, 256))
                            
                        video = np.concatenate([np.stack(wrist_images), np.stack(images)], axis=1)
                        # video=np.stack(images)
                        imageio.mimsave(video_path, video, fps=1.0 / STEP_DURATION * 1.25)
                        command_save_path = os.path.join(save_dir_path, 'lang.txt')

                        info_string = f'{goal_instruction}\n\n{FLAGS.checkpoint_weights_path}:{FLAGS.checkpoint_step}'
                        with open(command_save_path, 'wb') as file: 
                            file.write(str.encode(info_string, 'utf-8'))
            except KeyboardInterrupt: 
                pass 
                

if __name__ == "__main__":
    app.run(main)
