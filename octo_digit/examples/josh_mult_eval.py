"""
This script shows how we evaluated a finetuned Octo model on a real WidowX robot. While the exact specifics may not
be applicable to your use case, this script serves as a didactic example of how to use Octo in a real-world setting.

If you wish, you may reproduce these results by [reproducing the robot setup](https://rail-berkeley.github.io/bridgedata/)
and installing [the robot controller](https://github.com/rail-berkeley/bridge_data_robot)
"""

from ctypes import Union
from dataclasses import dataclass
from distutils.sysconfig import PREFIX
from email.policy import default
from multiprocessing.sharedctypes import Value
import pickle
from datetime import datetime
from functools import partial
import os
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
from widowx_envs.widowx_env_service import WidowXClient, WidowXConfigs, WidowXStatus
from octo.data.utils.text_processing import MuseEmbedding

from octo.model.octo_model import OctoModel
from octo.model.resnet_model import ResnetModel
from octo.utils.gym_wrappers import HistoryWrapper, ObsProcessingWrapper, RHCWrapper, ResizeImageWrapper, ResizeImageWrapperDict, TemporalEnsembleWrapper
import widowx_envs.utils.transformation_utils as tr
from PIL import Image
from collections import OrderedDict
from itertools import combinations, filterfalse
from eval_config import  OBS_KEY_MAP, CALCULATED_FIELDS, NAME_TO_COMMAND, NAME_TO_WINDOW_SIZE, PATHS, RESIZE_MAP, MultimodalCommand, RESPONSE_TO_RESULT
import tensorflow as tf
from tvl_embedder import TVLEmbedder 
from recursive_dict_print import recursive_dict_print
from flax.traverse_util import flatten_dict, unflatten_dict
from envs.widowx_env import LostConnection

import os 
VERBOSE = os.environ.get('VERBOSE', '').lower() == 'true'
RECUR_PRINT = os.environ.get('RECUR_PRINT', '').lower() == 'true'


np.set_printoptions(suppress=True)

logging.set_verbosity(logging.WARNING)


FLAGS = flags.FLAGS

class params: 
    pass 

flags.DEFINE_list(
    'paths',
    PATHS,
    '.'
)

flags.DEFINE_string(
    'paths_file', 
    None, 
    '.'
)

flags.DEFINE_bool(
    'compile_early', 
    True, 
    '.'
)

flags.DEFINE_string(
    "checkpoint_weights_path",

    "NULL",

    "Path to checkpoint"    
)

flags.DEFINE_string(
    'mode',
    'train', 
    '.'
)


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

flags.DEFINE_string(
    "ip", 
    "128.32.175.252", 
    "IP address of the robot"
)
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

flags.DEFINE_string("video_save_path", "./10k_videos", "Path to save video")
flags.DEFINE_integer("num_timesteps", 80, "num timesteps")
flags.DEFINE_integer("window_size", 2, "Observation history length")
flags.DEFINE_integer("exec_horizon", 1, "Length of action sequence to execute")


flags.DEFINE_string(
    "eval_log_dir", 
    './eval_logs', 
    '.'
)

# show image flag
flags.DEFINE_bool("show_image", False, "Show image")

flags.DEFINE_string('ckpt_dir', os.path.join(os.environ['HOME'], 'tpu_octo_ckpts', 'ckpts'), 'directory containing the checkpoints')

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

##############################################################################

##############################################################################
#TODO(CARLO): Try with this False (probably better) and True
FLIP_CHANNELS = False


def initialize_widowx_env(FLAGS, STATE, env_params, tvl_embedder): 
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
    env = ResizeImageWrapperDict(env, RESIZE_MAP)
    STATE.env = ObsProcessingWrapper(env, OBS_KEY_MAP, CALCULATED_FIELDS, flip_channels=FLIP_CHANNELS, tvl_embedder=tvl_embedder)
    STATE.pre_history_wrap = STATE.env
    STATE.reinit = True 
    STATE.need_to_reset = False 


def click_exit_wrapper(click_func, is_exit_checker=False, **click_kwargs):
    got_response = False
    should_quit_bool = False 
    while not got_response: 
        try: 
            response = click_func(**click_kwargs) 
            got_response = True 
        except click.exceptions.Abort as e:
            response = None 
            should_quit = click_exit_wrapper(click.confirm, is_exit_checker=True, text='Quit?', default=False) 
            should_quit_bool = should_quit[0] or should_quit[1] 
            
            if should_quit_bool: 
                got_response = True 
            if is_exit_checker and should_quit[0] is not None: 
                got_response = True 
                response = should_quit[0]
        
    return response, should_quit_bool


def change_to_model(STATE, new_idx): 
    STATE.curr_name, STATE.model, STATE.policy_fn = STATE.info_models[STATE.model_idx]
    new_window_size = NAME_TO_WINDOW_SIZE[STATE.curr_name]
    if STATE.reinit or new_window_size != STATE.window_size: 
        STATE.window_size = new_window_size
        env = HistoryWrapper(STATE.pre_history_wrap, STATE.window_size)
        STATE.env = TemporalEnsembleWrapper(env, FLAGS.exec_horizon)
        STATE.reinit = False 
    STATE.command = NAME_TO_COMMAND(FLAGS.mode, STATE.curr_name)
    if isinstance(STATE.command, str): 
        STATE.text = STATE.command 
    else: 
        STATE.command.reset()
        STATE.text = STATE.command.curr_annotation()

def main(_):
    STATE = params() 
    eval_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if "digit_embeddings" in CALCULATED_FIELDS: 
        tvl_embedder = TVLEmbedder('cuda:0')
    else: 
        tvl_embedder = None 


    if FLAGS.paths_file is not None: 
        with open(FLAGS.paths_file, 'r') as file:
            FLAGS.paths = [line.strip() for line in file if line.strip()] 
    models = []
    
    for name in FLAGS.paths: 
        full_name = name 
        if not name.startswith('/'): 
            full_name = os.path.join(FLAGS.ckpt_dir, name)
    
        
        load_kwargs = {
            "checkpoint_path": full_name, 
            "step": FLAGS.checkpoint_step,
        }
        
        print(full_name)
        print(f"Loading model   {name}...")
        model = OctoModel.load_pretrained(
            **load_kwargs
        )
        print("Model loaded!\n")
        models.append((name, model))


    def sample_actions(
        pretrained_model: OctoModel,
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
        partial(
        sample_actions,
        model,
        rng=jax.random.PRNGKey(0)
        )
        for _, model in models 
    ]
    
    STATE.info_models = [(name, model, policy_fn) for (name, model), policy_fn in zip(models, policy_fns)]
    STATE.reset_environment = True
    while STATE.reset_environment: 
        STATE.reset_environment = False 
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
    
        STATE.pre_history_wrap, STATE.env = initialize_widowx_env(FLAGS, STATE, env_params, tvl_embedder)

        if FLAGS.compile_early: 
            print('Pre-compiling policy fns...')
            prev_window_size = -1 
            for name, model, policy_fn in STATE.info_models:
                    print(f'Compiling policy for {name}...') 
                    new_window_size = NAME_TO_WINDOW_SIZE[name]
                    if new_window_size != prev_window_size: 
                        dummy_env = HistoryWrapper(STATE.pre_history_wrap, new_window_size)
                        dummy_env = TemporalEnsembleWrapper(dummy_env, FLAGS.exec_horizon)
                        prev_window_size = new_window_size

                    dummy_obs, _ = dummy_env.reset()
                    dummy_obs['timestep_pad_mask'] = np.array([[True for _ in range(new_window_size)]])[0]
                    recursive_dict_print(dummy_obs)
                    
                    dummy_command: Union[str, MultimodalCommand] = NAME_TO_COMMAND(FLAGS.mode, name)
                    if isinstance(dummy_command, str): 
                        dummy_text = dummy_command 
                    else: 
                        dummy_command.reset()
                        dummy_text = dummy_command.curr_annotation()
                    dummy_task = model.create_tasks(texts=[dummy_text])
                    policy_fn(dummy_obs, dummy_task)
            
            
        # goal sampling loop
        STATE.model_idx = 0
        STATE.run_idx = 0
        STATE.reinit = True 
        STATE.text = 'DUMMY: REPLACE!'
        STATE.command = ''
        STATE.uninitialized = True 
        STATE.need_to_reset = False 
        STATE.curr_name = ''
        STATE.go_next = False
        STATE.curr_window_size = -1 
        STATE.need_to_reset = False 
        STATE.is_first_run = True 
        change_to_model(STATE, STATE.model_idx)

        while True:
            STATE.should_log = False 
           
            try: 
                modality = "l"

                if modality == "g":
                    raise NotImplementedError

                elif modality == "l":
                    response, should_exit = click_exit_wrapper(
                        click.confirm, 
                        text= "Reset envrionment?   ", 
                        default=STATE.need_to_reset
                    )
                    if should_exit: 
                        break
                    if response: 
                        input('Restart server and then hit enter:   ')
                        initialize_widowx_env(FLAGS, STATE, env_params, tvl_embedder)
                        STATE.pre_history_wrap, env = initialize_widowx_env(FLAGS, env_params, tvl_embedder)
                        env = HistoryWrapper(STATE.pre_history_wrap, FLAGS.window_size)
                        STATE.env = TemporalEnsembleWrapper(env, FLAGS.exec_horizon)
                        STATE.need_to_reset = False 

                    if STATE.model_idx == len(STATE.info_models) - 1: 
                        if isinstance(STATE.command, str) or STATE.command.is_last(): 
                            response, should_exit = click_exit_wrapper(
                                click.confirm, 
                                text="At last model/annotation pairing. Proceed to new run?",
                                default=True
                            )
                            if should_exit: 
                                break 
                            elif response: 
                                print(f'Currently on run {STATE.run_idx}.')
                                response = input('Enter new run index (0 - 9):    ') 
                                if response == 's': 
                                    pass 
                                else: 
                                    new_run = False 
                                    if response == '' or response.lower() == 'n': 
                                        STATE.run_idx  = (STATE.run_idx + 1 ) % 10 
                                        new_run = True 
                                    else: 
                                        try: 
                                            new_run_idx = int(response.strip())
                                            if new_run_idx not in range(10): 
                                                raise ValueError
                                            STATE.run_idx = new_run_idx
                                            new_run = (STATE.run_idx != new_run_idx)
                                        except ValueError: 
                                            print(f'Invalid response. Continuing with run index = {STATE.run_idx}')

                    response, should_exit = click_exit_wrapper(click.confirm, text=f'Change run index? currently {run_idx}:      ', default=STATE.is_first_run)
                    STATE.is_first_run = False
                    if should_exit: 
                        break 
                    elif response:
                            response = input('Enter new run index (0 - 9):    ') 
                            if response == '' or response.lower() == 'n': 
                                run_idx  = (run_idx + 1 ) % 10 
                            else: 
                                try: 
                                    new_run_idx = int(response.strip())
                                    if new_run_idx not in range(10): 
                                        raise Value
                                    run_idx = new_run_idx
                                except ValueError: 
                                    print(f'Invalid response. Continuing with run index = {run_idx}')

                    print('Current info:   ')
                    print(f'Mode:          {FLAGS.mode}')
                    print(f'Run:           {STATE.run_idx}')
                    print(f'Model:         {STATE.curr_name}')
                    print(f'Model idx:     {STATE.model_idx}')
                    print(f'Annotation:    {STATE.text}')
                    print('##################')
                    print('Next model/annotation info:     ')
                    next_model_idx = (STATE.model_idx + 1 ) % len(STATE.info_models)
                    next_model_name = STATE.info_models[next_model_idx][0]
                    next_command = NAME_TO_COMMAND[next_model_name]
                    if isinstance(next_command, str): 
                        next_annotation = next_command 
                    else: 
                        next_annotation = next_command.all_annotations[0]
                    
                    if not (isinstance(STATE.command, str) or STATE.command.is_last()): 
                        next_model_name = STATE.curr_name
                        next_model_idx = STATE.model_idx 
                    print(f'Model:         {next_model_name}')
                    print(f'Model idx:     {next_model_idx}')
                    print(f'Annotation:    {next_annotation}')

                        
                    response, should_exit = click_exit_wrapper(click.confirm, 
                        text="Proceed to next model/annotation pairing?", 
                        default=STATE.go_next 
                    )

                    if should_exit: 
                        break
                    elif response:
                        change_to_model(STATE, next_model_idx)
                    else: 
                        response, should_exit = click_exit_wrapper(
                            click.confirm, 
                            text='Keep everything the same?', 
                            default= not STATE.go_next
                        )
                        if should_exit:
                            break 
                        elif not response:
                            response, should_exit = click_exit_wrapper(click.confirm, text="Change model?", default=False)
                            if should_exit: 
                                break 
                            elif response: 
                                for i, (name, _, _) in enumerate(STATE.info_models): 
                                    print(f'{i}     {name}')  
                                    response =  input('Choose model index:   ')
                                    new_model_idx = STATE.model_idx 
                                    if response == '' or response.lower() == 'n': 
                                        new_model_idx = (STATE.model_idx + 1) % len(STATE.info_models)
                                    else: 
                                        try: 
                                            new_model_idx = int(response)
                                        except ValueError as e: 
                                            print(f'Received error {e} as a result of entering {response}. Keeping index the same...')
                                    change_to_model(STATE, new_model_idx)
                            if isinstance(STATE.command, MultimodalCommand): 
                                info_str = ', '.join([f'{i}={key}' for i, key in enumerate(STATE.command.string_combos)]) 
                                while True: 
                                    command_idx = input(
                                        f'Choose index    ({info_str})'
                                    ).strip()
                                    if command_idx == '' or command_idx == 'n': 
                                        break 
                                    try: 
                                        command_idx = int(command_idx)
                                        break 
                                    except ValueError: 
                                        print('Invalid index, try again')
                                if command_idx == '': 
                                    STATE.text = STATE.command.curr_annotation()
                                elif command_idx == 'n': 
                                    STATE.text = STATE.command.curr_annotation()
                                else: 
                                    STATE.command.index = command_idx 
                                    STATE.text = STATE.command.curr_annotation()

                    STATE.task = model.create_tasks(texts=[STATE.text])
                    # For logging purposes
                    goal_instruction = STATE.text
                    goal_image = jnp.zeros_like(goal_image)
                else:
                    raise ValueError

                # reset env
                images = []
                wrist_images = []
                obs, _ = env.reset()
        
                time.sleep(2.0)
                print(f'Model:   {STATE.curr_name}\nWindow size:   {FLAGS.window_size}\nInstruction:    {STATEtext}')
                input("Press [Enter] to start.")

                # do rollout
                last_tstep = time.time()
                
                goals = []
                t = 0
                pred_acs = [] 
                WINDOW_SIZE = FLAGS.window_size
                pad_mask = np.array([[True for _ in range(WINDOW_SIZE)]])[0]
                STATE.go_next = False 
                while t < FLAGS.num_timesteps:
                    if time.time() > last_tstep + STEP_DURATION:
                        if t > 5: 
                            should_log = True 
                            STATE.go_next = True 
                        if RECUR_PRINT: 
                            recursive_dict_print(obs)
                            print('########')
                            print(recursive_dict_print(model.example_batch['observation']))
                        last_tstep = time.time()
                        step_info = { 
                            'step': t, 
                        }
                        
                        images.append(obs["image_primary"][-1])
                        wrist_images.append(obs["image_wrist"][-1])
                        goals.append(goal_image)


                        if FLAGS.show_image:
                            bgr_img = cv2.cvtColor(obs["image_primary"][-1], cv2.COLOR_RGB2BGR)
                            cv2.imshow("img_view", bgr_img)
                            cv2.waitKey(20)

                        obs['timestep_pad_mask'] = pad_mask
                        forward_pass_time = time.time()

                        pred_action = np.array(policy_fn(obs, STATE.task), dtype=np.float64)
                        pred_acs.append(pred_action[0].copy())
                        action = pred_action
                        step_info['forward_pass_time'] = time.time() - forward_pass_time

                        # perform environment step
                        start_time = time.time()
                        try: 
                            obs, _, _, truncated, _ = env.step(action)
                        except LostConnection: 
                            STATE.need_to_reset = True
                            STATE.reset_environment = True
                            STATE.should_log = False 
                            STATE.go_next = False 
                            break 
                        step_info['step_time'] = time.time() - start_time
                        step_info['total_time'] = time.time() - last_tstep
                        info_str =  ''
                        for key, val in step_info.items():
                            info_str += f'{key}:      {val}\n'
                        print(info_str)

                        t += 1

                        if truncated:
                            break
            except KeyboardInterrupt: 
                pass 

            should_log, should_exit = click_exit_wrapper(click.confirm, text="Log results?   ", default=STATE.should_log)
            if not should_log: 
                STATE.go_next = False 

            if should_exit: 
                break 
            try: 
                if should_log: 
                    got_response = False 
                    while not got_response: 
                        response = input('Enter s for success, m for a find, f for failure, or n to avoid logging:     ').lower().strip()
                        if response in RESPONSE_TO_RESULT: 
                            run_result = RESPONSE_TO_RESULT[response]
                            got_response = True 
                    if run_result != 'exit': 
                        curr_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                        model_log_dir = os.path.join(FLAGS.eval_log_dir, curr_name )
                        if not os.path.exists(model_log_dir): 
                            os.makedirs(model_log_dir)
                        log_file_path = os.path.join(model_log_dir, 'log.txt')
                        video_dir = os.path.join(model_log_dir, f'{FLAGS.mode}_{run_idx}', curr_time)
                        if not os.path.exists(video_dir): 
                            os.makedirs(video_dir)

                        info_str = f'mode:   {FLAGS.mode}    run: {run_idx}    command: {text}     result:   {run_result}    video path:    {video_dir}\n'
                        with open(log_file_path, 'a') as log_file: 
                            log_file.write(info_str)

                            log_path = os.path.join(FLAGS.eval_log_dir, curr_name, f'eval_{eval_timestamp}')
                
                        for i, im in enumerate(wrist_images): 
                            wrist_images[i] = cv2.resize(im, (256, 256))
                            
                        video = np.concatenate([np.stack(wrist_images), np.stack(images)], axis=1)
                        video_path = os.path.join(video_dir, 'video.mp4')
                        imageio.mimsave(video_path, video, fps=1.0 / STEP_DURATION * 1.25)
                        with open(os.path.join(video_dir, 'vid_log.txt'), 'a') as file: 
                            file.write(info_str)
                        finished_logging = True 
                        print('\n\n')
            except KeyboardInterrupt: 
                pass         

    
                
                

if __name__ == "__main__":
    app.run(main)
