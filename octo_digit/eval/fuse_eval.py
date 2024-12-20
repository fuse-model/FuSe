"""
This script shows how we evaluated a finetuned Octo model on a real WidowX robot. While the exact specifics may not
be applicable to your use case, this script serves as a didactic example of how to use Octo in a real-world setting.

If you wish, you may reproduce these results by [reproducing the robot setup](https://rail-berkeley.github.io/bridgedata/)
and installing [the robot controller](https://github.com/rail-berkeley/bridge_data_robot)
"""

from typing import Union
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
from copy import deepcopy

from absl import app, flags, logging
import click
import cv2
from eval.envs.widowx_env import convert_obs, state_to_eep, wait_for_obs, WidowXGym, LostConnection
import imageio
import jax
import jax.numpy as jnp
import numpy as np
from widowx_envs.widowx_env_service import WidowXClient, WidowXConfigs, WidowXStatus
from octo.data.utils.text_processing import MuseEmbedding

import operator
from octo.model.octo_model import OctoModel
from octo.model.bcz_model import BczModel
from eval.envs.gym_wrappers import HistoryWrapper, ObsProcessingWrapper, RHCWrapper, ResizeImageWrapper, ResizeImageWrapperDict, TemporalEnsembleWrapper
import widowx_envs.utils.transformation_utils as tr
from PIL import Image
from collections import OrderedDict
from itertools import combinations, filterfalse
from eval.eval_config import  (
    WIDOWX_SERVICE_SIZE_MAP, 
    OBS_KEY_MAP, 
    CALCULATED_FIELDS, 
    NAME_TO_COMMAND, 
    NAME_TO_WINDOW_SIZE, 
    PATHS, 
    RESIZE_MAP, 
    MultimodalCommand, 
    AnnotationList, 
    GroupedAnnotations, 
    CommandType,
    RESPONSE_TO_RESULT,
    PRINT_OBS_INFO,
    TEXT_PROCESSOR,
    DECODE_LANG,
    DECODE_EXCLUDE,
    TAKE_INPUT
)
import tensorflow as tf
from tvl_embedder import TVLEmbedder 
from eval.recursive_dict_print import recursive_dict_print
from flax.traverse_util import flatten_dict, unflatten_dict

import os 

from eval.decode_language import get_language_decoded

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
    # "128.32.175.120",
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
flags.DEFINE_integer("num_timesteps", 100, "num timesteps")
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
FLIP_CHANNELS = False
Model = Union[OctoModel, BczModel]

# def get_digit_size(model: Model): 
#     try:
#         digit_size = model.example_batch['observation']['image_digit_left'].shape[-3:-1]
#         print('Digit size set to', digit_size)
#     except KeyError:
#         print('No digits found in example batch!')
#         digit_size = (256, 256) 
#     return digit_size

# def wrap_env(STATE): 
#     digit_size = get_digit_size(STATE.model)
#     resize_map = dict()
#     for k, v in RESIZE_MAP.items(): 
#         if 'digit' in k or 'background' in k: 
#             resize_map[k] = digit_size
#         else: 
#             resize_map[k] = v
    
#     env = ResizeImageWrapperDict(STATE.widowx_pure_env, resize_map)
#     STATE.env = ObsProcessingWrapper(env, OBS_KEY_MAP, CALCULATED_FIELDS, flip_channels=FLIP_CHANNELS, tvl_embedder=STATE.tvl_embedder)
#     STATE.pre_history_wrap = STATE.env
#     STATE.rewrap = True 
#     STATE.reset_server = False

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
    STATE.widowx_pure_env = WidowXGym(
        widowx_client, WIDOWX_SERVICE_SIZE_MAP, FLAGS.blocking, STICKY_GRIPPER_NUM_STEPS
    )
    
    # wrap the robot environment
    env = ResizeImageWrapperDict(STATE.widowx_pure_env, RESIZE_MAP)
    STATE.env = ObsProcessingWrapper(env, OBS_KEY_MAP, CALCULATED_FIELDS, flip_channels=FLIP_CHANNELS, tvl_embedder=tvl_embedder)
    STATE.pre_history_wrap = STATE.env
    STATE.rewrap = True 
    STATE.reset_server = False 


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
    STATE.model_idx = new_idx
    STATE.curr_name, STATE.curr_ckpt_step, STATE.model, STATE.policy_fn = STATE.info_models[STATE.model_idx]
    new_window_size = NAME_TO_WINDOW_SIZE[STATE.curr_name]
    # wrap_env(STATE)
    # # if STATE.rewrap or new_window_size != STATE.window_size: 
    # STATE.window_size = new_window_size
    # env = HistoryWrapper(STATE.pre_history_wrap, STATE.window_size)
    # STATE.env = TemporalEnsembleWrapper(env, FLAGS.exec_horizon)
    # STATE.rewrap  = False 
    if STATE.rewrap or new_window_size != STATE.window_size: 
        STATE.window_size = new_window_size
        env = HistoryWrapper(STATE.pre_history_wrap, STATE.window_size)
        STATE.env = TemporalEnsembleWrapper(env, FLAGS.exec_horizon)
        STATE.rewrap  = False 
    STATE.command = NAME_TO_COMMAND(FLAGS.mode, STATE.curr_name)
    if isinstance(STATE.command, str): 
        STATE.text = STATE.command 
    elif isinstance(STATE.command, GroupedAnnotations):
        STATE.command.reset()
        STATE.text = STATE.command.curr_annotation()
    else: 
        raise ValueError(STATE.command)

def main(_):
    STATE = params() 
    eval_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if "digit_embeddings" in CALCULATED_FIELDS: 
        tvl_embedder = TVLEmbedder('cuda:0')
    else: 
        tvl_embedder = None 
    STATE.tvl_embedder = tvl_embedder


    if FLAGS.paths_file is not None: 
        with open(FLAGS.paths_file, 'r') as file:
            FLAGS.paths = [line.strip() for line in file if line.strip() and '#' not in line.strip()] 
            if FLAGS.paths[0].find(' ') < 0:
                FLAGS.paths = [(path, 50_000) for path in FLAGS.paths]
            else: 
                FLAGS.paths = [[s.strip() for s in path.split(' ')] for path in FLAGS.paths]
    models = []
    
    def calculate_num_parameters(model): 
        param_sizes = jax.tree_map(lambda x: x.size, model.params)
        return jax.tree_util.tree_reduce(operator.add, param_sizes)
    
    def calculate_head_params(model): 
        params = model.params
        flat_params = flatten_dict(params)
        gen_head_params = {k: v for k, v in flat_params.items() if k[0].startswith('heads_gen')}
        total_size = 0
        for k, v in gen_head_params.items(): 
            total_size += v.size
        print(total_size)
    
    for name, ckpt_step in FLAGS.paths: 
        full_name = name 
        if not name.startswith('/'): 
            full_name = os.path.join(FLAGS.ckpt_dir, name)
    
        
        load_kwargs = {
            "checkpoint_path": full_name, 
            "step": ckpt_step,
        }    
        if TEXT_PROCESSOR: 
            load_kwargs['text_processor_spec'] = TEXT_PROCESSOR
        print(full_name)
        model_name = full_name.split('/')[-1]
        if model_name.startswith('josh_pod_final_bcz'): 
            ModelType = BczModel
        else: 
            ModelType = OctoModel
        print(f"Loading model {name} at step {ckpt_step}...")
        model = ModelType.load_pretrained(
            **load_kwargs
        )
        
        print(f"Model with {calculate_num_parameters(model)} parameters loaded!\n")
        models.append((name, ckpt_step, model))


    def sample_actions(
        pretrained_model: Model,
        observations,
        tasks,
        rng, 

    ):
        # if isinstance(pretrained_model, OctoModel):
        #     print(pretrained_model.config['model']['observation_tokenizers'].keys())
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
        for _, _, model in models 
    ]
    
    
    def test_with_example_batch(model: Model, name: str, policy_function, num_samples=20): 
        observations = jax.tree_map(lambda arr: arr[0], model.example_batch['observation'])
        tasks = model.create_tasks(texts=['Dummy command'])
        start_compile = time.time()
        policy_function(observations, tasks)
        end_compile = time.time()
        
        start_compute = time.time()
        for _ in range(num_samples): 
            policy_function(observations, tasks)
        end_compute = time.time()
        
        info_str = (
            f'Model {name} with {calculate_num_parameters(model)} parameters took {end_compile - start_compile} ' 
            f'seconds to compile, and averaged {(end_compute - start_compute) / num_samples} s per sample.'
        )
        print(info_str)
        
    STATE.info_models = [(name, ckpt_step, model, policy_fn) for (name, ckpt_step, model), policy_fn in zip(models, policy_fns)]
    
    
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
    
        initialize_widowx_env(FLAGS, STATE, env_params, tvl_embedder)

        if FLAGS.compile_early: 
            print('Pre-compiling policy fns...')
            dummy_env = HistoryWrapper(STATE.pre_history_wrap, 2)
            dummy_env = TemporalEnsembleWrapper(dummy_env, FLAGS.exec_horizon)
            # dummy_env = wrap_env(STATE)
            dummy_obs, _ = dummy_env.reset()
            dummy_obs['timestep_pad_mask'] = np.array([[True for _ in range(2)]])[0]
            recursive_dict_print(dummy_obs) 
            prev_window_size = 2
            digit_size = (-1, -1)
            for name, ckpt_step, model, policy_fn in STATE.info_models:            
                    print(f'Compiling policy for ({name}, {ckpt_step})...') 
                    new_window_size = NAME_TO_WINDOW_SIZE[name]
                    if new_window_size != prev_window_size: 
                        obs = {} 
                        for k, v in dummy_obs.items(): 
                            # print(k, v.shape)
                            no_window_v = v[0]
                            obs[k] = np.stack([no_window_v.copy() for _ in range(new_window_size)])
                    else: 
                        obs = dummy_obs
                    # new_digit_size = get_digit_size(model)

                    # def resize_digit_images(arr, new_size): 
                    #     prev_shape = arr.shape
                    #     print('prev shape', prev_shape)
                    #     prev_dig_shape = arr.shape[-3:-1]
                    #     print('prev dig shape', prev_dig_shape)
                    #     flat_arr = jnp.reshape(arr, (-1, *prev_dig_shape, 3))
                    #     print('flat arr', flat_arr.shape)
                    #     reshaped_flat_arr = []
                    #     for img in flat_arr: 
                    #         print('img', img.shape, new_size)
                    #         new_img = tf.image.resize(
                    #             img, size=new_size, method="lanczos3", antialias=True
                    #         )
                    #         reshaped_flat_arr.append(new_img)
                    #     reshaped_arr = jnp.reshape(jnp.array(reshaped_flat_arr), (*prev_shape[:-3], *new_size, 3))
                    #     print(reshaped_arr.shape)
                    #     return reshaped_arr
                    
                    # if digit_size != new_digit_size: 
                    #     for k, v in obs.items(): 
                    #         if 'digit' in k or 'background' in k: 
                    #             obs[k] = resize_digit_images(v, new_digit_size)
                    #         else: 
                    #             obs[k] = v
                    # digit_size = new_digit_size
                    
                    dummy_command: CommandType = NAME_TO_COMMAND(FLAGS.mode, name)
                    if isinstance(dummy_command, str): 
                        dummy_text = dummy_command 
                    elif isinstance(dummy_command, GroupedAnnotations): 
                        dummy_command.reset()
                        dummy_text = dummy_command.curr_annotation()
                    else: 
                        raise ValueError(dummy_command)
                    print(dummy_text, type(dummy_text))
                    dummy_task = model.create_tasks(texts=[dummy_text])
                    policy_fn(obs, dummy_task)
                    if DECODE_LANG and name not in DECODE_EXCLUDE:
                        model.decode_language(obs)
                    # for _ in range(3):
                    #     model.decode_language(dummy_obs) 
                    
            
            
            
        # goal sampling loop
        STATE.model_idx = 0
        STATE.run_idx = 0   
        STATE.text = 'DUMMY: REPLACE!'
        STATE.command = ''
        STATE.curr_name = ''
        STATE.go_next = False
        STATE.is_first_run = True 
        change_to_model(STATE, STATE.model_idx)

        finished_run = True
        while True:
            STATE.should_log = False 
           
            try: 
                modality = "l"

                if modality == "g":
                    raise NotImplementedError

                elif modality == "l":
                    response, should_exit = click_exit_wrapper(
                        click.confirm, 
                        text= "Reset environment?   ", 
                        default=STATE.reset_server
                    )
                    if should_exit: 
                        break
                    if response: 
                        input('Restart server and then hit enter:   ')
                        initialize_widowx_env(FLAGS, STATE, env_params, tvl_embedder)
                        env = HistoryWrapper(STATE.pre_history_wrap, STATE.window_size)
                        STATE.env = TemporalEnsembleWrapper(env, FLAGS.exec_horizon)
                        STATE.reset_server = False 
                        STATE.rewrap = False 

                    if finished_run and STATE.model_idx == len(STATE.info_models) - 1: 
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
                                got_response = False 
                                while True:
                                    response = input('Enter new run index (0 - 9):    ') 
                                    new_run = False 
                                    try: 
                                        new_run_idx = int(response.strip())
                                        if new_run_idx not in range(10): 
                                            raise ValueError
                                        STATE.run_idx = new_run_idx
                                        new_run = (STATE.run_idx != new_run_idx)
                                        break
                                    except (ValueError, KeyboardInterrupt) as e: 
                                        print(f'Invalid response.')

                    response, should_exit = click_exit_wrapper(click.confirm, text=f'Change run index? currently {STATE.run_idx}:      ', default=STATE.is_first_run)
                    STATE.is_first_run = False
                    if should_exit: 
                        break 
                    elif response:
                            response = input('Enter new run index (0 - 9):    ') 
                            if response == '' or response.lower() == 'n': 
                                STATE.run_idx  = (STATE.run_idx + 1 ) % 10 
                            else: 
                                try: 
                                    new_run_idx = int(response.strip())
                                    if new_run_idx not in range(10): 
                                        raise Value
                                    STATE.run_idx = new_run_idx
                                except ValueError: 
                                    print(f'Invalid response. Continuing with run index = {STATE.run_idx}')
                    print('#############################')
                    print('Current info:   ')
                    print(f'Mode:          {FLAGS.mode}')
                    print(f'Run:           {STATE.run_idx}')
                    print(f'Model:         {STATE.curr_name}')
                    print(f'Step:          {STATE.curr_ckpt_step}')
                    print(f'Model idx:     {STATE.model_idx}')
                    print(f'Annotation:    {STATE.text}')
                    print('#############################')
                    print('Next model/annotation info:     ')
                    if isinstance(STATE.command, str) or STATE.command.is_last(): 
                            
                        next_model_idx = (STATE.model_idx + 1 ) % len(STATE.info_models)
                        next_model_name, next_ckpt_step = STATE.info_models[next_model_idx][:2]
                        next_command = NAME_TO_COMMAND(FLAGS.mode, next_model_name)
                        if isinstance(next_command, str): 
                            next_annotation = next_command 
                        else: 
                            next_annotation = next_command.peek_next()
                    else: 
                        next_model_idx = STATE.model_idx 
                        next_model_name = STATE.curr_name 
                        next_ckpt_step = STATE.curr_ckpt_step
                        # next_annotation = STATE.command.all_annotations[STATE.command.index + 1]
                        next_annotation = STATE.command.peek_next()

                    print(f'Model:         {next_model_name}')
                    print(f'Step:          {next_ckpt_step}')
                    print(f'Model idx:     {next_model_idx}')
                    print(f'Annotation:    {next_annotation}')
                    print('#############################')

                        
                    response, should_exit = click_exit_wrapper(click.confirm, 
                        text="Proceed to next model/annotation pairing?", 
                        default=STATE.go_next 
                    )

                    if should_exit: 
                        break
                    elif response:
                        if next_model_idx != STATE.model_idx: 
                            change_to_model(STATE, next_model_idx)
                        else: 
                            STATE.text = STATE.command.next_annotation()
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
                                for i, (name, ckpt_num, _, _) in enumerate(STATE.info_models): 
                                    print(f'{i}     {name}     step: {ckpt_num}')  
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
                                info_str = ', '.join([f'{i}={key}' for i, key in enumerate(STATE.command.key_list)]) 
                                while True: 
                                    command_idx = input(
                                        f'Choose index    ({info_str}):     '
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

                    if TAKE_INPUT:
                        STATE.text = input('Enter command:   ')
                    
                    STATE.task = STATE.model.create_tasks(texts=[STATE.text])
                    # For logging purposes
                    goal_instruction = STATE.text
                    goal_image = jnp.zeros_like(goal_image)
                else:
                    raise ValueError

                # reset env
                images = []
                wrist_images = []
                dig_left_images=  []
                dig_right_images = [] 
                obs, _ = STATE.env.reset()

                STATE.task = STATE.model.create_tasks(texts=[STATE.text])
                time.sleep(1.0)
                print(f'Model:   {STATE.curr_name}\nStep: {STATE.curr_ckpt_step}\nWindow size:   {STATE.window_size}\nInstruction:    {STATE.text}')
                input("Press [Enter] to start.")

                # do rollout
                last_tstep = time.time()
                
                goals = []
                t = 0
                pred_acs = [] 
                WINDOW_SIZE = STATE.window_size
                pad_mask = np.array([[True for _ in range(WINDOW_SIZE)]])[0]
                STATE.go_next = False 
                while t < FLAGS.num_timesteps:
                    if time.time() > last_tstep + STEP_DURATION:
                        if t > 5: 
                            STATE.should_log = True 
                            STATE.go_next = True 
                        if PRINT_OBS_INFO: 
                            recursive_dict_print(obs)
                            print('########')
                            print(recursive_dict_print(STATE.model.example_batch['observation']))
                        last_tstep = time.time()
                        step_info = { 
                            'step': t, 
                        }
                        
                        images.append(obs["image_primary"][-1])
                        wrist_images.append(obs["image_wrist"][-1])
                        dig_left_images.append(obs['dig_left'][-1])
                        dig_right_images.append(obs['dig_right'][-1])
                        goals.append(goal_image)


                        if FLAGS.show_image:
                            bgr_img = cv2.cvtColor(obs["image_primary"][-1], cv2.COLOR_RGB2BGR)
                            cv2.imshow("img_view", bgr_img)
                            cv2.waitKey(20)

                        obs['timestep_pad_mask'] = pad_mask
                        forward_pass_time = time.time()

                        pred_action = np.array(STATE.policy_fn(obs, STATE.task), dtype=np.float64)
                        pred_acs.append(pred_action[0].copy())
                        action = pred_action
                        step_info['forward_pass_time'] = time.time() - forward_pass_time

                        # perform environment step
                        start_time = time.time()
                        try: 
                            obs, _, _, truncated, _ = STATE.env.step(action)
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
                        # print(obs['xyz'].shape)
                        # print(obs['xyz'][-1][-1])
                        # out = STATE.model.decode_language(obs)
                        # gen_str = '|'.join([v[0] for k, v in out.items()])
                        # print(gen_str)

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
                    print('Text was:     ', STATE.text)
                    print('Logging commands: ')
                    for k, v in RESPONSE_TO_RESULT.items(): 
                        print(f'Use {k} for:    {v}')
                    while not got_response: 
                        # response = input('Enter s for success, m for a find, f for failure, or n to avoid logging:     ').lower().strip()
                        response = input('Enter response:       ')
                        
                        if response in RESPONSE_TO_RESULT: 
                            run_result = RESPONSE_TO_RESULT[response]
                            got_response = True 
                    if run_result != 'exit': 
                        curr_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                        model_log_dir = os.path.join(FLAGS.eval_log_dir, STATE.curr_name, str(STATE.curr_ckpt_step))
                        if not os.path.exists(model_log_dir): 
                            os.makedirs(model_log_dir)
                        log_file_path = os.path.join(model_log_dir, 'log.txt')

                        if isinstance(STATE.command, str): 
                            cmd_key = 'simple'
                        else: 
                            cmd_key = STATE.command.curr_key

                        video_dir = os.path.join(model_log_dir, f'{FLAGS.mode}', f'{cmd_key}', f'{STATE.run_idx}')
                        video_path = os.path.join(video_dir, f'{curr_time}.mp4')

                        os.makedirs(video_dir, exist_ok=True)
                        info_str = f'mode:   {FLAGS.mode}    run: {STATE.run_idx}    command: {STATE.text}     result:   {run_result}    video path:    {video_path}\n'
                        with open(log_file_path, 'a') as log_file: 
                            log_file.write(info_str)
                
                        for i, (im_w, im_l, im_r) in enumerate(zip(wrist_images, dig_left_images, dig_right_images)): 
                            wrist_images[i] = cv2.resize(im_w, (256, 256))
                            dig_left_images[i] = cv2.resize(im_l, (256, 256))
                            dig_right_images[i] = cv2.resize(im_r, (256, 256))
                            
                        video = np.concatenate([np.stack(wrist_images), np.stack(images), np.stack(dig_left_images), np.stack(dig_right_images)], axis=1)
                        imageio.mimsave(video_path, video, fps=1.0 / STEP_DURATION * 1.25)
                        with open(os.path.join(video_dir, f'log.txt'), 'a') as file: 
                            file.write(info_str)
                        # print('wrote to video dir', video_dir)
                        finished_logging = True 
                        # print(DECODE_LANG, STATE.curr_name)
                        
                        should_show, should_exit = click_exit_wrapper(click.confirm, text="Log images?   ", default=False)
                        if should_show:
                            overhead = cv2.resize(images[-1], (640, 480))
                            Image.fromarray(overhead).save(os.path.join(video_dir, f'overhead.png'))
                            wrist = cv2.resize(wrist_images[-1], (320, 240))
                            Image.fromarray(wrist).save(os.path.join(video_dir, f'wrist.png'))
                            dig_left = cv2.resize(dig_left_images[-1], (320, 240))
                            Image.fromarray(dig_left).save(os.path.join(video_dir, f'dig_l.png'))
                            dig_right = cv2.resize(dig_right_images[-1], (320, 240))
                            Image.fromarray(dig_right).save(os.path.join(video_dir, f'dig_r.png'))
                        if DECODE_LANG and STATE.curr_name not in DECODE_EXCLUDE:
                            print('\n\n')
                            print('Decoding language from last observation...')
                            out = STATE.model.decode_language(obs)
                            gen_str = '|'.join([v[0] for k, v in out.items()])
                            gen_log_file_path = os.path.join(video_dir, 'gen.txt')
                            print('generated:', gen_str)
                            with open(gen_log_file_path, 'a') as gen_log_file: 
                                gen_log_file.write(gen_str + '\n')
                            
            except KeyboardInterrupt: 
                pass         

    
                
                

if __name__ == "__main__":
    app.run(main)
