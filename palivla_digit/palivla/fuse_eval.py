"""
This script shows how we evaluated a finetuned PaliVLA model on a real WidowX robot. While the exact specifics may not
be applicable to your use case, this script serves as a didactic example of how to use Octo in a real-world setting.

If you wish, you may reproduce these results by [reproducing the robot setup](https://rail-berkeley.github.io/bridgedata/)
and installing [the robot controller](https://github.com/rail-berkeley/bridge_data_robot)
"""

from datetime import datetime
from functools import partial
import os
import time

from absl import app, flags, logging
import click
import cv2
from octo_digit_eval.eval.envs.widowx_env import WidowXGym, LostConnection
import imageio
import jax
import jax.numpy as jnp
import numpy as np
from palivla import dataset
from palivla.types import ArrayTree, RolloutBatch
from widowx_envs.widowx_env_service import WidowXClient, WidowXConfigs
import librosa

import tensorflow as tf

import os 

import jax
import jax.numpy as jnp
import numpy as np
import sys
from functools import partial
from ml_collections import ConfigDict, FrozenConfigDict

from big_vision.models.proj.paligemma import paligemma
from big_vision.trainers.proj.paligemma import predict_fns

import tensorflow as tf
from tensorflow_text import SentencepieceTokenizer

from palivla.train_state import PaliVLATrainState
from palivla.tokenizer import Tokenizer
import orbax.checkpoint as ocp
from jax.sharding import NamedSharding, PartitionSpec as P
import jax_smi
from copy import deepcopy
from ml_collections import config_flags

from scalax.sharding import (
    MeshShardingHelper,
    FSDPShardingRule,
    PartitionSpec,
)
np.set_printoptions(suppress=True)

jax_smi.initialise_tracking()

sys.path.append(".")

jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
tf.config.set_visible_devices([], "GPU")

logging.set_verbosity(logging.WARNING)


FLAGS = flags.FLAGS


flags.DEFINE_string(
    "ip", 
    "128.32.175.252", 
    "IP address of the robot"
)
flags.DEFINE_integer("port", 5556, "Port of the robot")

flags.DEFINE_spaceseplist("initial_eep",[0.11796844, -0.01554691,  0.23344009], "Initial position") # neutral 

flags.DEFINE_bool("blocking", False, "Use the blocking controller")


flags.DEFINE_integer(
    "im_size", 
    256, 
    "Image size", 
)

flags.DEFINE_string("video_save_path", "./woven-wildflower-21", "Path to save video")
flags.DEFINE_integer("num_timesteps", 100, "num timesteps")


##############################################################################
import signal

class DelayedKeyboardInterrupt:
    def __enter__(self):
        self.signal_received = False
        self.old_handler = signal.signal(signal.SIGINT, self.handler)
                
    def handler(self, sig, frame):
        self.signal_received = (sig, frame)
    
    def __exit__(self, type, value, traceback):
        signal.signal(signal.SIGINT, self.old_handler)
        if self.signal_received:
            self.old_handler(*self.signal_received)
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
    'mic_topic': '/mic/mic_raw',
}


##############################################################################

def load_pretrained_model(config, mesh, model_sharding, data_sharding):
    restore_checkpoint_manager = ocp.CheckpointManager(
        config.resume_from_checkpoint_dir,
        item_handlers=PaliVLATrainState.get_checkpoint_handlers(),
    )
    model_train_state = PaliVLATrainState.restore(
        checkpoint_manager=restore_checkpoint_manager,
            step=config.resume_from_checkpoint_step,
            load_optimizer=False,
            mesh=mesh,
            model_sharding=model_sharding,
            data_sharding=data_sharding,
        )
    
    return model_train_state

    
def add_batch_dim(tree: ArrayTree):
    return jax.tree_map(lambda x: x[None], tree)

def process_observations(observation, background_images):
    # Images
    image_scale = 1.0/127.5
    image_shift = -1.0

    front_image = tf.image.resize(observation["image_0"], (224, 224)).numpy() * image_scale + image_shift
    wrist_image = tf.image.resize(observation["image_1"], (224, 224)).numpy() * image_scale + image_shift

    tactile_mean = np.array([0.209282, -0.23046867, 0.07245745])
    tactile_std = np.array([6.41063034, 3.83920391, 4.75675555])

    background_left = tf.image.resize(background_images["image_digit_left_background"], (224, 224)).numpy().astype(np.float32)
    digit_left = tf.image.resize(observation["digit_l"], (224, 224)).numpy().astype(np.float32)
    background_right = tf.image.resize(background_images["image_digit_right_background"], (224, 224)).numpy().astype(np.float32)
    digit_right = tf.image.resize(observation["digit_r"], (224, 224)).numpy().astype(np.float32)

    digit_left = (digit_left - background_left - tactile_mean) / tactile_std / 3
    digit_right = (digit_right - background_right - tactile_mean) / tactile_std / 3

    # Mel spectrogram
    mel_spectro_mean = -6.163405
    mel_spectro_std = 14.010228
    MIC_SAMPLE_FREQ = 44100
    MEL_HOP_LENGTH = 104 # selected to make mel_spectrogram have dimension 128x128
    N_MELS = 128
    MEL_HOP_LENGTH = 347
    mic_data = observation['mic']
    spectrogram_nonfft = np.abs(librosa.stft(mic_data, hop_length=MEL_HOP_LENGTH))
    mel_spectro = librosa.feature.melspectrogram(S=spectrogram_nonfft**2, sr=MIC_SAMPLE_FREQ, n_mels=N_MELS)
    mel_spectro = (librosa.power_to_db(mel_spectro) - mel_spectro_mean) / mel_spectro_std

    mel_spectro = np.expand_dims(mel_spectro, axis=-1).repeat(3, axis=-1)
    mel_spectro = tf.image.resize(mel_spectro, (224, 224)).numpy()

    sensors = {
        'image_primary': front_image,
        'image_wrist': wrist_image,
        'image_digit_left': digit_left,
        'image_digit_right': digit_right,
        'mel_spectro': mel_spectro,
        'modality_idx': jnp.array([0]), # only needed for fuse generation loss
    }

    sensor_masks = jax.tree_map(lambda obs: jnp.squeeze(jnp.array([True], dtype=jnp.bool_), axis=-1), sensors)
    sensor_masks['modality_idx'] &= False 
    return sensors, sensor_masks




##############################################################################

def main(_):
    FLAGS = flags.FLAGS
    config = FLAGS.config


    mesh = MeshShardingHelper([-1], ["fsdp"])
    model_sharding = FSDPShardingRule("fsdp", fsdp_axis_size=mesh.mesh.shape["fsdp"])
    data_sharding = PartitionSpec("fsdp")

    model_state = load_pretrained_model(config, mesh, model_sharding, data_sharding)
    dataset_statistics = model_state.dataset_statistics[list(model_state.dataset_statistics.keys())[0]]
    print(dataset_statistics.keys())
    action_mean = dataset_statistics["action"]["mean"]
    action_mean[-1] = 0
    action_std = dataset_statistics["action"]["std"]
    action_std[-1] = 1


    # set up the widowx client
    if FLAGS.initial_eep is not None:
        assert isinstance(FLAGS.initial_eep, list)
        initial_eep = [float(e) for e in FLAGS.initial_eep]
        start_state = np.concatenate([initial_eep, [0, 0, 0, 1]])
    else:
        start_state = None

    env_params = deepcopy(WidowXConfigs.DefaultEnvParams)
    env_params.update(deepcopy(ENV_PARAMS))
    env_params["start_state"] = list(start_state)
    widowx_client = WidowXClient(host=FLAGS.ip, port=FLAGS.port)
    widowx_client.init(env_params, image_size=FLAGS.im_size)

    env = WidowXGym(widowx_client, {
        'image_0': (256, 256),
        'image_1': (128, 128),
        'digit_l': (224, 224),
        'digit_r': (224, 224),
    }, FLAGS.blocking, STICKY_GRIPPER_NUM_STEPS)
    obs, _ = env.reset()

    background_images = {}
    background_images["image_digit_left_background"] = obs["digit_l"]
    background_images["image_digit_right_background"] = obs["digit_r"]
    processed, _ = process_observations(obs, background_images)
    for k, v in processed.items():
        try:
            print(k, v.shape)
        except AttributeError:
            print(k, v)

    
    def sample_actions(
        observations,
        tasks,
    ):
        def _extract_prompt(tokens, mask, mask_ar, gen_start):
            seq_len = tokens.shape[0]
            prompt = jnp.where(
                jnp.arange(seq_len) < gen_start,
                tokens,
                jnp.zeros_like(tokens),
            )
            prompt_mask = jnp.where(
                jnp.arange(seq_len) < gen_start,
                mask,
                jnp.zeros_like(mask),
            )
            prompt_ar = jnp.where(
                jnp.arange(seq_len) < gen_start,
                mask_ar,
                jnp.zeros_like(mask_ar),
            )
        
            return {
                "prompt": prompt,
                "prompt_mask": prompt_mask,
                "prompt_ar": prompt_ar,
            }
        nonlocal background_images
        sensors, sensor_masks = process_observations(observations, background_images)
        text = model_state.tokenizer.tokenize_language_instruction({'task': {"language_instruction": tasks}})
        tokenized = model_state.tokenizer.prepare_tokens_for_generation(
            text,
        )
        
        tokens = tokenized['tokens'].numpy()
        mask_ar = tokenized['mask_ar'].numpy()
        mask_input = tokenized['mask_input'].numpy()
        gen_start = (
            jnp.argmax(tokens == model_state.tokenizer_config.begin_of_action_token, axis=-1) + 1
        )
        prompt_info = _extract_prompt(tokens, mask_input, mask_ar, gen_start)
        prompt_info = add_batch_dim(prompt_info)

        rollout_batch = RolloutBatch(
            sensor_data=add_batch_dim(sensors), 
            sensor_masks=add_batch_dim(sensor_masks),
            prompt=prompt_info["prompt"],
            prompt_mask=prompt_info["prompt_mask"],
            prompt_ar=prompt_info["prompt_ar"],
        )

        unnormalized_action_tokens = model_state.decode(rollout_batch)
        unnormalized_actions = model_state.detokenize_action(unnormalized_action_tokens, obs=rollout_batch.sensor_data)[0]
        return unnormalized_actions * action_std + action_mean


    # goal sampling loop
    goal_instruction = "none"
    while True:
        try: 
            while True:
                print("Current instruction: ", goal_instruction)
                if click.confirm("Take a new instruction?", default=False):
                    goal_instruction = input("Instruction?")

                # reset env
                obs, _ = env.reset()
                background_images["image_digit_left_background"] = obs["digit_l"]
                background_images["image_digit_right_background"] = obs["digit_r"]
                time.sleep(2.0)

                input("Press [Enter] to start.")

                # do rollout
                last_tstep = time.time()
                images = []
                t = 0
                env.widowx_client.start_recording()
                try:
                    while t < FLAGS.num_timesteps:
                        if time.time() > last_tstep + STEP_DURATION:
                            last_tstep = time.time()

                            # save images
                            images.append(obs["image_0"])

                            # get action
                            forward_pass_time = time.time()
                            action = np.array(sample_actions(obs, goal_instruction), dtype=np.float64)
                            print(action)
                            action = action[0]
                            print("forward pass time: ", time.time() - forward_pass_time)

                            # perform environment step
                            start_time = time.time()
                            with DelayedKeyboardInterrupt():
                                obs, _, _, truncated, _ = env.step(action)
                            print("step time: ", time.time() - start_time)

                            t += 1

                            if truncated:
                                break
                except KeyboardInterrupt:
                    print("Keyboard interrupt detected.")
                resp = click.confirm('Save on server?', default=(t >= 5 ))
                env.widowx_client.stop_recording(resp)
                # save video
                if FLAGS.video_save_path is not None:
                    os.makedirs(FLAGS.video_save_path, exist_ok=True)
                    curr_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    save_path = os.path.join(
                        FLAGS.video_save_path,
                        f"{curr_time}.mp4",
                    )
                    video = np.stack(images)
                    imageio.mimsave(save_path, video, fps=1.0 / STEP_DURATION * 3)
        except LostConnection:
            input('Restart server and hit enter:   ')
            
            env_params = deepcopy(WidowXConfigs.DefaultEnvParams)
            env_params.update(deepcopy(ENV_PARAMS))
            env_params["start_state"] = list(start_state)
            widowx_client = WidowXClient(host=FLAGS.ip, port=FLAGS.port)
            print(env_params)
            widowx_client.init(env_params, image_size=FLAGS.im_size)

            env = WidowXGym(widowx_client, {
                'image_0': (256, 256),
                'image_1': (128, 128),
                'digit_l': (224, 224),
                'digit_r': (224, 224),
            }, FLAGS.blocking, STICKY_GRIPPER_NUM_STEPS)
        
if __name__ == "__main__":
    config_flags.DEFINE_config_file(
        "config", "palivla/configs/fuse_config_eval.py", "Path to the config file.", lock_config=False
    )
    app.run(main)


