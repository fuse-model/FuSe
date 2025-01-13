"""
This script shows how we evaluated a finetuned Octo model on a real WidowX robot. While the exact specifics may not
be applicable to your use case, this script serves as a didactic example of how to use Octo in a real-world setting.

If you wish, you may reproduce these results by [reproducing the robot setup](https://rail-berkeley.github.io/bridgedata/)
and installing [the robot controller](https://github.com/rail-berkeley/bridge_data_robot)
"""
from datetime import datetime
from functools import partial
import os
import time

from absl import app, flags, logging
from ml_collections import config_flags
import click
import cv2
import imageio
import jax
import numpy as np
import os 

from octo.model.octo_model import OctoModel
from eval.utils import DelayedKeyboardInterrupt, initialize_widowx_env, sample_actions
from eval.envs.widowx_env import LostConnection

np.set_printoptions(suppress=True)

logging.set_verbosity(logging.WARNING)

flags.DEFINE_string(
    "ip", 
    "128.32.175.252", 
    "IP address of the robot"
)
flags.DEFINE_integer("port", 5556, "Port of the robot")
flags.DEFINE_spaceseplist("goal_eep", [0.3, 0.0, 0.15], "Goal position")
flags.DEFINE_spaceseplist("initial_eep",[0.11796844, -0.01554691,  0.23344009], "Initial position") # neutral 

flags.DEFINE_bool("blocking", False, "Use the blocking controller")

flags.DEFINE_integer("num_timesteps", 100, "num timesteps")
flags.DEFINE_integer("window_size", 2, "Observation history length")
flags.DEFINE_integer("exec_horizon", 1, "Length of action sequence to execute")
flags.DEFINE_string("checkpoint_weights_path", None, "Path to directory containing checkpoints")
flags.DEFINE_integer("checkpoint_step", 50_000, "Step of checkpoint to load")
flags.DEFINE_string("video_dir", None, "Path to directory to save video")
flags.DEFINE_bool("verbose", False, "Print step times")
flags.DEFINE_bool("debug_env", False, "Whether to use a debugging dummy action server or not")

FLAGS = flags.FLAGS

##############################################################################

def main(_):
    model = OctoModel.load_pretrained(
        checkpoint_path=FLAGS.checkpoint_weights_path,
        step=FLAGS.checkpoint_step,
        text_processor_spec=FLAGS.config.text_processor_spec,
    )

    policy_fn = partial(
        sample_actions,
        model,
        rng=jax.random.PRNGKey(0)
    )
    
    env = initialize_widowx_env(FLAGS)
    obs, _ = env.reset()
    command = ""
    pad_mask = np.array([[True for _ in range(FLAGS.window_size)]])[0]
    while True:
        try: 
            if click.confirm("Reset environment?   ", default=False):
                input("Restart the action server, then hit enter to continue...")
                env = initialize_widowx_env(FLAGS)

            if click.confirm(f"Take new command? Currently: {command}", default=(command == "")):
                command = input("Enter command: ")

            task = model.create_tasks(texts=[command])
            images = []
            wrist_images = []
            dig_left_images=  []
            dig_right_images = [] 

            obs, _ = env.reset()

            t = 0
            pred_acs = [] 
            input("Press [Enter] to start.")
            last_tstep = time.time()
            env.start_recording_on_server()
            while t < FLAGS.num_timesteps:
                if time.time() > last_tstep + FLAGS.config.env_params['move_duration']:
                    last_tstep = time.time()
                    step_info = { 
                        'step': t, 
                    }
                    
                    images.append(obs["image_primary"][-1])
                    wrist_images.append(obs["image_wrist"][-1])
                    dig_left_images.append(obs['dig_left'][-1])
                    dig_right_images.append(obs['dig_right'][-1])

                    obs['timestep_pad_mask'] = pad_mask
                    forward_pass_start = time.time()

                    action = np.array(policy_fn(obs, task), dtype=np.float64)
                    pred_acs.append(action[0].copy())
                    step_info['forward_pass_time'] = time.time() - forward_pass_start

                    # perform environment step
                    start_time = time.time()
                    try: 
                        with DelayedKeyboardInterrupt():
                            obs, _, _, truncated, _ = env.step(action)
                    except LostConnection: 
                        break 
                    step_info['step_time'] = time.time() - start_time
                    step_info['total_time'] = time.time() - last_tstep
                    info_str =  ''
                    for key, val in step_info.items():
                        if 'time' in key and not FLAGS.verbose: 
                            continue
                        info_str += f'{key}:      {val}\n'
                    print(info_str, end="" if not FLAGS.verbose else '\n')
                    t += 1
                    if truncated:
                        break
        except KeyboardInterrupt: 
            if click.confirm(
                "Exit? ",
                default=False
            ):
                break

        env.stop_recording_on_server(
            click.confirm(
                "Save high-fidelity video and observations on server (note: may be >100MB, use with caution)?  ",
                default=True
            )
        )


        # save video of received observations on client
        if FLAGS.video_dir:
            if not os.path.exists(FLAGS.video_dir): 
                os.makedirs(FLAGS.video_dir)
            curr_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            video_save_path = os.path.join(FLAGS.video_dir, f'{curr_time}.mp4')

            def format_frame(wrist_im, im, dig_left_im, dig_right_im):
                wrist_im = cv2.resize(wrist_im, (256, 256))
                dig_left_im = cv2.resize(dig_left_im, (256, 256))
                dig_right_im = cv2.resize(dig_right_im, (256, 256))
                im = cv2.resize(im, (256, 256))

                # create 2x2 grid
                return np.concatenate([np.concatenate([wrist_im, im], axis=1), np.concatenate([dig_left_im, dig_right_im], axis=1)], axis=0)

            video = []
            for i, (im, im_w, im_l, im_r) in enumerate(zip(images, wrist_images, dig_left_images, dig_right_images)): 
                frame = format_frame(im_w, im, im_l, im_r)
                video.append(frame)
            
            video = np.stack(video)
            imageio.mimsave(video_save_path, video, fps=1.0 / FLAGS.config.env_params['move_duration'] * 1.25)
                            
                

if __name__ == "__main__":
    default_config_file = os.path.join(
        os.path.dirname(__file__), "eval_config.py"
    )
    config_flags.DEFINE_config_file(
        "config",
        default_config_file,
        "File path to environment setup and experiment parameters",
        lock_config=False,
    )
    app.run(main)

