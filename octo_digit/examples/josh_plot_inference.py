import cv2
import jax
import tensorflow as tf
import tensorflow_datasets as tfds
import tqdm
import rlds
import mediapy as media
from PIL import Image
from IPython import display
import numpy as np
import imageio
import os
from octo.model.octo_model import OctoModel
import matplotlib.pyplot as plt


model_to_dataset = { 
    'finetune_vizonly_diffusion_val_20240517_021703': '9.9.0', # '8.8.0', 
    'finetune_vizonly_mse_val_20240517_022456': '9.9.0', # '8.8.0', 
    # 'finetune_vizonly_diffusion_val_small_20240517_022228': '9.9.0',
    # 'finetune_vizonly_mse_val_small_20240517_022019': '9.9.0'
}

splits = ['train[:1]', 'val[:1]']
model_prefix = f'/home/joshwajones/tpu_octo_ckpts/'
checkpoint_step = 50000
dataset_prefix = '/home/joshwajones/tensorflow_datasets/digit_dataset/'
verbose = True 
task_modes = ['goals', 'language', 'multimodal'] 

remap_keys = { 
    'image_0': 'image_primary', 
    'image_1': 'image_wrist'
}

resize_keys = { 
    'image_primary': (256, 256), 
    'image_wrist': (128, 128),
}

def convert_obs(obs): 
    new_obs = {} 
    for key, val in obs.items(): 
        new_key = remap_keys.get(key, key)
        if new_key in resize_keys: 
            val = cv2.resize(np.array(val), resize_keys[new_key])
        new_obs[new_key] = val 
    # new_obs = jax.tree_map(lambda x: x[None], new_obs)
    return new_obs 

def recursive_dict_stack(dicts): 
    stacked_dict = {} 
    for key in dicts[0]: 
        vals = [d[key] for d in dicts] 
        if isinstance(vals[0], dict): 
            stacked_dict[key] = recursive_dict_stack(vals)
        else: 
            try: 
                stacked_dict[key] = np.stack(vals)[None]
            except: 
                print(key)
    return stacked_dict

for model_name, dataset_version in model_to_dataset.items(): 
    checkpoint_weights_path = model_prefix + model_name
    model = OctoModel.load_pretrained(
        checkpoint_weights_path,
        checkpoint_step,
    )
    builder = tfds.builder_from_directory(builder_dir=f'/home/joshwajones/tensorflow_datasets/digit_dataset/{dataset_version}')
    for split in splits: 
        for task_mode in task_modes: 
            split_name = split[:split.index('[')]
            save_name = f'{task_mode}/{split_name}_{model_name}'
            ds = builder.as_dataset(split=split) 
            episode = next(iter(ds))
            steps = list(episode['steps'])
            observations = [convert_obs(step['observation']) for step in steps]

            primary_images = [obs['image_primary'] for obs in observations]
            wrist_images = [obs['image_wrist'] for obs in observations]
            wrist_imgs_save = [cv2.resize(np.array(img), (256, 256)) for img in wrist_images]

            # goal_obs = observations[-1]
            goal_obs = { 
                'image_primary': primary_images[-1], 
                'image_wrist': wrist_images[-1],
            }
            goal_obs = jax.tree_map(lambda x: x[None], goal_obs)
            language_instruction = steps[0]['language_instruction'].numpy().decode()

            # visualize episode
            if verbose: 
                print(f'Instruction: {language_instruction}')
                
                video = np.concatenate([np.stack(wrist_imgs_save), np.stack(primary_images)], axis=1)
                imageio.mimsave(f'./pred_v_true_vids/{save_name}.mp4', video, fps=5.0)
            
            WINDOW_SIZE = 2

            # create `task` dict
            # task = model.create_tasks(goals={"image_primary": goal_image[None]})   # for goal-conditioned
            task_kwargs = {}
            if "language" in task_mode or "multimodal" in task_mode: 
                task_kwargs["texts"] = [language_instruction]
            if "goals" in task_mode or "multimodal" in task_mode: 
                task_kwargs['goals'] = goal_obs 
            task = model.create_tasks(**task_kwargs)     

            pred_actions, true_actions = [], []
            for step in tqdm.tqdm(range(0, len(primary_images) - WINDOW_SIZE + 1)):
                # observation = recursive_dict_stack(observations[step : step + WINDOW_SIZE])
                # observation['timestep_pad_mask'] = np.array([[True, True]])
                input_images = np.stack(primary_images[step : step + WINDOW_SIZE])[None]
                input_wrist = np.stack(wrist_images[step : step + WINDOW_SIZE])[None]
                observation = {
                    'image_primary': input_images,
                    'image_wrist': input_wrist, 
                    'timestep_pad_mask': np.array([[True, True]]),
                }

                # this returns *normalized* actions --> we need to unnormalize using the dataset statistics
                actions = model.sample_actions(observation, task, rng=jax.random.PRNGKey(0), unnormalization_statistics=model.dataset_statistics["action"])[0]
                # norm_actions = norm_actions[0]   # remove batch
                # actions = (
                #     norm_actions * model.dataset_statistics["bridge_dataset"]['action']['std']
                #     + model.dataset_statistics["bridge_dataset"]['action']['mean']
                # )

                pred_actions.append(actions)
                true_actions.append(
                    steps[step+1]['action']
                )
            
            ACTION_DIM_LABELS = ['x', 'y', 'z', 'yaw', 'pitch', 'roll', 'grasp']

            # build image strip to show above actions

            img_strip = np.concatenate(np.array(video[::3]), axis=1)

            # set up plt figure
            figure_layout = [
                ['image'] * len(ACTION_DIM_LABELS),
                ACTION_DIM_LABELS
            ]
            plt.rcParams.update({'font.size': 12})
            fig, axs = plt.subplot_mosaic(figure_layout)
            fig.set_size_inches([45, 10])

            # plot actions
            pred_actions = np.array(pred_actions).squeeze()
            true_actions = np.array(true_actions).squeeze()
            for action_dim, action_label in enumerate(ACTION_DIM_LABELS):
                # actions have batch, horizon, dim, in this example we just take the first action for simplicity
                axs[action_label].plot(pred_actions[:, 0, action_dim], label='predicted action')
                axs[action_label].plot(true_actions[:, action_dim], label='ground truth')
                axs[action_label].set_title(action_label)
                axs[action_label].set_xlabel('Time in one episode')

            axs['image'].imshow(img_strip)
            axs['image'].set_xlabel('Time in one episode (subsampled)')
            plt.legend()
            fig.savefig(f'./pred_v_true/{save_name}.png')
