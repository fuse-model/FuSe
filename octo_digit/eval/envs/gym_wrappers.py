from collections import deque
from concurrent.futures import process
import logging
from typing import Optional, Sequence, Tuple, Union

import gym
import gym.spaces
import jax
import numpy as np
import tensorflow as tf

import librosa
import time 
import os 
from eval.eval_config import VERBOSE
from multimodalMAE.vit import ViT
from multimodalMAE.mae import MAE

def stack_and_pad(history: deque, num_obs: int):
    """
    Converts a list of observation dictionaries (`history`) into a single observation dictionary
    by stacking the values. Adds a padding mask to the observation that denotes which timesteps
    represent padding based on the number of observations seen so far (`num_obs`).
    """
    horizon = len(history)
    full_obs = {k: np.stack([dic[k] for dic in history]) for k in history[0]}
    pad_length = horizon - min(num_obs, horizon)
    timestep_pad_mask = np.ones(horizon)
    timestep_pad_mask[:pad_length] = 0
    full_obs["timestep_pad_mask"] = timestep_pad_mask
    return full_obs


def space_stack(space: gym.Space, repeat: int):
    """
    Creates new Gym space that represents the original observation/action space
    repeated `repeat` times.
    """

    if isinstance(space, gym.spaces.Box):
        return gym.spaces.Box(
            low=np.repeat(space.low[None], repeat, axis=0),
            high=np.repeat(space.high[None], repeat, axis=0),
            dtype=space.dtype,
        )
    elif isinstance(space, gym.spaces.Discrete):
        return gym.spaces.MultiDiscrete([space.n] * repeat)
    elif isinstance(space, gym.spaces.Dict):
        return gym.spaces.Dict(
            {k: space_stack(v, repeat) for k, v in space.spaces.items()}
        )
    else:
        raise ValueError(f"Space {space} is not supported by Octo Gym wrappers.")


def listdict2dictlist(LD):
    return {k: [dic[k] for dic in LD] for k in LD[0]}


def add_octo_env_wrappers(
    env: gym.Env,
    config: dict,
    **kwargs,
):
    """Adds env wrappers for action normalization, multi-action
    future prediction, image resizing, and history stacking.

    Uses defaults from model config, but all can be overridden through kwargs.

    Arguments:
        env: gym Env
        config: PretrainedModel.config
        dataset_statistics: from PretrainedModel.load_dataset_statistics
        # Additional (optional) kwargs
        exec_horizon: int for RHCWrapper
        resize_size: None or tuple or list of tuples for ResizeImageWrapper
        horizon: int for HistoryWrapper
    """
    exec_horizon = kwargs.get(
        "exec_horizon", config["model"]["heads"]["action"]["kwargs"]["pred_horizon"]
    )

    logging.info("Running receding horizon control with exec_horizon: ", exec_horizon)
    env = RHCWrapper(env, exec_horizon)
    resize_size = kwargs.get(
        "resize_size",
        config["dataset_kwargs"]["frame_transform_kwargs"]["resize_size"],
    )

    logging.info("Resizing images w/ parameters", resize_size)
    env = ResizeImageWrapper(env, resize_size)

    horizon = kwargs.get("horizon", config["window_size"])
    logging.info("Adding history of size: ", horizon)
    env = HistoryWrapper(env, horizon)

    logging.info("New observation space: ", env.observation_space)
    return env


try: 
    from tvl_enc import tacvis
    ON_TPUS = False 
    
except ModuleNotFoundError: 
    ON_TPUS = True 
if not ON_TPUS: 
    from tvl_embedder import TVLEmbedder 
    import torch 

class ObsProcessingWrapper(gym.ObservationWrapper): 
    def __init__(self, 
        env: gym.Env, 
        remap_keys: dict, 
        new_fields: Sequence[str], 
        flip_channels: bool = False, 
        do_background_subtraction: bool = True,
        tvl_embedder: TVLEmbedder = None,
    ): 
        super().__init__(env)
        def flatten_dict(dic, separator="_"):
            flat = {}
            for key, val in dic.items(): 
                if isinstance(val, dict): 
                    for sub_key, sub_val in val.items(): 
                        full_key = separator.join([key, sub_key])
                        flat[full_key] = sub_val
                else: 
                    flat[key] = val 
            return flat 
        self._remap_keys = flatten_dict(remap_keys) 
        self._new_fields = new_fields
       
        # if "digit_embeddings" in self._new_fields: 
        #     self._setup_tvl_encoder()
        self._tvl_encoder = tvl_embedder

        if "siglip" in self._new_fields: 
            self._setup_siglip()

        self._do_background_subtraction = do_background_subtraction
        self._flip_channels = flip_channels
        self.dev = 0
        embed_mae = False 
        for field in new_fields:
            if field.startswith('mae'):
                embed_mae = True
                break 
        if embed_mae: 
            self._setup_mae()
    
    def _setup_mae(self):
        v = ViT(
            image_size = 256,
            tactile_size = 128,
            image_patch_size = 32,
            tactile_patch_size = 16,
            num_classes = 1000,
            dim = 1024,
            depth = 6,
            heads = 8,
            mlp_dim = 2048,
            pool = 'mean',
            image_channels=6,
            tactile_channels=6
        )
        self.mae_uniform = MAE(
            encoder = v,
            masking_image = 0.85,   # the paper recommended 75% masked patches
            masking_tactile = 0.85, # the paper recommended 75% masked patches
            decoder_dim = 512,      # paper showed good results with just 512
            decoder_depth = 6,       # anywhere from 1 to 8
            masking_type='uniform'       # uniform or asymmetric or modality
        )
        self.mae_uniform.load_state_dict(torch.load(f'/home/josh/multiMAE/multimodalMAE/uniform_95.pth'))
        self.mae_uniform.cuda(self.dev)
        
        # self.mae_asym = MAE(
        #     encoder = v,
        #     masking_image = 0.85,   # the paper recommended 75% masked patches
        #     masking_tactile = 0.85, # the paper recommended 75% masked patches
        #     decoder_dim = 512,      # paper showed good results with just 512
        #     decoder_depth = 6,       # anywhere from 1 to 8
        #     masking_type='asymmetric'       # uniform or asymmetric or modality
        # )
        # self.mae_asym.load_state_dict(torch.load(f'/home/josh/multiMAE/multimodalMAE/asymmetric_25.pth'))
        # self.mae_asym.cuda(self.dev)
        
        
        
    

    def _setup_tvl_encoder(self, tvl_device='cuda:0'): 
        # if ON_TPUS: 
        #     self._tvl_encoder = None 
        # else: 
        #     self._tvl_encoder = TVLEmbedder(tvl_device)
        raise NotImplementedError

        
    def _setup_siglip(self): 
        raise NotImplementedError

    def _remap_dict(self, old_dic): 
        key_mappings = self._remap_keys 
        new_dic = {} 
        for new_key, old_key in key_mappings.items(): 
            if old_key in old_dic: 
                new_dic[new_key] = old_dic[old_key]
        return new_dic 

    def observation(self, observation): 
        start = time.time() 
        if 'xyz' in observation:
            xyz = observation['xyz']
        else:
            xyz = 0
        processed_obs = self._remap_dict(observation)
        if self._flip_channels: 
            for key, val in processed_obs.items(): 
                if key.startswith('image'): 
                    processed_obs[key] = val[..., ::-1]

        if self._do_background_subtraction and "image_digit_left" in processed_obs: 
            dig_l, dig_r = processed_obs['image_digit_left'], processed_obs['image_digit_right']
            processed_obs['dig_left'] = dig_l
            processed_obs['dig_right'] = dig_r
            back_l, back_r = processed_obs['image_digit_left_background'], processed_obs['image_digit_right_background']
            processed_obs["image_digit_left"] = np.array(dig_l, dtype=np.int16) - np.array(back_l, dtype=np.int16)
            processed_obs["image_digit_right"] = np.array(dig_r, dtype=np.int16) - np.array(back_r, dtype=np.int16)
    
        for new_field in self._new_fields: 
            if new_field == "spectro": # assume mel-spectro 
                MIC_SAMPLE_FREQ = 44100
                MEL_HOP_LENGTH = 104 # selected to make mel_spectrogram have dimension 128x128
                N_MELS = 128
                MEL_HOP_LENGTH = 347
                start_mic = time.time() 
                mic_data = observation['mic']
                spectrogram_nonfft = np.abs(librosa.stft(mic_data, hop_length=MEL_HOP_LENGTH))
                mel_spectro = librosa.feature.melspectrogram(S=spectrogram_nonfft**2, sr=MIC_SAMPLE_FREQ, n_mels=N_MELS)
                mel_spectro = librosa.power_to_db(mel_spectro)
                
                # processed_obs['spectro'] = observation['mel_spectro']
                processed_obs['mel_spectro'] = mel_spectro
                end_mic = time.time() 
                if VERBOSE: 
                    print('DELTA MIC    ', end_mic - start_mic)

            elif new_field == "digit_embeddings": 
                raise NotImplementedError
                tvl_obs_dict = {
                    "digit_left": processed_obs["image_digit_left"], 
                    "background_l": observation["background_l"], 
                    "digit_right": processed_obs["image_digit_right"], 
                    "background_r": observation["background_r"], 
                }
                # shape: 2 x (embdim = 768)
                digit_embeddings = self._tvl_encoder.get_embeddings(tvl_obs_dict)['digit_embeddings'].cpu().detach().numpy() 

                digit_0_embedding, digit_1_embedding = digit_embeddings
                processed_obs["digit_left_embedding"] = digit_0_embedding
                processed_obs["digit_right_embedding"] = digit_1_embedding
            
            # elif new_field.startswith('mae'):
            #     def to_torch_style(image_0, image_1):
            #         combined = torch.cat([image_0, image_1], dim=-1)
            #         combined = combined.permute(0, 3, 1, 2)
            #         combined = combined.float() / 255.0
            #         combined = combined.cuda(self.dev)
            #         return combined
                
            #     tac_left, tac_right = processed_obs['image_digit_left'], processed_obs['image_digit_right']
            #     tac_left = torch.from_numpy(tac_left[None, ...])
            #     tac_right = torch.from_numpy(tac_right[None, ...])
            #     tac = to_torch_style(tac_left, tac_right)
            #     obs = {'tactile': tac}
            #     tac_suffix = '_tac' if 'tac' in new_field else ''
            #     if not tac_suffix:
            #         primary = torch.from_numpy(processed_obs['image_primary'][None, ...])
            #         wrist = torch.from_numpy(processed_obs['image_wrist'][None, ...])
            #         image = to_torch_style(primary, wrist)
            #         obs['image'] = image
            
            #     with torch.no_grad():
            #         encoder = self.mae_asym
            #         embedding, _ = encoder.encode(obs, test=True)
            #         embedding = embedding.mean(dim=1)
            #         embedding = embedding[0].detach().cpu().numpy()
            #         processed_obs[f'asym{tac_suffix}'] = embedding
                    
            #         encoder = self.mae_uniform
            #         embedding, _ = encoder.encode(obs, test=True)
            #         embedding = embedding.mean(dim=1)
            #         embedding = embedding[0].detach().cpu().numpy()
            #         processed_obs[f'uniform{tac_suffix}'] = embedding    
            elif new_field == 'mae_tac_uniform':
                def to_torch_style(image_0, image_1):
                    combined = torch.cat([image_0, image_1], dim=-1)
                    combined = combined.permute(0, 3, 1, 2)
                    combined = combined.float() / 255.0
                    combined = combined.cuda(self.dev)
                    return combined
                
                tac_left, tac_right = processed_obs['image_digit_left'], processed_obs['image_digit_right']
                
                tac_left = torch.from_numpy(tac_left[None, ...])
                tac_right = torch.from_numpy(tac_right[None, ...])
                tac = to_torch_style(tac_left, tac_right)
                obs = {'tactile': tac}
        
            
                with torch.no_grad():
                    encoder = self.mae_uniform
                    embedding, _ = encoder.encode(obs, test=True)
                    embedding = embedding.mean(dim=1)
                    embedding = embedding[0].detach().cpu().numpy()
                    processed_obs[f'uniform_tac'] = embedding    
                for key in ['image_digit_left', 'image_digit_right', 'image_digit_left_background', 'image_digit_right_background']:
                    processed_obs.pop(key)

            elif new_field == "siglip": 
                raise NotImplementedError

            else: 
                raise ValueError('Unsupported new field')

        


        # TEMP
        processed_obs["imu"] = np.zeros((15, 3)).flatten()
        processed_obs['xyz'] = xyz
        end  = time.time() 
        if VERBOSE: 
            print('OBS_PROCESSING_WRAPPER',  end-start)
        return processed_obs


class HistoryWrapper(gym.Wrapper):
    """
    Accumulates the observation history into `horizon` size chunks. If the length of the history
    is less than the length of the horizon, we pad the history to the full horizon length.
    A `timestep_pad_mask` key is added to the final observation dictionary that denotes which timesteps
    are padding.
    """

    def __init__(self, env: gym.Env, horizon: int):
        super().__init__(env)
        self.horizon = horizon

        self.history = deque(maxlen=self.horizon)
        self.num_obs = 0

        self.observation_space = space_stack(self.env.observation_space, self.horizon)

    def step(self, action):
        obs, reward, done, trunc, info = self.env.step(action)
        self.num_obs += 1
        self.history.append(obs)
        assert len(self.history) == self.horizon
        full_obs = stack_and_pad(self.history, self.num_obs)

        return full_obs, reward, done, trunc, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.num_obs = 1
        self.history.extend([obs] * self.horizon)
        full_obs = stack_and_pad(self.history, self.num_obs)

        return full_obs, info


class RHCWrapper(gym.Wrapper):
    """
    Performs receding horizon control. The policy returns `pred_horizon` actions and
    we execute `exec_horizon` of them.
    """

    def __init__(self, env: gym.Env, exec_horizon: int):
        super().__init__(env)
        self.exec_horizon = exec_horizon

    def step(self, actions):
        if self.exec_horizon == 1 and len(actions.shape) == 1:
            actions = actions[None]
        assert len(actions) >= self.exec_horizon
        rewards = []
        observations = []
        infos = []

        for i in range(self.exec_horizon):
            obs, reward, done, trunc, info = self.env.step(actions[i])
            observations.append(obs)
            rewards.append(reward)
            infos.append(info)

            if done or trunc:
                break

        infos = listdict2dictlist(infos)
        infos["rewards"] = rewards
        infos["observations"] = observations

        return obs, np.sum(rewards), done, trunc, infos


class TemporalEnsembleWrapper(gym.Wrapper):
    """
    Performs temporal ensembling from https://arxiv.org/abs/2304.13705
    At every timestep we execute an exponential weighted average of the last
    `pred_horizon` predictions for that timestep.
    """

    def __init__(self, env: gym.Env, pred_horizon: int, exp_weight: int = 0):
        super().__init__(env)
        self.pred_horizon = pred_horizon
        self.exp_weight = exp_weight

        self.act_history = deque(maxlen=self.pred_horizon)

        self.action_space = space_stack(self.env.action_space, self.pred_horizon)

    def step(self, actions):
        assert len(actions) >= self.pred_horizon

        self.act_history.append(actions[: self.pred_horizon])
        num_actions = len(self.act_history)

        # select the predicted action for the current step from the history of action chunk predictions
        curr_act_preds = np.stack(
            [
                pred_actions[i]
                for (i, pred_actions) in zip(
                    range(num_actions - 1, -1, -1), self.act_history
                )
            ]
        )

        # more recent predictions get exponentially *less* weight than older predictions
        weights = np.exp(-self.exp_weight * np.arange(num_actions))
        weights = weights / weights.sum()
        # compute the weighted average across all predictions for this timestep
        action = np.sum(weights[:, None] * curr_act_preds, axis=0)

        return self.env.step(action)

    def reset(self, **kwargs):
        self.act_history = deque(maxlen=self.pred_horizon)
        return self.env.reset(**kwargs)


class ResizeImageWrapperDict(gym.ObservationWrapper):
    def __init__(
        self,
        env: gym.Env,
        resize_size: dict,
    ):
        super().__init__(env)
        self.keys_to_resize = resize_size # name to size 
    
    def observation(self, observation):
        # print("In resizer")
        # print(self.keys_to_resize)
        # print(observation.keys())
        # for k, v in observation.items(): 
        #     try: 
        #         print(k, v.shape)
        #     except AttributeError: 
        #         pass 
        for k, size in self.keys_to_resize.items():
            image = tf.image.resize(
                observation[k], size=size, method="lanczos3", antialias=True
            )
            image = tf.cast(tf.clip_by_value(tf.round(image), 0, 255), tf.uint8).numpy()
            observation[k] = image
        # for k, v in observation.items(): 
        #     try: 
        #         print(k, v.shape)
        #     except AttributeError: 
        #         pass 
        return observation

class ResizeImageWrapper(gym.ObservationWrapper):
    def __init__(
        self,
        env: gym.Env,
        resize_size: Optional[Union[Tuple, Sequence[Tuple]]],
    ):
        super().__init__(env)
        assert isinstance(
            self.observation_space, gym.spaces.Dict
        ), "Only Dict observation spaces are supported."
        spaces = self.observation_space.spaces

        if resize_size is None:
            self.keys_to_resize = {}
        elif isinstance(self.resize_size, tuple):
            self.keys_to_resize = {k: resize_size for k in spaces if "image_" in k}
        else:
            self.keys_to_resize = {
                f"image_{i}": resize_size[i] for i in range(len(resize_size))
            }
        logging.info(f"Resizing images: {self.keys_to_resize}")
        for k, size in self.keys_to_resize.items():
            spaces[k] = gym.spaces.Box(
                low=0,
                high=255,
                shape=size + (3,),
                dtype=np.uint8,
            )
        self.observation_space = gym.spaces.Dict(spaces)

    def observation(self, observation):
        for k, size in self.keys_to_resize.items():
            image = tf.image.resize(
                observation[k], size=size, method="lanczos3", antialias=True
            )
            image = tf.cast(tf.clip_by_value(tf.round(image), 0, 255), tf.uint8).numpy()
            observation[k] = image
        return observation


class UnnormalizeActionProprio(gym.ActionWrapper, gym.ObservationWrapper):
    """
    Un-normalizes the action and proprio.
    """

    def __init__(
        self,
        env: gym.Env,
        action_proprio_metadata: dict,
    ):
        self.action_proprio_metadata = jax.tree_map(
            lambda x: np.array(x),
            action_proprio_metadata,
            is_leaf=lambda x: isinstance(x, list),
        )
        super().__init__(env)

    def unnormalize(self, data, metadata):
        mask = metadata.get("mask", np.ones_like(metadata["mean"], dtype=bool))
        return np.where(
            mask,
            (data * metadata["std"]) + metadata["mean"],
            data,
        )

    def normalize(self, data, metadata):
        mask = metadata.get("mask", np.ones_like(metadata["mean"], dtype=bool))
        return np.where(
            mask,
            (data - metadata["mean"]) / (metadata["std"] + 1e-8),
            data,
        )

    def action(self, action):
        return self.unnormalize(action, self.action_proprio_metadata["action"])

    def observation(self, obs):
        if "proprio" in self.action_proprio_metadata:
            obs["proprio"] = self.normalize(
                obs["proprio"], self.action_proprio_metadata["proprio"]
            )
        else:
            assert "proprio" not in obs, "Cannot normalize proprio without metadata."
        return obs
