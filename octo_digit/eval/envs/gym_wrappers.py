from collections import deque
import logging
from typing import Optional, Sequence, Tuple, Union

import gym
import gym.spaces
import jax
import librosa
import numpy as np
import tensorflow as tf


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


def create_mel_spectro(mic_data, mic_sample_freq=44100, n_mels=128, mel_hop_length=347):
    spectrogram_nonfft = np.abs(
        librosa.stft(mic_data, hop_length=mel_hop_length)
    )
    mel_spectro = librosa.feature.melspectrogram(
        S=spectrogram_nonfft**2, sr=mic_sample_freq, n_mels=n_mels
    )
    mel_spectro = librosa.power_to_db(mel_spectro)
    return mel_spectro


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


class ObsProcessingWrapper(gym.ObservationWrapper):
    def __init__(
        self,
        env: gym.Env,
        remap_keys: dict,
        new_fields: Sequence[str],
        do_background_subtraction: bool = True,
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

        self._do_background_subtraction = do_background_subtraction
        self.dev = 0

    def _remap_dict(self, old_dic):
        key_mappings = self._remap_keys
        new_dic = {}
        for new_key, old_key in key_mappings.items():
            if old_key in old_dic:
                new_dic[new_key] = old_dic[old_key]
        return new_dic

    def observation(self, observation):
        if "xyz" in observation:
            xyz = observation["xyz"]
        else:
            xyz = 0
        processed_obs = self._remap_dict(observation)
        if self._do_background_subtraction and "image_digit_left" in processed_obs:
            dig_l, dig_r = (
                processed_obs["image_digit_left"],
                processed_obs["image_digit_right"],
            )
            processed_obs["dig_left"] = dig_l
            processed_obs["dig_right"] = dig_r
            back_l, back_r = (
                processed_obs["image_digit_left_background"],
                processed_obs["image_digit_right_background"],
            )
            processed_obs["image_digit_left"] = np.array(
                dig_l, dtype=np.int16
            ) - np.array(back_l, dtype=np.int16)
            processed_obs["image_digit_right"] = np.array(
                dig_r, dtype=np.int16
            ) - np.array(back_r, dtype=np.int16)

        for new_field in self._new_fields:
            if new_field == "spectro":  # assume mel-spectro
                processed_obs["mel_spectro"] = create_mel_spectro(observation["mic"])
            else:
                raise ValueError("Unsupported new field")

        processed_obs["xyz"] = xyz
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
        self.keys_to_resize = resize_size  # name to size

    def observation(self, observation):
        for k, size in self.keys_to_resize.items():
            image = tf.image.resize(
                observation[k], size=size, method="lanczos3", antialias=True
            )
            image = tf.cast(tf.clip_by_value(tf.round(image), 0, 255), tf.uint8).numpy()
            observation[k] = image
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
