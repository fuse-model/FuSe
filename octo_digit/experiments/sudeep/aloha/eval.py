import os
import sys
import time
import flax
import yaml
from collections import deque

from absl import app, flags, logging

import numpy as np
import tensorflow as tf
import cv2
import jax
import jax.numpy as jnp
from functools import partial

from octo.model.octo_model import OctoModel
from scipy.spatial.transform import Rotation as R


sys.path.append("/home/huzheyuan/Desktop/language-dagger/src")
sys.path.append("/home/huzheyuan/Desktop/language-dagger/src/aloha_pro/aloha_scripts/")
from aloha_pro.aloha_scripts.real_env import make_real_env

tf.config.set_visible_devices([], "GPU")

np.set_printoptions(suppress=True)

logging.set_verbosity(logging.INFO)

FLAGS = flags.FLAGS
flags.DEFINE_bool("deterministic", True, "Whether to sample action deterministically")
flags.DEFINE_float("temperature", 1e-7, "Temperature for sampling actions")
flags.DEFINE_string("checkpoint", None, "Path to checkpoint")


def stack_and_pad_obs(fn, horizon):
    """
    This turns a function that takes a fixed length observation history into a function that
    takes just the current observation (or sequence of observations since the last policy call).
    The full observation history is saved inside this function. This function handles stacking
    the list of observation dictionaries to form a dictionary of arrays. This function also pads
    the observation history to the full horizon length. A `pad_mask` key is added to the final
    observation dictionary that denotes which timesteps are padding.
    """

    full_history = []

    def stack_obs(obs):
        dict_list = {k: [dic[k] for dic in obs] for k in obs[0]}
        return jax.tree_map(
            lambda x: np.stack(x), dict_list, is_leaf=lambda x: type(x) == list
        )

    def wrapped_fn(obs, *args, **kwargs):
        nonlocal full_history
        if isinstance(obs, list):
            full_history.extend(obs)
        else:
            full_history.append(obs)
        history = full_history[-horizon:]
        pad_length = horizon - len(history)
        pad_mask = np.ones(horizon)
        pad_mask[:pad_length] = 0
        history = [history[0]] * pad_length + history
        full_obs = stack_obs(history)
        full_obs["timestep_pad_mask"] = pad_mask
        # full_obs['proprio'][-1] = 0 # TODO fix this up also
        return fn(full_obs, *args, **kwargs)

    return wrapped_fn


def supply_rng(f, rng=jax.random.PRNGKey(0)):
    def wrapped(*args, **kwargs):
        nonlocal rng
        rng, key = jax.random.split(rng)
        return f(*args, rng=key, **kwargs)

    return wrapped


@partial(jax.jit, static_argnames="argmax")
def sample_actions(
    pretrained_model: OctoModel,
    observations,
    tasks,
    rng,
    argmax=False,
    temperature=1.0,
):
    # add batch dim to observations
    observations = jax.tree_map(lambda x: x[None], observations)
    # tasks = jax.tree_map(lambda x: x[None], tasks)
    logging.warning(
        "observations: %s", flax.core.pretty_repr(jax.tree_map(jnp.shape, observations))
    )
    logging.warning("tasks: %s", flax.core.pretty_repr(jax.tree_map(jnp.shape, tasks)))
    actions = pretrained_model.sample_actions(
        observations,
        tasks,
        rng=rng,
        argmax=argmax,
        temperature=temperature,
    )

    # unbatch the actions and return
    return actions[0]


def action_denormalizer(action, scale, loc, mask):
    action = action.copy()
    action[mask] *= scale[mask]
    action[mask] += loc[mask]
    return action


def index_and_resize(obs_dict, height, width, obs_key=None, zero_image=False, bgr_input=True):
    if obs_key is not None:
        assert zero_image == False
        assert obs_key in obs_dict, f"only keys are {list(obs_dict.keys())}"

        img = obs_dict[obs_key][:,:,:3][:,:,::-1].copy() if bgr_input \
              else obs_dict[obs_key][:,:,:3].copy()
        cv2.imwrite(f'test_{obs_key}.jpg', cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)[:,:,::-1])
        return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

    assert zero_image, "outputs zero_image if no obs_key!"
    return np.zeros((height, width, 3), dtype=np.uint8)


def load_checkpoint(weights_path):
    weights_path = weights_path.rstrip('/')
    checkpoint_path = os.path.dirname(os.path.expanduser(weights_path))
    step = int(weights_path.split('/')[-1])
    model = OctoModel.load_pretrained(checkpoint_path, step=step)

    with open(os.path.join(checkpoint_path, 'exp_hparams.yaml'), 'r') as f:
        experiment_hparams = yaml.safe_load(f)

    metadata_key = experiment_hparams.get('metadata_key', None)
    action_proprio_metadata = model.dataset_statistics if metadata_key is None \
                              else model.dataset_statistics[metadata_key]
    action_metadata = action_proprio_metadata['action']

    if experiment_hparams['normalization'] == 'normal':
        action_loc = np.array(action_metadata["mean"])
        action_scale = np.array(action_metadata["std"])
    elif experiment_hparams['normalization'] == 'bounds':
        ac_min = np.array(action_metadata["p01"])
        ac_max = np.array(action_metadata["p99"])
        action_loc = 0.5 * (ac_max + ac_min)
        action_scale = 0.5 * (ac_max - ac_min)
    else:
        raise ValueError

    action_mask = np.array(action_metadata.get('mask', np.ones_like(action_scale, dtype=np.bool_)))
    action_denorm_fn = partial(action_denormalizer, scale=action_scale, loc=action_loc, mask=action_mask)

    logging.info("############# USING FOLLOWING NORM STATS ###############")
    logging.info(f"Using {experiment_hparams['normalization']} norm!")
    logging.info(f"action loc: {action_loc}")
    logging.info(f"action scale: {action_scale}")
    logging.info(f"action mask: {action_mask}")

    policy_fn = stack_and_pad_obs(
        supply_rng(
            partial(
                sample_actions,
                model,
                argmax=FLAGS.deterministic,
                temperature=FLAGS.temperature,
            ),
        ),
        horizon=1, #model.config["dataset_kwargs"]["traj_transform_kwargs"]["window_size"],
    )
    return policy_fn, action_denorm_fn, experiment_hparams, model


class OctoPolicy:
    def __init__(self, policy_fn, action_denorm_fn, exp_hparams, model):
        self.policy_fn = policy_fn
        self.action_denorm_fn = action_denorm_fn
        self.img_mapping = exp_hparams['img_mapping']
        self.model = model

        self.goal = self.model.create_tasks(texts=["uncap the pen"])
        self._last_time = None
        self.gamma = float(exp_hparams['gamma'])
        self.period = 1.0 / float(exp_hparams['hz'])
        self.last_ac = None
        self.T = int(exp_hparams['control_horizon'])
        self._action_history = deque(maxlen=self.T)

    def _convert_obs(self, observation):
        proprio = observation['qpos']
        obs_dict = {
            k: index_and_resize(observation["images"], bgr_input=True, **img_hparams) for k, img_hparams in self.img_mapping.items()
        }
        obs_dict['proprio'] = proprio.astype(np.float32)
        return obs_dict

    def forward(self, observation):
        if not len(self._action_history):
            obs_hist = [self._convert_obs(observation)]
            actions = np.array(self.policy_fn(obs_hist, self.goal))[0]
            [self._action_history.append(ac) for ac in actions[:self.T]]

        raw_ac = self.action_denorm_fn(self._action_history.popleft())
        last_ac = self.last_ac if self.last_ac is not None else raw_ac
        self.last_ac = self.gamma * raw_ac + (1 - self.gamma) * last_ac

        if self._last_time is not None:
            delta = time.time() - self._last_time
            if delta < self.period:
                time.sleep(self.period - delta)

            hz = 1.0 / (time.time() - self._last_time)
            logging.info(f"Effective HZ: {hz}")
        self._last_time = time.time()

        return self.last_ac

    def load_goal_imgs(self, goal_img_dict):
        logging.warning('settting image goal')
        goal_imgs = {
            k: index_and_resize(goal_img_dict, bgr_input=False, **img_hparams)[None] for k, img_hparams in self.img_mapping.items()
        }
        self.goal = self.model.create_tasks(goals=goal_imgs)

    def load_lang(self, text):
        logging.warning(f'setting text goal: \"{text}\"')
        self.goal = self.model.create_tasks(texts=[text])

    def null_forward(self):
        img_obs = {img_hparams['obs_key']: np.zeros((img_hparams['height'], img_hparams['width'], 3), dtype=np.uint8) for img_hparams in self.img_mapping.values() if 'obs_key' in img_hparams}
        null_obs = dict(images=img_obs, qpos=np.zeros((14,)))
        self.forward(null_obs)

    def reset(self):
        self._action_history.clear()
        self.last_ac = None


def main(_):
    policy = OctoPolicy(*load_checkpoint(FLAGS.checkpoint))
    # compile the policy using a dummy "null" observation
    policy.null_forward()

    env = make_real_env(init_node=True)

    # Roll out the policy num_rollout times
    for rollout_num in range(10):

        last_input = None
        while last_input != "y":
            if last_input == "r":
                obs = env.reset()
            last_input = input("Continue with rollout (y; r to reset now)?")

        policy.reset()

        obs = env.reset()
        for _ in range(500):
            ac = policy.forward(obs.observation)
            obs = env.step(ac)

if __name__ == "__main__":
    app.run(main)
