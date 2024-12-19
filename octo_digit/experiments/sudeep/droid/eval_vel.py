import os
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

# r2d2 robot imports
from r2d2.user_interface.eval_gui import EvalGUI

tf.config.set_visible_devices([], "GPU")

np.set_printoptions(suppress=True)

logging.set_verbosity(logging.INFO)

FLAGS = flags.FLAGS
flags.DEFINE_bool("deterministic", True, "Whether to sample action deterministically")
flags.DEFINE_float("temperature", 1e-7, "Temperature for sampling actions")
flags.DEFINE_string("checkpoint", None, "Path to checkpoint")


def normalize(vec, eps=1e-12):
    norm = np.linalg.norm(vec, axis=-1)
    norm = np.maximum(norm, eps)
    return vec / norm
    out = (vec.T / norm).T
    return out


def rmat_to_euler_scipy(rot_mat, degrees=False):
    euler = R.from_matrix(rot_mat).as_euler("xyz", degrees=degrees)
    return euler


def euler_to_rmat_scipy(euler, degrees=False):
    return R.from_euler("xyz", euler, degrees=degrees).as_matrix()


def R6_to_rmat_scipy(r6_mat):
    a1, a2 = r6_mat[:3], r6_mat[3:]
    b1 = normalize(a1)
    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = normalize(b2)
    b3 = np.cross(b1, b2, axis=-1)
    out = np.vstack((b1, b2, b3))
    return out


def rotate_velocity(vel, frame, degrees=False):
    t_frame, dt = frame[:3], vel[:3]
    R_frame = euler_to_rmat_scipy(frame[3:6], degrees=degrees)

    is_R6 = len(vel) > 7
    dR = R6_to_rmat_scipy(vel[3:9]) if is_R6 else \
         euler_to_rmat_scipy(vel[3:6], degrees=degrees)

    dt_new = np.matmul(R_frame, dt)
    dR_euler_new = rmat_to_euler_scipy(np.matmul(R_frame, np.matmul(dR, R_frame.T)), degrees=degrees)
    result = np.concatenate([dt_new, dR_euler_new])
    return result


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
        horizon=model.config["dataset_kwargs"]["traj_transform_kwargs"]["window_size"],
    )
    return policy_fn, action_denorm_fn, experiment_hparams, model


class OctoPolicy:
    def __init__(self, policy_fn, action_denorm_fn, exp_hparams, model):
        self.policy_fn = policy_fn
        self.action_denorm_fn = action_denorm_fn
        self.img_mapping = exp_hparams['img_mapping']
        self.abs_gripper_ac = exp_hparams['absolute_gripper_action']
        self.action_in_wrist_frame = exp_hparams['action_in_wrist_frame']
        self.model = model

        self.goal = self.model.create_tasks(texts=["Execute a task."])
        self._last_time = None
        self._action_history = deque(maxlen=int(exp_hparams['control_horizon']))
        self.exp_weight = float(exp_hparams['exp_weight'])

    def _convert_obs(self, observation):
        state = observation['robot_state']
        print('pos', state['cartesian_position'])
        proprio = np.concatenate((state['cartesian_position'], np.array([state['gripper_position']])))
        obs_dict = {
            k: index_and_resize(observation["image"], bgr_input=True, **img_hparams) for k, img_hparams in self.img_mapping.items()
        }
        obs_dict['proprio'] = proprio.astype(np.float32)
        return obs_dict

    def forward(self, observation):
        obs_hist = [self._convert_obs(observation)]

        action = np.array(self.policy_fn(obs_hist, self.goal))
        self._action_history.append(action)
        num_actions = len(self._action_history)
        curr_act_preds = np.stack(
                [
                    pred_actions[i]
                    for (i, pred_actions) in zip(
                        range(num_actions - 1, -1, -1), self._action_history
                    )
                ]
            )

        # more recent predictions get exponentially *less* weight than older predictions
        weights = np.exp(-self.exp_weight * np.arange(num_actions))
        weights = weights / weights.sum()
        # compute the weighted average across all predictions for this timestep
        action = self.action_denorm_fn(np.sum(weights[:, None] * curr_act_preds, axis=0))

        # denormalize action using dataset mean/std stats
        is_R6 = len(action) > 7 # assume this is an R6 action

        # move action to current wrist frame (if enabled)
        if self.action_in_wrist_frame:
            cart_pos = observation['robot_state']['cartesian_position']
            action[:6] = rotate_velocity(action[:-1], cart_pos)
        elif is_R6:
            action[3:6] = rmat_to_euler_scipy(R6_to_rmat_scipy(action[3:9]))

        if is_R6:
            action[6] = action[-1] # copy the gripper action to spot 7
            action = action[:7]

        logging.debug(f"cur action: {action}")

        cur_time = time.time()
        if self._last_time is not None:
            hz = 1.0 / (cur_time - self._last_time)
            logging.info(f"Effective HZ: {hz}")
        self._last_time = cur_time

        if self.abs_gripper_ac:
            action[-1] -= observation['robot_state']['gripper_position']

        print(f"action: {action}")
        return np.clip(action, -1, 1)

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
        robot_state = dict(cartesian_position=np.zeros((6,)), gripper_position=0)
        null_obs = dict(image=img_obs, robot_state=robot_state)
        self.forward(null_obs)


def main(_):
    policy = OctoPolicy(*load_checkpoint(FLAGS.checkpoint))
    # compile the policy using a dummy "null" observation
    policy.null_forward()

    # start up R2D2 eval gui
    EvalGUI(policy=policy)


if __name__ == "__main__":
    app.run(main)
