import time

import gym
import numpy as np
from pyquaternion import Quaternion
from widowx_envs.widowx_env_service import WidowXClient


def state_to_eep(xyz_coor, zangle: float):
    """
    Implement the state to eep function.
    Refered to `bridge_data_robot`'s `widowx_controller/widowx_controller.py`
    return a 4x4 matrix
    """
    assert len(xyz_coor) == 3
    DEFAULT_ROTATION = np.array([[0, 0, 1.0], [0, 1.0, 0], [-1.0, 0, 0]])
    new_pose = np.eye(4)
    new_pose[:3, -1] = xyz_coor
    new_quat = Quaternion(axis=np.array([0.0, 0.0, 1.0]), angle=zangle) * Quaternion(
        matrix=DEFAULT_ROTATION
    )
    new_pose[:3, :3] = new_quat.rotation_matrix
    # yaw, pitch, roll = quat.yaw_pitch_roll
    return new_pose


def wait_for_obs(widowx_client):
    obs = widowx_client.get_observation()
    while obs is None:
        print("Waiting for observations...")
        obs = widowx_client.get_observation()
        time.sleep(1)
    return obs


unflat_sizes = {
    "image_0": (480, 640),
    "image_1": (480, 640),
    "digit": (240, 320),
    "background": (240, 320),
}
unflat_sizes = {
    "image_0": (256, 256),
    "image_1": (128, 128),
    "digit": (256, 256),
    "background": (256, 256),
}
possible_prefixes = unflat_sizes.keys()


def convert_obs(obs, im_size=256):
    processed_obs = {}
    for key, val in obs.items():
        prefix = None
        for possible_prefix in possible_prefixes:
            if key.startswith(possible_prefix):
                prefix = possible_prefix
                break

        if prefix is not None:
            resize_size = unflat_sizes[prefix]
            full_size = (3,) + resize_size
            processed_obs[key] = (
                val.reshape(full_size).transpose(1, 2, 0) * 255
            ).astype(np.uint8)
        else:
            processed_obs[key] = val
    if "mel_spectro" in obs:
        processed_obs["mel_spectro"] = obs["mel_spectro"].reshape((128, 128))
    return processed_obs


def null_obs(img_size):
    return {
        "image_primary": np.zeros((img_size, img_size, 3), dtype=np.uint8),
        "proprio": np.zeros((8,), dtype=np.float64),
    }


class WidowXGym(gym.Env):
    """
    A Gym environment for the WidowX controller provided by:
    https://github.com/rail-berkeley/bridge_data_robot
    Needed to use Gym wrappers.
    """

    def __init__(
        self,
        widowx_client: WidowXClient,
        im_size: int = 256,
        blocking: bool = True,
        sticky_gripper_num_steps: int = 1,
    ):
        self.widowx_client = widowx_client
        self.im_size = im_size
        self.blocking = blocking
        self.observation_space = gym.spaces.Dict(
            {
                "image_0": gym.spaces.Box(
                    low=np.zeros((im_size, im_size, 3)),
                    high=255 * np.ones((im_size, im_size, 3)),
                    dtype=np.uint8,
                ),
                "proprio": gym.spaces.Box(
                    low=np.ones((8,)) * -1, high=np.ones((8,)), dtype=np.float64
                ),
            }
        )  # TODO: update this
        self.action_space = gym.spaces.Box(
            low=np.zeros((7,)), high=np.ones((7,)), dtype=np.float64
        )
        self.sticky_gripper_num_steps = sticky_gripper_num_steps
        self.is_gripper_closed = False
        self.num_consecutive_gripper_change_actions = 0

    def step(self, action):
        # sticky gripper logic
        start_time = time.time()
        if (action[-1] < 0.5) != self.is_gripper_closed:
            self.num_consecutive_gripper_change_actions += 1
        else:
            self.num_consecutive_gripper_change_actions = 0

        if self.num_consecutive_gripper_change_actions >= self.sticky_gripper_num_steps:
            self.is_gripper_closed = not self.is_gripper_closed
            self.num_consecutive_gripper_change_actions = 0
        action[-1] = 0.0 if self.is_gripper_closed else 1.0

        self.widowx_client.step_action(action, blocking=self.blocking)
        after_step = time.time()

        raw_obs = self.widowx_client.get_observation()
        # print(raw_obs.keys())
        after_rec_obs = time.time()
        truncated = False
        if raw_obs is None:
            # this indicates a loss of connection with the server
            # due to an exception in the last step so end the trajectory
            truncated = True
            obs = null_obs(self.im_size)  # obs with all zeros
        else:
            obs = convert_obs(raw_obs, self.im_size)
        end_time = time.time()

        # def _to_float32_flat_image(image):
        #     return np.float32(image.flatten()) / 255.0
        # def _get_processed_image(image=None, size=None):

        #     # from skimage.transform import resize
        #     # downsampled_trimmed_image = resize(image, (size, size), anti_aliasing=True, preserve_range=True).astype(np.uint8)
        #     downsampled_trimmed_image = np.transpose(image, (2, 0, 1))
        #     return _to_float32_flat_image(downsampled_trimmed_image)

        # obs = {}
        # obs['image_0'] = _get_processed_image(np.zeros((480, 640, 3)))
        # obs['image_1'] = _get_processed_image(np.zeros((480, 640, 3)))
        # obs['spectro'] = np.zeros((13230,))
        # obs['digit_l'] = obs['digit_r'] = _get_processed_image(np.zeros((240, 320, 3)))
        # obs['state'] = np.zeros(7,)
        # obs = convert_obs(obs, self.im_size)

        print(
            f"Step:  enter-act={after_step - start_time}  after_act-receive_obs={after_rec_obs - after_step}  receive_obs-end={end_time - after_rec_obs}    end={end_time-start_time}"
        )

        return obs, 0, False, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.widowx_client.reset()

        self.is_gripper_closed = False
        self.num_consecutive_gripper_change_actions = 0

        def _to_float32_flat_image(image):
            return np.float32(image.flatten()) / 255.0

        def _get_processed_image(image=None, size=None):
            # from skimage.transform import resize
            # downsampled_trimmed_image = resize(image, (size, size), anti_aliasing=True, preserve_range=True).astype(np.uint8)
            downsampled_trimmed_image = np.transpose(image, (2, 0, 1))
            return _to_float32_flat_image(downsampled_trimmed_image)

        raw_obs = wait_for_obs(self.widowx_client)
        print(raw_obs.keys())
        obs = convert_obs(raw_obs, self.im_size)
        # obs = {}
        # obs['image_0'] = _get_processed_image(np.zeros((480, 640, 3)))
        # obs['image_1'] = _get_processed_image(np.zeros((480, 640, 3)))
        # obs['spectro'] = np.zeros((13230,))
        # obs['digit_l'] = obs['digit_r'] = _get_processed_image(np.zeros((240, 320, 3)))
        # obs['state'] = np.zeros(7,)
        # obs = convert_obs(obs, self.im_size)

        return obs, {}
