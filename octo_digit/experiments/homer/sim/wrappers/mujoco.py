import gym
import numpy as np
from scipy.spatial.transform import Rotation
from experiments.homer.sim.wrappers.dmcgym import DMCGYM
import mujoco_manipulation
import tensorflow as tf

def convert_obs(obs):
    obs["image_primary"] = obs.pop("pixels")
    obs["proprio"] = np.concatenate(
        [
            obs.pop("end_effector_pos"),
            Rotation.from_quat(obs.pop("end_effector_quat")).as_euler("xyz"),
            obs.pop("right_finger_qpos"),
        ]
    )
    obs.pop("left_finger_qpos")
    return obs


def filter_info_keys(info):
    keep_keys = ["place_success"]
    return {k: v for k, v in info.items() if k in keep_keys}

def unstandardize_action(action):
    # reverse standardization transform
    gripper_action = (1 - action[:, -1:]) * 255
    return np.concatenate([action[:, :6], gripper_action], axis=1)

class MujocoManipWrapper(gym.Wrapper):
    """
    Wrapper for Mujoco Manipulation sim environments.
    """

    def __init__(self, env: gym.Env, goals_path: str,):
        super().__init__(env)
        self.env = env
        self.observation_space = gym.spaces.Dict(
            {
                "image_primary": gym.spaces.Box(
                    low=np.zeros((128, 128, 3)),
                    high=255 * np.ones((128, 128, 3)),
                    dtype=np.uint8,
                ),
                "proprio": gym.spaces.Box(
                    low=np.zeros((7,)), high=np.ones((7,)), dtype=np.uint8
                ),
            }
        )
        self.current_goal = None
        self.place_success = False
        with tf.io.gfile.GFile(tf.io.gfile.join(goals_path), "rb") as f:
            self.goals = np.load(f, allow_pickle=True).item()

    def step(self, action, *args):
        action = unstandardize_action(action[None])[0]
        obs, reward, done, trunc, info = self.env.step(action, *args)
        info = filter_info_keys(info)
        self.place_success = info["place_success"]
        return (
            convert_obs(obs),
            reward,
            done,
            trunc,
            info,
        )

    def get_goal(self):
        return self.current_goal

    def get_instruction(self):
        return "put the shoe on the circle"

    def get_episode_metrics(self):
        return {"place_success": self.place_success}

    def reset(self, **kwargs):
        idx = np.random.randint(len(self.goals["observation"]["image"]))
        goal_image = self.goals["observation"]["image"][idx]
        original_object_positions = self.goals["observation"]["info/initial_positions"][
            idx
        ]
        # original_object_quats = self.goals["observation"]["info/initial_quats"][idx]
        target_position = self.goals["observation"]["info/target_position"][idx]
        object_names = self.goals["observation"]["info/object_names"][idx]
        target_object = self.goals["observation"]["info/target_object"][idx]
        self.env.task.change_props(object_names)
        self.env.task.init_prop_poses = original_object_positions
        self.env.task.target_pos = target_position
        self.env.target_obj = target_object
        obs, info = self.env.reset()
        obs = convert_obs(obs)

        goal = {"image_primary": goal_image[None]}

        self.current_goal = goal
        self.place_success = False

        info = filter_info_keys(info)
        return obs, info

# register gym environment
gym.register(
    "franka-shoe-pick-place",
    entry_point=lambda base_env_kwargs={}, **kwargs: MujocoManipWrapper(DMCGYM(
        mujoco_manipulation.load("franka_shoe_pick_and_place", **base_env_kwargs)), **kwargs
    ),
)
