#! /usr/bin/python3
# This is the highest level interface to interact with the widowx setup.

import time
import cv2
import argparse
import numpy as np
import logging
import traceback

from typing import Optional, Tuple, Any
from widowx_envs.utils.exceptions import Environment_Exception

# install from: https://github.com/youliangtan/edgeml
from edgeml.action import ActionClient, ActionServer, ActionConfig
from edgeml.internal.utils import mat_to_jpeg, jpeg_to_mat, compute_hash

##############################################################################


class WidowXConfigs:
    DefaultEnvParams = {
        "fix_zangle": 0.1,
        "move_duration": 0.2,
        "adaptive_wait": True,
        "move_to_rand_start_freq": 1,
        "override_workspace_boundaries": [
            [0.1, -0.15, -0.1, -1.57, 0],
            [0.45, 0.25, 0.18, 1.57, 0],
        ],
        "override_workspace_boundaries": None,
        "action_clipping": "xyz",
        "catch_environment_except": False,
        "start_state": [0.3, 0.0, 0.15, 0, 0, 0, 1], # pose when reset is called
        "skip_move_to_neutral": False,
        "return_full_image": False,
        "camera_topics": [
            {"name": "/blue/image_raw"}, 
            {"name": "/wrist/image_raw", "is_python_node": True}
        ],
        "digit_topics": [
            {"name": '/digit_left/image_raw', "width": 320, "height":240, "is_python_node": True},
            {"name": '/digit_right/image_raw', "width": 320, "height":240, "is_python_node": True}
        ],
    }
    DefaultActionConfig = ActionConfig(
        port_number=5556,
        action_keys=["init", "move", "gripper", "reset", "step_action", "reboot_motor"],
        observation_keys=["image", "state", "full_image"],
        broadcast_port=5556 + 1,
    )


def print_red(x): return print("\033[91m{}\033[00m".format(x))

##############################################################################


class WidowXStatus:
    NO_CONNECTION = 0
    SUCCESS = 1
    EXECUTION_FAILURE = 2
    NOT_INITIALIZED = 3

##############################################################################


class WidowXActionServer():
    """
    This is the highest level abstraction of the widowx setup. We will run
    this as a server, and we can have multiple clients connect to it and
    reveives the observation and control the widowx robot.
    """

    def __init__(self, port: int = 5556, testing: bool = False):
        edgeml_config = WidowXConfigs.DefaultActionConfig
        edgeml_config.port_number = port
        edgeml_config.broadcast_port = port + 1

        self.testing = testing  # TODO: remove this soon
        self.bridge_env = None
        self.__server = ActionServer(edgeml_config,
                                     obs_callback=self.__observe,
                                     act_callback=self.__action,
                                     log_level=logging.WARNING)

        self._env_params = {}  # default nothing
        self._image_size = None  # default None
        self._action_methods = {
            "init": self.__init,
            "gripper": self.__gripper,
            "move": self.__move,
            "step_action": self.__step_action,
            "reset": self.__reset,
            "reboot_motor": self.__reboot_motor,
        }

    def start(self, threaded: bool = False):
        """
        This starts the server. Default is blocking.
        """
        self.__server.start(threaded)

    def stop(self):
        """Stop the server."""
        self.__server.stop()
        return WidowXStatus.SUCCESS

    def init_robot(self, env_params, image_size):
        """Public method to init the robot"""
        if self.bridge_env:
            del self.bridge_env

        from widowx_envs.widowx_env import BridgeDataRailRLPrivateWidowX, BridgeDataDigitWidowX
        from multicam_server.topic_utils import IMTopic
        from tf.transformations import quaternion_from_euler
        from tf.transformations import quaternion_matrix

        # brute force way to convert json to IMTopic
        _env_params = env_params.copy()
        cam_imtopic = []
        for cam in _env_params["camera_topics"]:
            imtopic_obj = IMTopic.from_dict(cam)
            cam_imtopic.append(imtopic_obj)
        _env_params["camera_topics"] = cam_imtopic

        def get_tf_mat(pose):
            # convert pose to a 4x4 tf matrix, rpy to quat
            quat = quaternion_from_euler(pose[3], pose[4], pose[5])
            tf_mat = quaternion_matrix(quat)
            tf_mat[:3, 3] = pose[:3]
            return tf_mat

        self.get_tf_mat = get_tf_mat
        self.bridge_env = BridgeDataDigitWidowX(_env_params, image_size=image_size)
        print("Initialized bridge env.")

    def hard_reset(self) -> bool:
        """This conduct a hard reset the Widowxenv"""
        if self._image_size is None:
            print_red("env was not init before")
            return False
        self.init_robot(self._env_params, self._image_size)

    def __action(self, type: str, req_payload: dict) -> dict:
        if type not in self._action_methods:
            return {"status": WidowXStatus.EXECUTION_FAILURE}
        if type != "init" and self.bridge_env is None:
            print_red("WARNING: env not initialized.")
            return {"status": WidowXStatus.NOT_INITIALIZED}

        status = self._action_methods[type](req_payload)
        return {"status": status}

    def __init(self, payload) -> WidowXStatus:
        do_reinit = not self._env_params == payload["env_params"]
        self._env_params = payload["env_params"]
        self._image_size = payload["image_size"]

        if self.testing:
            print_red("WARNING: Running in testing mode, \
                no env will be initialized.")
            return WidowXStatus.NOT_INITIALIZED

        elif not do_reinit:
            print_red("env already initialized")
            self.__reset({})
            return WidowXStatus.SUCCESS

        self.init_robot(payload["env_params"], payload["image_size"])
        return WidowXStatus.SUCCESS
    
    def __reboot_motor(self, payload) -> WidowXStatus:
            joint_name = payload["joint_name"]
            print(f"Experimental: Rebooting motor {joint_name}")
            self.bridge_env.controller().reboot_motor(joint_name)
            return WidowXStatus.SUCCESS

    def __observe(self, types: list) -> dict:
        if self.bridge_env:
            # we will default return image and proprio only
            obs = self.bridge_env.current_obs()
        else:
            # use dummy img with random noise
            img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            obs = {"image": img, "state": {}, "full_image": img}
            print_red("WARNING: No bridge env not initialized.")
        return obs

    def __gripper(self, payload) -> WidowXStatus:
        if payload["open"] > 0.5:  # convert to bool, for future float support
            self.bridge_env.controller().open_gripper()
        else:
            self.bridge_env.controller().close_gripper()
        return WidowXStatus.SUCCESS

    def __move(self, payload) -> WidowXStatus:
        pose = payload["pose"]
        if pose.shape == (4, 4):
            eep = pose
        else:
            eep = self.get_tf_mat(pose)
        try:
            self.bridge_env.controller().move_to_eep(
                eep,
                duration=payload["duration"],
                blocking=payload["blocking"],
                step=False,
            )
            self.bridge_env._reset_previous_qpos()
        except Environment_Exception as e:
            print_red("Move execution error: {}".format(e))
            return WidowXStatus.EXECUTION_FAILURE
        return WidowXStatus.SUCCESS

    def __step_action(self, payload) -> WidowXStatus:
        self.bridge_env.step(payload["action"], blocking=payload["blocking"])
        return WidowXStatus.SUCCESS

    def __reset(self, payload) -> WidowXStatus:
        # NOTE(YL): in bridge env, the entire process is very stateful,
        #           even reset might not be enough to reset the state
        #           this will require further debug, a full refactoring
        # self.bridge_env.controller().move_to_neutral(duration=1.0)
        # self.bridge_env.controller().open_gripper()
        self.bridge_env.reset()
        self.bridge_env.start()
        return WidowXStatus.SUCCESS

##############################################################################


class WidowXClient():
    def __init__(self,
                 host: str = "localhost",
                 port: int = 5556,
                 ):
        """
        Args:
            :param host: the host ip address
            :param port: the port number
        """
        edgeml_config = WidowXConfigs.DefaultActionConfig
        edgeml_config.port_number = port
        edgeml_config.broadcast_port = port + 1
        self.__client = ActionClient(host, edgeml_config)
        print("Initialized widowx client.")

    def init(self,
             env_params: dict,
             image_size: int = 256,
             ) -> WidowXStatus:
        """
        Initialize the environment.
            :param env_params: a dict of env params
            :param image_size: the size of the image to return
        """
        payload = {"env_params": env_params,
                   "image_size": image_size}
        res = self.__client.act("init", payload)
        return WidowXStatus.NO_CONNECTION if res is None else res["status"]

    def move(self,
             pose: np.ndarray,
             duration: float = 1.0,
             blocking: bool = False,
             ) -> WidowXStatus:
        """
        Command the arm to move to a given pose in space.
            :param pose: dim of 6, [x, y, z, roll, pitch, yaw] or
                         a 4x4 tf matrix
            :param duration: time to move to the pose. Not implemented
            :param blocking: whether to block the server until the move is done
        """
        assert len(pose) == 6 or pose.shape == (4, 4), "invalid pose shape"
        _payload = {"pose": pose, "duration": duration, "blocking": blocking}
        res = self.__client.act("move", _payload)
        return WidowXStatus.NO_CONNECTION if res is None else res["status"]

    def move_gripper(self, state: float) -> WidowXStatus:
        """Open or close the gripper. 1.0 is open, 0.0 is closed."""
        res = self.__client.act("gripper", {"open": state})
        return WidowXStatus.NO_CONNECTION if res is None else res["status"]

    def step_action(self, action: np.ndarray, blocking=False) -> WidowXStatus:
        """
        Step the action. size of 5 (3trans) or 7 (3trans1rot)
        Note that the action is in relative space.
        """
        assert len(action) in [5, 7], "invalid action shape"
        res = self.__client.act("step_action", {"action": action,
                                                "blocking": blocking})
        return WidowXStatus.NO_CONNECTION if res is None else res["status"]

    def reset(self) -> WidowXStatus:
        """Reset the arm to the neutral position."""
        res = self.__client.act("reset", {})
        return WidowXStatus.NO_CONNECTION if res is None else res["status"]

    def get_observation(self) -> Optional[dict]:
        """
        Get the current camera image and proprioceptive state.
            :return a dict of observations
        """
        res = self.__client.obs()
        if res is None:
            return None
        # NOTE: this is a lossy conversion, but faster data transfer
        # res["full_image"] = jpeg_to_mat(res["full_image"])
        return res

    def stop(self):
        """Stop the client."""
        self.__client.stop()

def reboot_motor(self, joint_name: str):
        """Experimentation: Force Reboot the motor.
        Supported joint names:
            - waist, shoulder, elbow, forearm_roll,
            - wrist_angle, wrist_rotate, gripper, left_finger, right_finger
        """
        self.__client.act("reboot_motor", {"joint_name": joint_name})

##############################################################################


def show_video(client, duration, full_image=True):
    """This shows the video from the camera for a given duration."""
    start = time.time()
    while (time.time() - start) < duration:
        res = client.get_observation()
        if res is None:
            print("No observation available... waiting")
            continue

        if full_image:
            img = res["full_image"]
        else:
            img = res["image"]
            # if img.shape[0] != 3:  # sanity check to make sure it's not flattened
            img = (img.reshape(3, 256, 256).transpose(1, 2, 0) * 255).astype(np.uint8)
        cv2.imshow("img", img)
        cv2.waitKey(10)  # 10 ms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--server', action='store_true')
    parser.add_argument('--client', action='store_true')
    parser.add_argument('--ip', type=str, default='localhost')
    parser.add_argument('--port', type=int, default=5556)
    parser.add_argument('--test', action='store_true', help='run in test mode')
    args = parser.parse_args()

    if args.server:
        widowx_server = WidowXActionServer(port=args.port, testing=args.test)

        # try capture errors and hard restart the server
        while True:
            try:
                # this is a blocking call
                widowx_server.start()
            except Exception as e:
                if e == KeyboardInterrupt:
                    break
                print(traceback.format_exc())
                print_red(f"{e}, Restarting server...")
                widowx_server.stop()
                widowx_server.hard_reset()
        widowx_server.stop()

    if args.client:
        widowx_client = WidowXClient(host=args.ip, port=args.port)

        # NOTE: this normally takes 10 seconds when first time init
        widowx_client.init(WidowXConfigs.DefaultEnvParams, image_size=256)

        # This ensures that the robot is ready to be controlled
        obs = None
        while obs is None:
            obs = widowx_client.get_observation()
            time.sleep(1)
            print("Waiting for robot to be ready...")

        # Coordinate Convention:
        #  - x: forward
        #  - y: left
        #  - z: up

        # move left up with slanted gripper
        res = widowx_client.move(np.array([0.2, 0.1, 0.3, 0, 0.47, 0.3]))
        assert args.test or res == WidowXStatus.SUCCESS, "move failed"
        show_video(widowx_client, duration=1.5)

        # test reboot motor, the gripper should get loosed for a quick moment
        widowx_client.reboot_motor("wrist_angle")
        show_video(widowx_client, duration=2)

        # NOTE, use blocking to make sure the qpos is reset after the move
        # this is important so that step_action works in this initial position
        res = widowx_client.move(np.array([0.2, 0.1, 0.3, 0, 1.57, 0.]), blocking=True)
        show_video(widowx_client, duration=0.5)

        # close gripper
        print("Closing gripper...")
        res = widowx_client.move_gripper(0.0)
        assert args.test or res == WidowXStatus.SUCCESS, "gripper failed"
        show_video(widowx_client, duration=2.5)

        print("Run step_action for 25 steps")
        for _ in range(25):
            start_time = time.time()
            widowx_client.step_action(np.array([-0.005, 0.005, 0.005, 0, 0, 0, 0]))
            print(f"Time taken for each step: {time.time() - start_time}")
        show_video(widowx_client, duration=0.5)

        # move right down
        res = widowx_client.move(np.array([0.2, -0.1, 0.1, 0, 1.57, 0]))
        assert args.test or res == WidowXStatus.SUCCESS, "move failed"
        show_video(widowx_client, duration=1.5)

        # open gripper
        print("Opening gripper...")
        res = widowx_client.move_gripper(1.0)
        assert args.test or res == WidowXStatus.SUCCESS, "gripper failed"
        show_video(widowx_client, duration=2.5)

        widowx_client.stop()
        print("Done all")


if __name__ == '__main__':
    main()
