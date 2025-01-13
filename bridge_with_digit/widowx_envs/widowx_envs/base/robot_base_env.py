#! /usr/bin/python3

from re import S
from widowx_envs.control_loops import Environment_Exception
import time
from widowx_envs.base.base_env import BaseEnv

from widowx_envs.utils import transformation_utils as tr
import numpy as np
from widowx_envs.utils.exceptions import Image_Exception
import copy
from widowx_envs.utils import AttrDict
import threading

# Rospkg lib
import rospy
from multicam_server.angle_streamer import AngleStreamer
from multicam_server.topic_utils import IMTopic
from multicam_server.camera_recorder import CameraRecorder
from multicam_server.sensor_recorder import IMURecorder, MicRecorder
from widowx_controller.widowx_controller import WidowX_Controller
from widowx_controller.widowx_controller import publish_transform
from PIL import Image
import datetime 

import logging
import json
from gym import spaces

from widowx_envs.utils import read_yaml_file
import os
from widowx_envs.utils.exceptions import Environment_Exception
from std_msgs.msg import Bool

def pix_resize(pix, target_width, original_width):
    return np.round((copy.deepcopy(pix).astype(np.float32) *
              target_width / float(original_width))).astype(np.int64)


class RobotBaseEnv(BaseEnv):
    """
    Inherited from abstract class BaseEnv
    """   
    def __init__(self, env_params):
        self._hp = self._default_hparams()
        self._override_defaults(env_params)
        self.savedir = None
        logging.info('initializing environment for {}'.format(self._hp.robot_name))
        self._robot_name = self._hp.robot_name
        self._setup_robot()

        self._obs_tol = self._hp.OFFSET_TOL

        assert (self._hp.gripper_attached == 'default' and not self._hp.continuous_gripper) or self._hp.gripper_attached != 'default', 'If gripper_attached == \'default\', continuous_gripper has to be False'
        self._controller = self._hp.robot_controller(
            self._robot_name,
            self._hp.print_debug,
            gripper_attached=self._hp.gripper_attached,
            gripper_params=self._hp.gripper_params,
            normal_base_angle=self._hp.workspace_rotation_angle_z)

        logging.getLogger('robot_logger').info('---------------------------------------------------------------------------')
        for name, value in self._hp.items():
            logging.getLogger('robot_logger').info('{}= {}'.format(name, value))
        logging.getLogger('robot_logger').info('---------------------------------------------------------------------------')
        self._cameras_to_reset = []
        self._cameras = [CameraRecorder(t, False, False) for t in self._hp.camera_topics]
        self._camera_info = [c.camera_info for c in self._cameras]
        camera_resetters = [rospy.Publisher(cam.topic_name + '/reset_flag', Bool) for cam in self._cameras]
        self._cameras_to_reset.extend(camera_resetters)

        if self._hp.get("record_img", False): 
            img = self._cameras[0].get_image()[1][..., ::-1]
            img = Image.fromarray(img)
            time_prefix = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            img.save(f'/home/robonet/trainingdata/robonetv2/bridge_data_v2/imgs/{time_prefix}.jpeg')

        if "depth_camera_topics" in self._hp:
            print("depth camera topics", self._hp.depth_camera_topics)
            self._depth_cameras = [CameraRecorder(t, False, False) for t in self._hp.depth_camera_topics]
            self._depth_camera_info = [c.camera_info for c in self._depth_cameras]
        else:
            self._depth_cameras = []
            self._depth_camera_info = []

        if "digit_topics" in self._hp: 
            for i, top in enumerate(self._hp.digit_topics): 
                if isinstance(top, dict): 
                    self._hp.digit_topics[i] = IMTopic(
                        top["name"], width=top["width"], height=top["height"], is_python_node=top["is_python_node"]
                    )
            print("digit topics", self._hp.digit_topics)
            self._digit_cameras = [CameraRecorder(t, False, False) for t in self._hp.digit_topics]
            self._digit_camera_info = [c.camera_info for c in self._digit_cameras]
            digit_resetters = [rospy.Publisher(cam.topic_name + '/reset_flag', Bool) for cam in self._digit_cameras]
            self._cameras_to_reset.extend(digit_resetters)
        else: 
            self._digit_cameras = [] 
            self._digit_camera_info = []
        
        if "imu_topic" in self._hp and self._hp.imu_topic: # Simple topic: just topic name
            self._IMU = IMURecorder(self._hp.imu_topic)
        else: 
            self._IMU = None 

        if "mic_topic" in self._hp and self._hp.mic_topic: 
            self._mic = MicRecorder(self._hp.mic_topic)
        else: 
            self._mic = None 

        if self._hp.get('stream_angles', False): 
            self._angle_streamer = AngleStreamer(self)

        self._reset_counter = 0
        self._controller.open_gripper(True)

        if len(self._cameras) > 1:
            first_cam_dim = (self._cameras[0].img_height, self._cameras[1].img_width)
            assert all([(c.img_height, c.img_width) == first_cam_dim for c in self._cameras[1:]]), \
                'Camera image streams do not match)'
        if self._cameras:
            self._height, self._width = self._cameras[0].img_height, self._cameras[0].img_width
        if self._digit_cameras: 
            self._digit_height, self._digit_width = self._digit_cameras[0].img_height, self._digit_cameras[0].img_width

        if len(self._cameras) == 1:
            self._cam_names = ['front']
        elif len(self._cameras) == 2:
            self._cam_names = ['front', 'left']
        else:
            self._cam_names = ['cam{}'.format(i) for i in range(len(self._cameras))]

        self._reset_counter, self._previous_target_qpos = 0, None
        
        if not self._hp.start_at_current_pos:
            self._controller.move_to_neutral(duration=3)
        eep = self._controller.get_cartesian_pose()[:3]
        print('\n current eep:')
        print(eep)
        print('\n')


        self.action_space = spaces.Box(
            np.asarray([-0.05, -0.05, -0.05, -0.25, -0.25, -0.25, 0.]),
            np.asarray([0.05, 0.05, 0.05, 0.25, 0.25, 0.25, 1.0]),
            dtype=np.float32)
        if self._hp.action_mode == '3trans':
            self._adim = self._base_adim = 4
            self._sdim = self._base_sdim = 4
            self.action_space.low = np.concatenate([self.action_space.low[:3], np.array([self.action_space.low[-1]])])
            self.action_space.high = np.concatenate([self.action_space.high[:3], np.array([self.action_space.high[-1]])])
        elif self._hp.action_mode == '3trans1rot':
            self._adim = self._base_adim = 5
            self._sdim = self._base_sdim = 5
            self.action_space.low = np.concatenate([self.action_space.low[:3], self.action_space.low[-2:]])
            self.action_space.high = np.concatenate([self.action_space.high[:3], self.action_space.high[-2:]])
        elif self._hp.action_mode == '3trans3rot':
            self._adim = self._base_adim = 7
            self._sdim = self._base_sdim = 7
        else:
            raise NotImplementedError('action mode {} not supported!'.format(self._hp.action_mode))

        self.time_for_get_obs = 0.08  # measured empirically, might need to be adjusted.

    def controller(self):
        """Public method to access the controller"""
        if self._controller is None:
            raise NotImplementedError('controller not implemented')
        # self._reset_previous_qpos()  # TODO (YL) check if this is necessary
        return self._controller

    def _default_hparams(self):
        default_dict = {'robot_name': None,
                        'robot_controller': WidowX_Controller,
                        'gripper_attached': 'default',
                        'camera_topics': [IMTopic('/cam0/image_raw', flip=True)],
                        'start_at_neutral': False,
                        'start_at_current_pos': False,
                        'OFFSET_TOL': 0.1,
                        'lower_bound_delta': [0., 0., 0., 0., 0.],
                        'upper_bound_delta': [0., 0., 0., 0., 0.],
                        'print_debug': False,
                        'move_duration': 0.3,
                        'action_clipping': 'xyz',
                        'override_workspace_boundaries': None,
                        'resetqpos_after_every_step': False,
                        'absolute_grasp_action': True,
                        'continuous_gripper': True,
                        'action_mode': '3trans3rot',
                        'start_state': [],
                        'wait_time': 0,
                        'adaptive_wait': True,
                        'workspace_rotation_angle_z': 0,
                        'wait_until_gripper_pose_reached': False,
                        'catch_environment_except': False,
                        'gripper_params': AttrDict(
                            des_pos_max=1,
                            des_pos_min=0,
                        ),
        }
        parent_params = super(RobotBaseEnv, self)._default_hparams()
        parent_params.update(default_dict)
        return parent_params

    def _setup_robot(self):
        if self._hp.override_workspace_boundaries is not None:
            self._low_bound = np.array(self._hp.override_workspace_boundaries[0])
            self._high_bound = np.array(self._hp.override_workspace_boundaries[1])
        else:
            config_file_path = '/'.join(__file__.split('/')[:-1]) + '/robot_configs.json'
            error = False
            try:
                robot_configs = json.load(open(config_file_path, 'r'))
                self._high_bound = np.array(robot_configs[self._robot_name]['bound'][1])
                self._low_bound = np.array(robot_configs[self._robot_name]['bound'][0])
            except ValueError:
                logging.error("Did you fill out the config file (stored at {})?".format(config_file_path))
            except KeyError:
                logging.error("Robot {} not valid! Is it in the config file (stored at{})? It may be not supported yet.".format(self._robot_name, config_file_path))
            if error:
                exit(1)

        self._high_bound += np.array(self._hp.upper_bound_delta, dtype=np.float64)
        self._low_bound += np.array(self._hp.lower_bound_delta, dtype=np.float64)
        self.rotate_workspace()

    def rotate_workspace(self):
        points = np.c_[np.array([self._low_bound[:3], self._high_bound[:3]]), [1, 1]]
        transf_matrix = tr.RpToTrans(tr.eulerAnglesToRotationMatrix([0, 0, self._hp.workspace_rotation_angle_z]), [0, 0, 0])
        rotated_points = np.dot(transf_matrix, points.T).T[..., :3]
        self._low_bound[:3] = rotated_points.min(axis=0)
        self._high_bound[:3] = rotated_points.max(axis=0)

    def _next_qpos(self, action):
        prev_transform, prev_gripperstate = self.get_target_state()
        delta_transform, grasp_action = tr.action2transform_local(action, self._controller.get_cartesian_pose()[:3])
        next_transform = delta_transform.dot(prev_transform)
        if self._hp.absolute_grasp_action:
            new_gripperstate = np.array(grasp_action)[None]
        else:
            new_gripperstate = np.clip(prev_gripperstate + grasp_action, self.action_space.low[-1], self.action_space.high[-1])
        if self._hp.action_clipping == 'xyz':
            new_next_transform = copy.deepcopy(next_transform)
            new_next_transform[:3, 3] = np.clip(next_transform[:3, 3], self._low_bound[:3], self._high_bound[:3])
            if not np.all(new_next_transform == next_transform):
                logger = logging.getLogger('robot_logger')
                logger.info("action clipped!")
                logger.info('delta {}'.format(next_transform[:3, 3] - new_next_transform[:3, 3]))
                print("action clipped!")
            # print(f"transform: ", next_transform[:3, 3]) 
            # print(f"new_transform: ", new_next_transform[:3, 3])
            next_transform = new_next_transform
        return next_transform, new_gripperstate

    def get_target_state(self):
        return tr.state2transform(self._previous_target_qpos, self._controller.default_rot)

    def step(self, action, tstamp_return_obs=None, blocking=True):
        """
        Applies the action and steps simulation
        :param action: action at time-step
        :param tstamp_return_obs: time-step when to return oberservations
        :return: obs dict where:
                  -each key is an observation at that step
                  -keys are constant across entire datastep (e.x. every-timestep has 'state' key)
                  -keys corresponding to numpy arrays should have constant shape every timestep (for caching)
                  -images should be placed in the 'images' key in a (ncam, ...) array
        """
        assert action.shape[0] == self._base_adim, "Action should have shape ({},) but has shape {}".format(self._base_adim, action.shape)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        if self._hp.action_mode == '3trans1rot':
            action = np.concatenate([action[:3], np.zeros(2), action[-2:]])  # insert zeros for pitch and roll
        if self._hp.action_mode == '3trans':
            action = np.concatenate([action[:3], np.zeros(3), np.array([action[-1]])])  # insert zeros for pitch, roll, yaw

        new_transform, new_gripperstate = self._next_qpos(action)

        # assume that the gripper open state is 1. and the close state is 0.

        if self._hp.continuous_gripper:
            self._controller.set_continuous_gripper_position(new_gripperstate)
        else:
            gripper_midpoint = 0.5
            if new_gripperstate < gripper_midpoint:
                self._controller.close_gripper()
            else:
                self._controller.open_gripper()

        t0 = time.time()
        publish_transform(new_transform, 'commanded_transform')

        self.move_except = False
        if self._hp.catch_environment_except:
            try:
                self._controller.move_to_eep(new_transform, duration=self._hp.move_duration, blocking=blocking)
            except:
                self.move_except = True
        else:
            self._controller.move_to_eep(new_transform, duration=self._hp.move_duration, blocking=blocking)

        if self._hp.wait_until_gripper_pose_reached:
            self._controller.wait_until_gripper_position_reached()
        logging.getLogger('robot_logger').info('time to set pos'.format(time.time() - t0))

        self._previous_target_qpos = tr.transform2state(new_transform, new_gripperstate, self._controller.default_rot)
        if self._hp.resetqpos_after_every_step:
            self._reset_previous_qpos()  # this can cause accumulating errors
        time.sleep(self._hp.wait_time)
        
        if tstamp_return_obs is None:
            return self.current_obs() # or None?

        if self._hp.adaptive_wait:
            t1 = time.time()
            self.adaptive_wait(tstamp_return_obs - self.time_for_get_obs * 1.1)
            # print('adaptive wait for ', time.time() - t1)
        obs = self.current_obs()
        return obs

    def adaptive_wait(self, time_stamp):
        """
        Args:
            time_stamp:  wait until timestamp
        """
        while time.time() < time_stamp:
            time.sleep(0.001)

    def get_full_state(self):
        eep = self._controller.get_cartesian_pose(matrix=True)
        return tr.transform2state(eep, self._controller.get_gripper_position(), self._controller.default_rot)\


    def current_obs(self):
        obs = {}
        t0 = time.time()
        j_angles, j_vel, eep = self._controller.get_state()
        obs['joint_effort'] = self._controller.get_joint_effort()

        obs['qpos'] = j_angles
        if j_vel is not None:
            obs['qvel'] = j_vel

        full_state = self.get_full_state()
        obs['full_state'] = full_state
        if self._hp.action_mode == '3trans3rot':
            obs['state'] = full_state
        elif self._hp.action_mode == '3trans':
            if not (abs(full_state[3]) < 0.2 and abs(full_state[4]) < 0.2):
                print("out of plane rotation detected!")
                raise Environment_Exception
            obs['state'] = np.concatenate([full_state[:3], np.array(full_state[6:])])  # remove roll and pitch since they are (close to) zero
        elif self._hp.action_mode == '3trans1rot':
            if not (abs(full_state[3]) < 0.2 and abs(full_state[4]) < 0.2):
                print("out of plane rotation detected!")
                raise Environment_Exception
            obs['state'] = np.concatenate([full_state[:3], full_state[5:]])  # remove roll and pitch since they are (close to) zero
        obs['desired_state'] = self._previous_target_qpos
        obs['time_stamp'] = rospy.get_time()
        obs['eef_transform'] = self._controller.get_cartesian_pose(matrix=True)
        self._last_obs = copy.deepcopy(obs)
        
        obs['images'] = self.render()
        if self._depth_cameras:
            obs['depth_images'] = self.depth_render()
        if self._digit_cameras: 
            obs['digit_images'] = self.digit_render()
        logging.getLogger('robot_logger').info('time for rendering {}'.format(time.time() - t0))

        obs['high_bound'], obs['low_bound'] = copy.deepcopy(self._high_bound), copy.deepcopy(self._low_bound)

        obs['env_done'] = False
        obs['t_get_obs'] = time.time() - t0

        obs['xyz'] = self._controller.get_cartesian_pose()[:3]
        obs["transform_pose"] = self._controller.get_cartesian_pose(matrix=True)

        if self._IMU: 
            obs['imu'] = self._IMU.get_reading()
        
        if self._mic: 
            obs['mic'] = self._mic.get_reading()
        return obs

    def _move_to_state(self, target_xyz, target_zangle, duration=1.5):
        self._controller.move_to_state(target_xyz, target_zangle, duration=duration)

    def _reset_previous_qpos(self):
        rospy.sleep(0.2)  # TODO (YL) check if this is necessary
        self._previous_target_qpos = self.get_full_state()
        # don't track orientation of axes which are not controlled
        if self._hp.action_mode == '3trans1rot':
            self._previous_target_qpos[3:5] = np.zeros(2)
        elif self._hp.action_mode == '3trans':
            self._previous_target_qpos[3:6] = np.zeros(3)

    def _end_reset(self):
        self._reset_previous_qpos()
        obs = self.current_obs()
        return obs

    def move_to_neutral(self, duration=2.):
        self._controller.move_to_neutral(duration)
        self._reset_previous_qpos()

    def move_to_grip(self, duration=2.): 
        self._controller.move_to_grip(duration)
        self._reset_previous_qpos()

    def move_to_neutral_low(self, duration=3.): 
        self._controller.move_to_neutral_low(duration)
        self._reset_previous_qpos()
        
    def reset(self, itraj=None, reset_state=None):
        """
        Resets the environment and returns initial observation
        :return: obs dict (look at step(self, action) for documentation)
        """
        self._controller.open_gripper(True)
        if self._hp.start_at_current_pos:
            return self._end_reset()

        self.move_to_neutral()
        if self._hp.start_at_neutral:
            return self._end_reset()

        if self._hp.start_state:
            start_state = np.array(self._hp.start_state)
            xyz = np.array(start_state[:3])
            theta = start_state[3]
            self._move_to_state(xyz, theta, 2.)
        else:
            rand_xyz = np.random.uniform(self._low_bound[:3], self._high_bound[:3])
            rand_zangle = np.random.uniform(self._low_bound[3], self._high_bound[3])
            self._move_to_state(rand_xyz, rand_zangle, 2.)
        time.sleep(self._hp.wait_time)
        return self._end_reset()

    def valid_rollout(self):
        """
        Checks if the environment is currently in a valid state
        Common invalid states include:
            - object falling out of bin
            - mujoco error during rollout
        :return: bool value that is False if rollout isn't valid
        """
        return True

    def render(self):
        """ Grabs images form cameras.
        If returning multiple images asserts timestamps are w/in OBS_TOLERANCE, and raises Image_Exception otherwise

        - dual: renders both left and main cameras
        - left: renders only left camera
        - main: renders only main (front) camera
        :param mode: Mode to render with (dual by default)
        :return: uint8 numpy array with rendering from sim
        """
        if not self._cameras:
            return []
        cam_imgs = self._render_from_camera(self._cameras)

        images = np.zeros((self.ncam, self._height, self._width, 3), dtype=np.uint8)
        for c, img in enumerate(cam_imgs):
            images[c] = img[:, :, ::-1]

        return images

    def depth_render(self):
        if not self._depth_cameras:
            return []
        cam_imgs = self._render_from_camera(self._depth_cameras)

        images = np.zeros((len(cam_imgs), cam_imgs[0].shape[0], cam_imgs[0].shape[1]), dtype=np.uint16)
        for c, img in enumerate(cam_imgs):
            images[c] = img

        return images

    def digit_render(self):
        if not self._digit_cameras: 
            return []
        cam_imgs = self._render_from_camera(self._digit_cameras)

        images = np.zeros((self.ndigit, self._digit_height, self._digit_width, 3), dtype=np.uint8)
        for c, img in enumerate(cam_imgs):
            images[c] = img[:, :, ::-1]

        return images


    def _render_from_camera(self, cameras):
        time_stamps = []
        cam_imgs = []

        # cur_time = rospy.get_time()
        cur_time = rospy.Time.now()
        for recorder in cameras:
            stamp, image = recorder.get_image()
            time_diff = (cur_time - stamp).to_sec()
            logging.getLogger('robot_logger').info("Current-Camera time difference {}".format(time_diff))

            if abs(time_diff) > 10 * self._obs_tol:    # no camera ping in half second => camera failure
                logging.getLogger('robot_logger').error("DeSYNC - no ping in more than {} seconds!".format(10 * self._obs_tol))
                # import pdb; pdb.set_trace()
                raise Image_Exception
            time_stamps.append(stamp)
            cam_imgs.append(image)

        return cam_imgs
    
    def reset_cameras(self): 
        print("doing reset")
        self._reset_counter += 1
        print(self._cameras_to_reset)
        for pub in self._cameras_to_reset: 
            pub.publish(True)

    @property
    def adim(self):
        """
        :return: Environment's action dimension
        """
        return self._adim

    @property
    def sdim(self):
        """
        :return: Environment's state dimension
        """
        return self._sdim

    @property
    def ncam(self):
        """
        Sawyer environment has ncam cameras
        """
        return len(self._cameras)

    @property
    def ndigit(self): 
        return len(self._digit_cameras)

    @property
    def camera_info(self):
        return self._camera_info
    
    @property
    def depth_camera_info(self):
        return self._depth_camera_info

    @property 
    def digit_camera_info(self): 
        return self._digit_camera_info

    @property
    def num_objects(self):
        """
        :return: Dummy value for num_objects (used in general_agent logic)
        """
        return 0
