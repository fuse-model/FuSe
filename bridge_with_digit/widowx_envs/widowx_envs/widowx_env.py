#! /usr/bin/python3

import pickle as pkl
import time
from widowx_envs.utils.transformation_utils import state2transform
import numpy as np
import math
from widowx_envs.base.robot_base_env import RobotBaseEnv
from widowx_envs.utils.exceptions import Environment_Exception
from widowx_envs.utils.utils import ask_confirm
from skimage.transform import resize

import os
from gym import spaces
import random
from widowx_envs.utils.exceptions import Environment_Exception

import rospy
from sensor_msgs.msg import Image
from multicam_server.topic_utils import IMTopic
from widowx_controller.widowx_controller import WidowX_Controller
import librosa

##############################################################################
MIC_SAMPLE_FREQ = 44100
MIC_OBS_LENGTH = 0.3 
NFFT = 256
HOP_LENGTH = 128
MEL_HOP_LENGTH = 104 # selected to make mel_spectrogram have dimension 128x128
N_MELS = 128
############

class WidowXEnv(RobotBaseEnv):
  
    def _default_hparams(self):
        robot_name = os.getenv('ROBONETV2_ARM')
        if robot_name is None:
            print('Environment variable ROBONETV2_ARM has to be set. \
                    Please define it based on \
                    https://github.com/Interbotix/interbotix_ros_manipulators/tree/main/interbotix_ros_xsarms')
            print('For instance in case of WidowX 250 Robot Arm 6DOF use:')
            print('echo "export ROBONETV2_ARM=wx250s" >> ~/.bashrc && source ~/.bashrc')
            raise RuntimeError
        default_dict = {
            'robot_name': robot_name,
            'randomize_initpos': 'full_area',
            'mode_rel': [True, True, True, True, True],
            'start_state': [0.3, 0.0, 0.15, 0, 0, 0, 1],
            'start_transform': None,
            'skip_move_to_neutral': False,
            'move_to_rand_start_freq': 1,
            'fix_zangle': False
        }
        parent_params = super(WidowXEnv, self)._default_hparams()
        parent_params.update(default_dict)
        return parent_params

    def reset(self, itraj=None):
        """
        Resets the environment and returns initial observation
        :return: obs dict (look at step(self, action) for documentation)
        """
        # self._controller.check_motor_status_and_reboot()
        self._controller.open_gripper(True)
        time.sleep(1.)
        if not self._hp.skip_move_to_neutral:
            # print("moving to neutral here")
            self._controller.move_to_neutral(duration=1.5)

        if itraj is None:
            print('moving to start state here ')
            self.move_to_startstate()
        else:
            if self._hp.move_to_rand_start_freq != -1:
                print('a')
                if itraj % self._hp.move_to_rand_start_freq == 0:
                    print('b')
                    self.move_to_startstate()

        self._reset_previous_qpos()
        obs = self.current_obs()
        return obs

    def move_to_startstate(self, start_state=None):
        if self._hp.start_state is not None or start_state is not None:
            if start_state is None:
                start_state = self._hp.start_state
                # start_state = pkl.load(open(self._hp.start_state + '/obs_dict.pkl', 'rb'))['state'][0]
            start_state = np.array(start_state)
            if start_state.shape[0] == 5:
                start_state = np.concatenate([start_state[:3], np.zeros(2), start_state[3:]])
            transform, _ = state2transform(start_state, self._controller.default_rot)
            assert isinstance(self._controller, WidowX_Controller)
            successful = False
            while not successful:
                try:
                    self._controller.move_to_eep(transform)
                    successful = True
                except Environment_Exception:
                    self.move_to_neutral()
        elif self._hp.start_transform is not None:
            path = self._hp.start_transform[0]
            tstep = self._hp.start_transform[1]
            transform = pkl.load(open(path + '/obs_dict.pkl', 'rb'))['eef_transform'][tstep]
            assert isinstance(self._controller, WidowX_Controller)
            successful = False
            while not successful:
                try:
                    self._controller.move_to_eep(transform)
                    successful = True
                except Environment_Exception:
                    self.move_to_neutral()
        else:
            if self._hp.randomize_initpos == 'restricted_space':
                startpos = np.random.uniform(self._low_bound[:3] + np.array([0, 0, 0.085]),
                                             self._high_bound[:3] - np.array([0, 0.07, 0.01]))
                # zangle = np.random.uniform(self._low_bound[3], self._high_bound[3])
                zangle = np.random.uniform(0.5*np.pi, -0.5*np.pi)  # np.pi is neutral!
            elif self._hp.randomize_initpos == 'full_area':
                startpos = np.random.uniform(self._low_bound[:3] + np.array([0, 0, 0.05]), self._high_bound[:3])
                zangle = np.random.uniform(self._low_bound[3], self._high_bound[3])
            elif self._hp.randomize_initpos == 'line':
                y_rand = np.random.uniform(self._low_bound[1] + 0.05, self._high_bound[1] - 0.04)
                z_rand = np.random.uniform(self._low_bound[2] + 0.05, self._high_bound[2])
                startpos = np.array([0.21, y_rand, z_rand])
                zangle = math.pi / 2
            else:
                raise NotImplementedError
            try:
                if self._hp.fix_zangle:
                    zangle = 0
                self._controller.move_to_state(startpos, zangle, duration=2)
                self._reset_previous_qpos()
            except Environment_Exception:
                self.move_to_startstate()  # retry with different sample position

    def start(self):
        self._controller.set_moving_time(self._hp.move_duration)

    def finish(self):
        pass

    def ask_confirmation(self, ):
        return ask_confirm("Was the trajectory okay? y/n")

##############################################################################

class RandomInit_WidowXEnv(WidowXEnv):
    def move_to_startstate(self):
        assert self._hp.start_transform is not None
        sampled_start_transform = random.choice(self._hp.start_transform)
        path = sampled_start_transform[0]
        tstep = sampled_start_transform[1]
        obs_dict = pkl.load(open(path + '/obs_dict.pkl', 'rb'))
        assert isinstance(self._controller, WidowX_Controller)
        # transform = obs_dict['eef_transform'][tstep]
        # self._controller.move_to_eep(transform)
        joint_angles = obs_dict['qpos'][tstep]
        self._controller.bot.arm.publish_positions(joint_angles, moving_time=1.5)
        # self._controller.wait_until_gripper_position_reached()
        # while self._controller.get_continuous_gripper_position() < 0.98:
        #     print(self._controller.get_continuous_gripper_position())
        #     self._controller._gripper.open()


##############################################################################

class VR_WidowX(WidowXEnv):
    def __init__(self, env_params=None, **kwargs):
        super(VR_WidowX, self).__init__(env_params, **kwargs)
        self.task_stage = 0

        from oculus_reader import OculusReader
        self.oculus_reader = OculusReader()

    def get_vr_buttons(self):
        poses, buttons = self.oculus_reader.get_transformations_and_buttons()
        if 'RG' in buttons:
            buttons['handle'] = buttons['RG']
        else:
            buttons['handle'] = False
        return buttons

    def _default_hparams(self):
        default_dict = {
            'num_task_stages': 1,
            'make_oculus_reader': True

        }
        parent_params = super(VR_WidowX, self)._default_hparams()
        parent_params.update(default_dict)
        return parent_params

    def step(self, action, get_obs_tstamp=None, blocking=False):
        """
        :param action:  endeffector velocities
        :return:  observations
        """
        obs = super(VR_WidowX, self).step(action, get_obs_tstamp, blocking)
        if self.get_vr_buttons()['B']:
            self.task_stage += 1
        return obs

    def current_obs(self):
        obs = super(VR_WidowX, self).current_obs()
        if self.task_stage == self._hp.num_task_stages:
            obs['env_done'] = True
        obs['task_stage'] = self.task_stage
        return obs

    def reset(self, itraj=None):
        self.task_stage = 0
        obs = super(VR_WidowX, self).reset(itraj=itraj)
        start_key = 'handle'
        print('waiting for {} button press to start recording. Press B to go to neutral, or A to take a picture of the goal object.'.format(start_key))
        buttons = self.get_vr_buttons()
        while not buttons[start_key]:
            buttons = self.get_vr_buttons()
            rospy.sleep(0.01)
            if 'B' in buttons and buttons['B']:
                self.move_to_neutral_low()
                # self.move_to_neutral()
                print("moved to neutral. waiting for {} button press to start recording.".format(start_key))
            elif 'A' in buttons and buttons['A']: 
                # self.move_to_grip()
                # print("moved to pre-grip. waiting for {} button press to start recording.".format(start_key))
                obs = self.current_obs() 
                self.target_img = obs['images'][0, ...][..., ::-1]
        return self.current_obs()

    def get_last_image(self): 
        print("Press A to take a picture of the collected object")
        buttons = self.get_vr_buttons()
        while not buttons['A']:
            buttons = self.get_vr_buttons()
            rospy.sleep(0.01)
        obs = self.current_obs()
        return obs['images']

    def ask_confirmation(self):
        print('current endeffector pos', self.get_full_state()[:3])
        print('current joint angles pos', self._controller.get_joint_angles())
        print('Was the trajectory okay? Press A to confirm and RJ to discard')
        buttons = self.get_vr_buttons()
        while not buttons['A'] and not buttons['RJ']:
            buttons = self.get_vr_buttons()
            rospy.sleep(0.01)

        if buttons['RJ']:
            print('trajectory discarded!')
            return False
        if buttons['A']:
            print('trajectory accepted!')
            return True

##############################################################################

class VR_WidowX_DAgger(VR_WidowX):
    def __init__(self, env_params=None, **kwargs):
        super(VR_WidowX_DAgger, self).__init__(env_params, **kwargs)

    def ask_confirmation(self):
        print('Was the trajectory okay? \
            Press A to save a successful trajectory, trigger to save an unsuccessful trajectory, and RJ to discard')
        buttons = self.get_vr_buttons()

        while buttons['A'] or buttons['RJ'] or buttons['RTr']:
            buttons = self.get_vr_buttons()
            rospy.sleep(0.01)

        while not buttons['A'] and not buttons['RJ'] and not buttons['RTr']:
            buttons = self.get_vr_buttons()
            rospy.sleep(0.01)

        if buttons['RJ']:
            print('trajectory discarded!')
            return False
        elif buttons['A']:
            print('successful trajectory accepted!')
            return 'Success'
        elif buttons['RTr']:
            print('unsuccessful trajectory accepted!')
            return 'Failure'

##############################################################################

class StateReachingWidowX(WidowXEnv):
    def __init__(self, env_params=None, fixed_goal=False):
        super(StateReachingWidowX, self).__init__(env_params)

        action_space = np.full(self.adim, 0.05)
        self.action_space = spaces.Box(-action_space, action_space, dtype=np.float64)
        self.observation_space = spaces.dict.Dict({
            "state": spaces.Box(-np.full(7, np.inf), np.full(7, np.inf), dtype=np.float64),
            "desired_goal": spaces.Box(-np.full(3, np.inf), np.full(3, np.inf), dtype=np.float64),
            "vector_to_goal": spaces.Box(-np.full(3, np.inf), np.full(3, np.inf), dtype=np.float64),
        })
        self.goal_threshold = 0.0175
        self.fixed_goal = fixed_goal
        self.goal_coord = None

    def _default_hparams(self):
        default_dict = {
            'camera_topics': [],
            'gripper_attached': 'custom',
            'skip_move_to_neutral': True,
            'action_space': '3trans',
            'fix_zangle': True,
            'override_workspace_boundaries': [[0.19, -0.08, 0.05, -1.57, 0], [0.31, 0.08, 0.055,  1.57, 0]]
        }
        parent_params = super(StateReachingWidowX, self)._default_hparams()
        parent_params.update(default_dict)
        return parent_params

    def current_obs(self):
        full_obs = super(StateReachingWidowX, self).current_obs()
        ee_coord = full_obs['full_state'][:3]
        vector_to_goal = self.goal_coord - ee_coord
        obs = {'vector_to_goal': vector_to_goal, 'state': self.get_full_state(),
               'joints': full_obs['qpos'], 'desired_goal': self.goal_coord, 'achieved_goal': ee_coord}
        return obs

    def step(self, action, get_obs_tstamp=None, blocking=True):
        obs = super(StateReachingWidowX, self).step(action)
        dist = np.linalg.norm(obs['vector_to_goal'])
        reward = -dist
        goal_reached = self.goal_reached(dist)
        return obs, reward, goal_reached, {}

    def goal_reached(self, distance):
        return distance < self.goal_threshold

    def reset(self):
        if self.fixed_goal:
            xyz = self._high_bound[:3] + self._low_bound[:3]
            xyz = xyz/2
        else:
            xyz = np.random.uniform(self._low_bound[:3], self._high_bound[:3])
        self.goal_coord = xyz
        obs = super(StateReachingWidowX, self).reset(itraj=0)
        return obs

##############################################################################

class ImageReachingWidowX(StateReachingWidowX):
    def __init__(self, env_params=None, publish_images=True, fixed_image_size=64, fixed_goal=False):

        super(ImageReachingWidowX, self).__init__(env_params, fixed_goal=fixed_goal)
        self.image_size = fixed_image_size

        action_space = np.full(self.adim, 0.05)
        self.action_space = spaces.Box(-action_space, action_space, dtype=np.float64)
        self.observation_space = spaces.dict.Dict({
            "image": spaces.Box(low=np.array([0]*self.image_size*self.image_size*3),
                                high=np.array([255]*self.image_size*self.image_size*3), dtype=np.uint8),
            "state": spaces.Box(-np.full(self.sdim, np.inf), np.full(self.sdim, np.inf), dtype=np.float64),
            "desired_goal": spaces.Box(-np.full(3, np.inf), np.full(3, np.inf), dtype=np.float64),
        })
        self.goal_coord = None
        self.goal_threshold = 0.02
        self.publish_images = publish_images
        if self.publish_images:
            from cv_bridge import CvBridge
            self.bridge = CvBridge()
            self.image_pub = rospy.Publisher("/robonetv2_image/image_raw", Image, queue_size=10)

    def _default_hparams(self):
        default_dict = {
            'camera_topics': [IMTopic('/camera0/image_raw')],
            'image_crop_xywh': None,  # can be a tuple like (0, 0, 100, 100)
            'transpose_image_to_chw': False,  # changes image to CHW format for use with pytorch
            'return_full_image': False,
        }
        parent_params = super(ImageReachingWidowX, self)._default_hparams()
        parent_params.update(default_dict)
        return parent_params

    @staticmethod
    def _to_float32_flat_image(image):
        return np.float32(image.flatten()) / 255.0

    def _get_processed_image(self, image=None):
        if image is None:
            image = super(StateReachingWidowX, self).current_obs()['images'][0]

        if self._hp['image_crop_xywh'] is None:
            trimmed_image = image
        else:
            x, y, w, h = self._hp['image_crop_xywh']
            trimmed_image = image[x:x+w, y:y+h]

        downsampled_trimmed_image = resize(trimmed_image, (self.image_size, self.image_size),
                                           anti_aliasing=True, preserve_range=True).astype(np.uint8)
        if self._hp['transpose_image_to_chw']:
            downsampled_trimmed_image = np.transpose(downsampled_trimmed_image, (2, 0, 1))
        return self._to_float32_flat_image(downsampled_trimmed_image)

    def step(self, action):
        obs = super(StateReachingWidowX, self).step(action)
        try:
            vector_to_goal = obs['vector_to_goal']
        except:
            raise NotImplementedError('Implemented only for the dict observation space')
        dist = np.linalg.norm(vector_to_goal)
        reward = -dist
        goal_reached = self.goal_reached(dist)
        if goal_reached:
            print('Done!')
        if self.publish_images:
            # try:
            self._publish_image(obs['image'])
            # except:
            #     raise NotImplementedError('Implemented only for the dict observation space')
        return obs, reward, goal_reached, {}

    def _publish_image(self, image):
        try:
            if self._hp['transpose_image_to_chw']:
                cv_image = np.uint8(image.reshape((3, self.image_size, self.image_size))*255.0)
                cv_image = np.transpose(cv_image, (1, 2, 0))
            else:
                cv_image = np.uint8(image.reshape((self.image_size, self.image_size, 3))*255.0)
            imgmsg = self.bridge.cv2_to_imgmsg(cv_image, 'rgb8')
            self.image_pub.publish(imgmsg)
        except Exception as e:
            print(e)

    def current_obs(self):
        full_obs = super(StateReachingWidowX, self).current_obs()
        image = full_obs['images'][0]
        ee_coord = full_obs['full_state'][:3]
        vector_to_goal = self.goal_coord - ee_coord
        processed_image = self._get_processed_image(image)

        obs = {'image': processed_image, 'state': self.get_full_state(), 'vector_to_goal': vector_to_goal,
               'joints': full_obs['qpos'], 'desired_goal': self.goal_coord, 'achieved_goal': ee_coord}
        if self._hp.return_full_image:
            obs['full_image'] = image
        # obs['gripper'] = full_obs['state'][-1]  # this dimension is not being updated for now
        return obs


##############################################################################

class BridgeDataRailRLPrivateWidowX(WidowXEnv):
    def __init__(self, env_params=None, reward_function=None, task_id=None, num_tasks=None, fixed_image_size=128,
                 control_viewpoint=0, # used for reward function
                 **kwargs
                 ):

        super().__init__(env_params)
        self.image_size = fixed_image_size
        self.task_id = task_id
        self.num_tasks = num_tasks

        self.observation_space = spaces.dict.Dict({
            "image": spaces.Box(low=np.array([0]*self.image_size*self.image_size*3),
                                high=np.array([255]*self.image_size*self.image_size*3), dtype=np.uint8),
            "state": spaces.Box(-np.full(self.sdim, np.inf), np.full(self.sdim, np.inf), dtype=np.float64),
            "desired_goal": spaces.Box(-np.full(3, np.inf), np.full(3, np.inf), dtype=np.float64),
        })
        self.move_except = False

        self.reward_function = reward_function
        self.control_viewpoint = control_viewpoint

    def _default_hparams(self):
        from multicam_server.topic_utils import IMTopic
        default_dict = {
            'gripper_attached': 'custom',
            'skip_move_to_neutral': True,
            'camera_topics': [IMTopic('/cam0/image_raw')],
            'image_crop_xywh': None,  # can be a tuple like (0, 0, 100, 100)
            # 'camera_topics': [IMTopic('/cam0/image_raw'), IMTopic('/cam1/image_raw'), IMTopic('/cam2/image_raw')],
        }
        parent_params = super()._default_hparams()
        parent_params.update(default_dict)
        return parent_params

    def reset(self, itraj=None):
        self.move_except = False
        return super().reset(itraj)
        # # TODO: (YL) test this, else just move bot back neutral
        self._controller.open_gripper(True)
        self._controller.move_to_neutral(duration=1.5)
        time.sleep(1.)
        self._reset_previous_qpos()
        return self.current_obs()

    @staticmethod
    def _to_float32_flat_image(image):
        return np.float32(image.flatten()) / 255.0

    def _get_processed_image(self, image=None):
        from skimage.transform import resize
        downsampled_trimmed_image = resize(image, (self.image_size, self.image_size), anti_aliasing=True, preserve_range=True).astype(np.uint8)
        downsampled_trimmed_image = np.transpose(downsampled_trimmed_image, (2, 0, 1))
        return self._to_float32_flat_image(downsampled_trimmed_image)

    def step(self, action, get_obs_tstamp=None, blocking=True):
        obs = super().step(action, get_obs_tstamp, blocking)
        return obs, None, obs['env_done'], {}

    def current_obs(self):
        t0 = time.time()
        full_obs = super().current_obs()
        processed_images = np.stack([self._get_processed_image(im) for im in full_obs['images']], axis=0)

        obs = {'image': processed_images, 'state': self.get_full_state(),
               'joints': full_obs['qpos'], 'env_done': full_obs['env_done'], 'full_obs': full_obs}
        if full_obs['env_done']:
            obs['terminals'] = 1
        else:
            obs['terminals'] = 0
        if self.move_except:
            obs['env_done'] = 1
        obs['full_image'] = full_obs['images']
        obs['t_get_obs'] = time.time() - t0
        return obs


class BridgeDataDigitWidowX(WidowXEnv):
    def __init__(self, env_params=None, reward_function=None, task_id=None, num_tasks=None, image_size=256,
                 wrist_size=128, 
                 dig_size=128, 
                 control_viewpoint=0, # used for reward function
                 **kwargs
                 ):

        super().__init__(env_params)
        self.image_size = image_size
        self.wrist_size = wrist_size
        self.dig_size = dig_size 
        self.digit_l_background = self.digit_r_background = None 
        self.orig_back_l = self.orig_back_r = None 

        self.task_id = task_id
        self.num_tasks = num_tasks

        # TODO: update this
        self.observation_space = spaces.dict.Dict({
            "image": spaces.Box(low=np.array([0]*self.image_size*self.image_size*3),
                                high=np.array([255]*self.image_size*self.image_size*3), dtype=np.uint8),
            "state": spaces.Box(-np.full(self.sdim, np.inf), np.full(self.sdim, np.inf), dtype=np.float64),
            "desired_goal": spaces.Box(-np.full(3, np.inf), np.full(3, np.inf), dtype=np.float64),
        })

        self.move_except = False
        self.reward_function = reward_function
        self.control_viewpoint = control_viewpoint

    def _default_hparams(self):
        from multicam_server.topic_utils import IMTopic
        default_dict = {
            'gripper_attached': 'custom',
            'skip_move_to_neutral': True,
            'camera_topics': [IMTopic('/cam0/image_raw')],
            'image_crop_xywh': None,  # can be a tuple like (0, 0, 100, 100)
            # 'camera_topics': [IMTopic('/cam0/image_raw'), IMTopic('/cam1/image_raw'), IMTopic('/cam2/image_raw')],
        }
        parent_params = super()._default_hparams()
        parent_params.update(default_dict)
        return parent_params

    def reset(self, itraj=None):
        self.move_except = False
        full_obs = super().reset(itraj)
        if "digit_images" in full_obs:
            assert len(full_obs['digit_images']) == 2, "You must connect and use two DIGITs."
            digit_l, digit_r = full_obs['digit_images']
            processed_digit_l = np.stack([ 
                self._get_processed_image(image=digit_l, size=self.dig_size)
            ], axis=0)
            processed_digit_r = np.stack([ 
                self._get_processed_image(image=digit_r, size=self.dig_size)
            ], axis=0)
            self.digit_l_background = processed_digit_l
            self.digit_r_background = processed_digit_r
            self.orig_back_l = digit_l 
            self.orig_back_r = digit_r
        else: 
            self.digit_l_background = self.digit_r_background = None 
        return full_obs 
        # # TODO: (YL) test this, else just move bot back neutral
        self._controller.open_gripper(True)
        self._controller.move_to_neutral(duration=1.5)
        time.sleep(1.)
        self._reset_previous_qpos()
        return self.current_obs()

    @staticmethod
    def _to_float32_flat_image(image):
        return np.float32(image.flatten()) / 255.0

    def _get_processed_image(self, image=None, size=None):
        if size is None: 
            size = self.image_size

        # from skimage.transform import resize
        # downsampled_trimmed_image = resize(image, (size, size), anti_aliasing=True, preserve_range=True).astype(np.uint8)
        downsampled_trimmed_image = np.transpose(image, (2, 0, 1))
        return self._to_float32_flat_image(downsampled_trimmed_image)

    def step(self, action, get_obs_tstamp=None, blocking=True):
        obs = super().step(action, get_obs_tstamp, blocking)
        return obs, None, obs['env_done'], {}

    def current_obs(self):
        t0 = time.time()
        full_obs = super().current_obs()

        if len(full_obs['images']) == 1: # primary only
            image_primary = full_obs['images'][0]
            image_wrist = None
        elif len(full_obs['images']) == 2: # primary + wrist 
            image_primary = full_obs['images'][0]
            image_wrist = full_obs['images'][1]

        else: 
            num_imgs = len(full_obs['images'])

            raise ValueError(f'full_obs has {num_imgs} images.')
        
        
        processed_primary = np.stack([
            self._get_processed_image(image=image_primary, size=self.image_size)
        ], axis=0)

        processed_wrist = None 
        if image_wrist is None: 
            processed_wrist = None 
        else: 
            processed_wrist = np.stack([ 
                self._get_processed_image(image=image_wrist, size=self.wrist_size)
            ], axis=0)
            
        if "digit_images" in full_obs:
            assert len(full_obs['digit_images']) == 2, "You must connect and use two DIGITs."
            digit_l, digit_r = full_obs['digit_images']
            if self.orig_back_l is None: 
                self.orig_back_l = digit_l.copy() 
                self.orig_back_r = digit_r.copy() 
            digit_l -= self.orig_back_l 
            digit_r -= self.orig_back_r
            processed_digit_l = np.stack([ 
                self._get_processed_image(image=digit_l, size=self.dig_size)
            ], axis=0)
            processed_digit_r = np.stack([ 
                self._get_processed_image(image=digit_r, size=self.dig_size)
            ], axis=0)
            if self.digit_l_background is None: 
                self.digit_l_background = processed_digit_l.copy() 
                self.digit_r_background = processed_digit_r.copy() 
            
                
        else: 
            processed_digit_l = processed_digit_r = None 

        obs = {
            'image_0': processed_primary, 
            'image_1': processed_wrist,
            'digit_l': processed_digit_l, 
            'digit_r': processed_digit_r, 
            'background_l': self.digit_l_background, 
            'background_r': self.digit_r_background, 
            'state': self.get_full_state(),
            'joints': full_obs['qpos'],
            'env_done': full_obs['env_done'],
            'full_obs': full_obs
        }

        if "imu" in full_obs: 
            obs['imu'] = full_obs['imu']
        if "mic" in full_obs: 
            mic_data = full_obs["mic"]
            spectrogram_data = np.abs(librosa.stft(mic_data, n_fft=NFFT, hop_length=HOP_LENGTH))
            spectrogram_nonfft = np.abs(librosa.stft(mic_data, hop_length=MEL_HOP_LENGTH))
            mel_spectro = librosa.feature.melspectrogram(S=spectrogram_nonfft**2, sr=MIC_SAMPLE_FREQ, n_mels=128)
            mel_spectro = librosa.power_to_db(mel_spectro)
            obs['mel_spectro'] = mel_spectro.flatten()[None, ...]
    

        if full_obs['env_done']:
            obs['terminals'] = 1
        else:
            obs['terminals'] = 0

        if self.move_except:
            obs['env_done'] = 1
            
        obs['full_image'] = full_obs['images']
        obs['t_get_obs'] = time.time() - t0
        tfinal = time.time()
        print("DELTA_T observation:     ", tfinal - t0)
        return obs



# class BridgeDataDigitWidowX(WidowXEnv):
#     def __init__(self, env_params=None, reward_function=None, task_id=None, num_tasks=None, fixed_image_size=128,
#                  control_viewpoint=0, # used for reward function
#                  **kwargs
#                  ):

#         super().__init__(env_params)
        
#         self.task_id = task_id
#         self.num_tasks = num_tasks

#         self.move_except = False

#         self.reward_function = reward_function
#         self.control_viewpoint = control_viewpoint

#     def _default_hparams(self):
        

#     def reset(self, itraj=None):
#         self.move_except = False
#         return super().reset(itraj)
#         # # TODO: (YL) test this, else just move bot back neutral
#         self._controller.open_gripper(True)
#         self._controller.move_to_neutral(duration=1.5)
#         time.sleep(1.)
#         self._reset_previous_qpos()
#         return self.current_obs()


#     def step(self, action, get_obs_tstamp=None, blocking=True):
#         obs = super().step(action, get_obs_tstamp, blocking)
#         return obs, None, obs['env_done'], {}


#     def current_obs(self):
#         t0 = time.time()
#         full_obs = super().current_obs()

#         if self._previous_target_qpos is not None: 
#             target, _ = self.get_target_state()
#         else: 
#             target = full_obs['transform_pose']
#         obs = {
#             'qpos': full_obs['qpos'], 
#             'images': full_obs['images'], 
#             'state': full_obs['full_state'],
#             'xyz': full_obs['xyz'],
#             'curr_transform': full_obs['transform_pose'],
#             'target': target, 
#             'eef_transform': full_obs['eef_transform'] 
#         }
#         if 'digit_images' in full_obs: 
#             obs['digit_images'] = full_obs['digit_images']
#         return obs

##############################################################################

class BridgeDataRailRLPrivateVRWidowX(BridgeDataRailRLPrivateWidowX, VR_WidowX_DAgger):
    def __init__(self, env_params=None, reward_function=None, task_id=None, num_tasks=None, fixed_image_size=128):
        super().__init__(env_params=env_params,
                         reward_function=reward_function,
                         task_id=task_id,
                         num_tasks=num_tasks,
                         fixed_image_size=fixed_image_size)

##############################################################################

class FinetuningBridgeDataWidowX(BridgeDataRailRLPrivateWidowX):
    def __init__(self, env_params=None, reward_function=None, task_id=None, num_tasks=None, fixed_image_size=128):
        super().__init__(env_params=env_params,
                         reward_function=reward_function,
                         task_id=task_id,
                         num_tasks=num_tasks,
                         fixed_image_size=fixed_image_size)
        
        self.step_duration = 0.2
        self.last_tstep = time.time()
        
    def get_contextual_diagnostics(self, a, b):
        return {}
    
    def get_image(self):
        return self.current_obs()['image']
    
    def reset_previous_qpos(self):
        return self._reset_previous_qpos()

    def get_obs(self):
        return self.current_obs()

    def set_last_tstep(self):
        self.last_tstep = time.time()
        return self.last_tstep

    def step(self, action):
        obs, rew, done, info = super().step(action, self.last_tstep+self.step_duration, blocking=False)
        obs = {
            'image_observation': obs['image'][0],
            'full_image_observation': obs['full_image'][0],
            'state': obs['state'],
        }
        return (obs, rew, done, info)

    def reset(self):
        obs = super().reset()
        obs = {
            'image_observation': obs['image'][0],
            'full_image_observation': obs['full_image'][0],
            'state': obs['state'],
        } 
        return obs

##############################################################################

if __name__ == '__main__':
    env = StateReachingWidowX()
    env.move_to_neutral()
    for i in range(5):
        env.reset()
        env.step(np.zeros(env.action_space.shape))
