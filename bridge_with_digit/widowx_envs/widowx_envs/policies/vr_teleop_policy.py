#! /usr/bin/python3
    
from widowx_envs.policies.policy import Policy
from widowx_envs.utils.utils import AttrDict
from widowx_envs.control_loops import Environment_Exception
import widowx_envs.utils.transformation_utils as tr

import numpy as np
import time

from pyquaternion import Quaternion
from widowx_controller.widowx_controller import publish_transform

##############################################################################

class VRTeleopPolicy(Policy):
    def __init__(self, ag_params, policyparams):

        """ Computes actions from states/observations. """
        self._hp = self._default_hparams()
        self._override_defaults(policyparams)

        self.last_pressed_times = {}
        self.env = ag_params.env_handle

        self.reader = self.env.oculus_reader
        # self.prev_vr_transform = None
        self.action_space = self.env._hp.action_mode

        self.prev_handle_press = False
        self.reference_vr_transform = None
        self.reference_robot_transform = None
        self.internal_counter = 0
        self.internal_counter_default_policy = 0

    def _default_hparams(self):
        dict = AttrDict(
            load_file="",
            type=None,
            policy_T=None,
        )
        default_dict = super(Policy, self)._default_hparams()
        default_dict.update(dict)
        return default_dict


    def get_pose_and_button(self):
        poses, buttons = self.reader.get_transformations_and_buttons()
        if poses == {}:
            return None, None, None, None
        return poses['r'], buttons['RTr'], buttons['rightTrig'][0], buttons['RG']


    def act_use_fixed_reference(self, t, i_tr, images, task_id):
        # print("time update cmds", time.time() - self.last_update_time)
        self.last_update_time = time.time()
        t1 = time.time()
        current_vr_transform, trigger, trigger_continuous, handle_button = self.get_pose_and_button()
        if current_vr_transform is None:
            return self.get_default_action(t, i_tr, images, task_id)
        else:
            if not self.prev_handle_press and handle_button:
                print("resetting reference pose")
                self.internal_counter_default_policy = 0
                self.reference_vr_transform = self.oculus_to_robot(current_vr_transform)
                self.initial_vr_offset = tr.RpToTrans(np.eye(3), self.reference_vr_transform[:3, 3])
                self.reference_vr_transform = tr.TransInv(self.initial_vr_offset).dot(self.reference_vr_transform)  ##

                self.reference_robot_transform, _ = self.env.get_target_state()
                if self.action_space == '3trans1rot':
                    self.reference_robot_transform = self.zero_out_pitchroll(self.reference_robot_transform)
                self.prev_commanded_transform = self.reference_robot_transform

            if not handle_button:
                self.internal_counter = 0
                self.internal_counter_default_policy += 1
                self.reference_vr_transform = None
                self.reference_robot_transform, _ = self.env.get_target_state()
                self.prev_handle_press = False
                if self.action_space == '3trans1rot':
                    self.reference_robot_transform = self.zero_out_pitchroll(self.reference_robot_transform)
                self.prev_commanded_transform = self.reference_robot_transform
                return self.get_default_action(t, i_tr, images, task_id)
        self.prev_handle_press = True
        self.internal_counter += 1

        current_vr_transform = self.oculus_to_robot(current_vr_transform)
        current_vr_transform = tr.TransInv(self.initial_vr_offset).dot(current_vr_transform)  ##

        publish_transform(current_vr_transform, 'currentvr_robotsystem')
        delta_vr_transform = current_vr_transform.dot(tr.TransInv(self.reference_vr_transform))

        publish_transform(self.reference_robot_transform, 'reference_robot_transform')
        M_rob, p_rob = tr.TransToRp(self.reference_robot_transform)
        M_delta, p_delta = tr.TransToRp(delta_vr_transform)
        new_robot_transform = tr.RpToTrans(M_delta.dot(M_rob), p_rob + p_delta)

        if self.action_space == '3trans1rot':
            new_robot_transform = self.zero_out_pitchroll(new_robot_transform)
        if self.action_space == '3trans':
            new_robot_transform = self.zero_out_yawpitchroll(new_robot_transform)
        publish_transform(new_robot_transform, 'des_robot_transform')

        prev_target_pos, _ = self.env.get_target_state()
        delta_robot_transform = new_robot_transform.dot(tr.TransInv(prev_target_pos))
        publish_transform(delta_robot_transform, 'delta_robot_transform')
        self.prev_commanded_transform = new_robot_transform

        des_gripper_position = (1 - trigger_continuous)
        actions = tr.transform2action_local(delta_robot_transform, des_gripper_position, self.env._controller.get_cartesian_pose()[:3])

        if self.env._hp.action_mode == '3trans1rot':
            actions = np.concatenate([actions[:3], np.array([actions[5]]), np.array([des_gripper_position])])  # only use the yaw rotation
        if self.env._hp.action_mode == '3trans':
            actions = np.concatenate([actions[:3], np.array([des_gripper_position])])  # only use the yaw rotation

        if np.linalg.norm(actions[:3]) > 0.5:
            print('delta transform too large!')
            print('Press c and enter to continue')
            import pdb; pdb.set_trace()
            raise Environment_Exception

        output = {'actions': actions, 'new_robot_transform':new_robot_transform, 'delta_robot_transform': delta_robot_transform, 'policy_type': 'VRTeleopPolicy'}

        if self._hp.policy_T and self.internal_counter >= self._hp.policy_T:
            output['done'] = True

        return output

    def act(self, t=None, i_tr=None, images=None, task_id=None):
        return self.act_use_fixed_reference(t, i_tr, images, task_id)
    
    def get_action(self, obs_np, task_id_vec=None):
        dict = self.act(images=obs_np, task_id=task_id_vec)
        return dict['actions'], {'policy_type': dict['policy_type']}

    def get_default_action(self, t, i_tr, images, task_id):
        return self.get_zero_action()

    def get_zero_action(self):
        if self.env._hp.action_mode == '3trans3rot':
            actions = np.concatenate([np.zeros(6), np.array([1])])
        elif self.env._hp.action_mode == '3trans1rot':
            actions = np.concatenate([np.zeros(4), np.array([1])])
        elif self.env._hp.action_mode == '3trans':
            actions = np.concatenate([np.zeros(3), np.array([1])])
        else:
            raise NotImplementedError
        return {'actions': actions, 'new_robot_transform':np.eye(4), 'delta_robot_transform': np.eye(4), 'policy_type': 'VRTeleopPolicy'}

    def zero_out_pitchroll(self, new_robot_transform):
        rot, xyz = tr.TransToRp(new_robot_transform)
        euler = tr.rotationMatrixToEulerAngles(rot.dot(self.env._controller.default_rot.transpose()), check_error_thresh=1e-5)
        euler[:2] = np.zeros(2)  # zeroing out pitch roll
        new_rot = tr.eulerAnglesToRotationMatrix(euler).dot(self.env._controller.default_rot)
        new_robot_transform = tr.RpToTrans(new_rot, xyz)
        return new_robot_transform

    def zero_out_yawpitchroll(self, new_robot_transform):
        rot, xyz = tr.TransToRp(new_robot_transform)
        euler = tr.rotationMatrixToEulerAngles(rot.dot(self.env._controller.default_rot.transpose()), check_error_thresh=1e-5)
        euler = np.zeros(3)  # zeroing out yaw pitch roll
        new_rot = tr.eulerAnglesToRotationMatrix(euler).dot(self.env._controller.default_rot)
        new_robot_transform = tr.RpToTrans(new_rot, xyz)
        return new_robot_transform

    def oculus_to_robot(self, current_vr_transform):
        current_vr_transform = tr.RpToTrans(Quaternion(axis=[0, 0, 1], angle=np.pi / 2).rotation_matrix,
                                            np.zeros(3)).dot(
            tr.RpToTrans(Quaternion(axis=[1, 0, 0], angle=np.pi / 2).rotation_matrix, np.zeros(3))).dot(
            current_vr_transform)
        return current_vr_transform

    def reset(self):
        self.internal_counter = 0
        self.internal_counter_default_policy = 0
        self.prev_vr_transform = None  # used for self.act_use_deltas only

        # used for act_use_fixed_reference only:
        self.prev_handle_press = False
        self.reference_vr_transform = None
        self.reference_robot_transform = None
