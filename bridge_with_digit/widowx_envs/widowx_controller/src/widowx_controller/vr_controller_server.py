#! /usr/bin/python3

import rospy
import time
from oculus_reader import OculusReader
from widowx_controller.widowx_controller import WidowX_Controller
from widowx_envs.control_loops import Environment_Exception
import widowx_envs.utils.transformation_utils as tr
from pyquaternion import Quaternion
import numpy as np

from std_msgs.msg import Float64

from widowx_controller.srv import GotoNeutral, GotoNeutralResponse
from widowx_controller.srv import MoveToEEP, MoveToEEPResponse
from widowx_controller.srv import MoveToState, MoveToStateResponse
from widowx_controller.srv import GetCartesianPose, GetCartesianPoseResponse
from widowx_controller.srv import GetState, GetStateResponse
from widowx_controller.srv import GetVRButtons, GetVRButtonsResponse
from widowx_controller.srv import EnableController, EnableControllerResponse
from widowx_controller.srv import DisableController, DisableControllerResponse

from widowx_controller.widowx_controller import publish_transform

##############################################################################

class VR_WidowX_ControllerServer(WidowX_Controller):
    def __init__(self, grasp_mode='continuous', *args, **kwargs):
        super(VR_WidowX_ControllerServer, self).__init__(*args, **kwargs)

        self.reader = OculusReader()

        self._moving_time = 0.05
        self.prev_handle_press = False
        self.reference_vr_transform = None
        self.reference_robot_transform = None

        self.do_reset = False
        self.task_stage = 0
        self.num_task_stages = 1e9
        self.last_update_time = time.time()

        self._control_loop_active = True
        rospy.Service('go_to_neutral', GotoNeutral, self.goto_neutral_service)
        rospy.Service('move_to_eep', MoveToEEP, self.move_to_eep_service)
        rospy.Service('move_to_state', MoveToState, self.move_to_state_service)
        rospy.Service('get_cartesian_pose', GetCartesianPose, self.get_cartesian_pose_service)
        rospy.Service('get_state', GetState, self.get_state_service)
        rospy.Service('get_vr_buttons', GetVRButtons, self.get_vr_buttons_service)
        rospy.Service('enable_controller', EnableController, self.enable_controller_service)
        rospy.Service('disable_controller', DisableController, self.disable_controller_service)

        self.pub_gripper_command = rospy.Publisher("/gripper_despos", Float64, queue_size=3)
        self.grasp_mode = grasp_mode

        self.last_pressed_times = {}

    def _init_gripper(self, gripper_attached, gripper_params):
        pass

    def goto_neutral_service(self, req):
        print('moving to neutral: seconds:', req.duration)
        self._control_loop_active = False
        self.move_to_neutral(req.duration)
        self._control_loop_active = True
        return GotoNeutralResponse()

    def move_to_eep_service(self, req):
        self._control_loop_active = False
        des_transform = np.array(req.des_eep).reshape(4,4)
        self.move_to_eep(des_transform, req.duration)
        self._control_loop_active = True
        return MoveToEEPResponse()

    def move_to_state_service(self, req):
        self._control_loop_active = False
        try:
            self.move_to_state(req.target_xyz, req.target_zangle, req.duration)
            success = True
        except Environment_Exception:
            success = False
        self._control_loop_active = True
        return MoveToStateResponse(success)

    def get_cartesian_pose_service(self, req):
        pose = self.get_cartesian_pose(matrix=True)
        return GetCartesianPoseResponse(pose.flatten())

    def get_state_service(self, req):
        joint_angles, joint_velocities, cartesian_pose = self.get_state()
        return GetStateResponse(joint_angles, joint_velocities, cartesian_pose)

    def get_vr_buttons_service(self, req):
        def check_press(key):
            if key in self.last_pressed_times:
                if time.time() - self.last_pressed_times[key] < 0.5:
                    return True
            return False
        resp = GetVRButtonsResponse(int(check_press('RG')), int(check_press('A')), int(check_press('B')), int(check_press('RJ')))
        return resp

    def enable_controller_service(self, req):
        self._control_loop_active = True
        return EnableControllerResponse()

    def disable_controller_service(self, req):
        self._control_loop_active = False
        return DisableControllerResponse()

    def get_pose_and_button(self):
        poses, buttons = self.reader.get_transformations_and_buttons()

        # store in dict the times buttons were pressed last.
        for key, value in buttons.items():
            if not isinstance(value, tuple):
                if value:
                    self.last_pressed_times[key] = time.time()

        if 'r' not in poses:
            return None, None, None, None
        return poses['r'], buttons['RTr'], buttons['rightTrig'][0], buttons['RG']

    def set_gripper_position(self, position):
        self.pub_gripper_command.publish(Float64(position))

    def update_robot_cmds(self, event):
        # print("time update cmds", time.time() - self.last_update_time)
        self.last_update_time = time.time()
        t1 = time.time()
        current_vr_transform, trigger, trigger_continuous, handle_button = self.get_pose_and_button()
        if current_vr_transform is None:
            return
        elif not self._control_loop_active:
            self.prev_handle_press = False
            return
        else:
            if not self.prev_handle_press and handle_button:
                print("resetting reference pose")
                self.reference_vr_transform = self.oculus_to_robot(current_vr_transform)
                self.initial_vr_offset = tr.RpToTrans(np.eye(3), self.reference_vr_transform[:3, 3])
                self.reference_vr_transform = tr.TransInv(self.initial_vr_offset).dot(self.reference_vr_transform)  ##

                self.reference_robot_transform = self.get_cartesian_pose(matrix=True)
                self.bot.arm.set_trajectory_time(moving_time=self._moving_time, accel_time=self._moving_time * 0.5)

            if not handle_button:
                self.reference_vr_transform = None
                self.reference_robot_transform = self.get_cartesian_pose(matrix=True)
                self.prev_handle_press = False
                return
        self.prev_handle_press = True

        print('gripper set point', 1 - trigger_continuous)
        self.set_gripper_position(1 - trigger_continuous)

        current_vr_transform = self.oculus_to_robot(current_vr_transform)
        current_vr_transform = tr.TransInv(self.initial_vr_offset).dot(current_vr_transform)  ##

        publish_transform(current_vr_transform, 'currentvr_robotsystem')
        publish_transform(self.reference_vr_transform, 'reference_vr_transform')

        delta_vr_transform = current_vr_transform.dot(tr.TransInv(self.reference_vr_transform))
        publish_transform(self.reference_robot_transform, 'reference_robot_transform')

        M_rob, v_rob = tr.TransToRp(self.reference_robot_transform)
        M_delta, v_delta = tr.TransToRp(delta_vr_transform)
        new_robot_transform = tr.RpToTrans(M_delta.dot(M_rob), v_rob + v_delta)
        # new_robot_transform = delta_vr_transform.dot(self.reference_robot_transform)

        publish_transform(new_robot_transform, 'des_robot_transform')

        delta_translation_norm = np.linalg.norm(self.get_cartesian_pose(matrix=True)[:3, 3] - new_robot_transform[:3, 3])
        # if delta_translation_norm > 0.15:
        if delta_translation_norm > 0.2:
            print('delta transform norm, too large: ', delta_translation_norm)
            import pdb; pdb.set_trace()

        try:
            tset = time.time()
            solution, success = self.bot.arm.set_ee_pose_matrix_fast(new_robot_transform, custom_guess=self.get_joint_angles())
            print("time for setting pos", time.time() - tset)
        except rospy.service.ServiceException:
            print('stuck during move')
            import pdb;
            pdb.set_trace()
            self.move_to_neutral()

        loop_time = time.time() - t1
        print("loop time", loop_time)
        if loop_time > 0.02:
            print('Warning: Control loop is slow!')

    def oculus_to_robot(self, current_vr_transform):
        current_vr_transform = tr.RpToTrans(Quaternion(axis=[0, 0, 1], angle=-np.pi / 2).rotation_matrix,
                                            np.zeros(3)).dot(
            tr.RpToTrans(Quaternion(axis=[1, 0, 0], angle=np.pi / 2).rotation_matrix, np.zeros(3))).dot(
            current_vr_transform)
        return current_vr_transform


def run():
    controller = VR_WidowX_ControllerServer(robot_name='wx250s', print_debug=True,
                                            gripper_attached='custom',
                                            enable_rotation='6dof')
    controller.set_moving_time(1)
    controller.move_to_neutral(duration=3)
    controller.set_moving_time(controller._moving_time)
    while not rospy.is_shutdown():
        controller.update_robot_cmds(None)


if __name__ == '__main__':
    run()
