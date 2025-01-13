import logging

import numpy as np
import rospy

from widowx_controller.srv import DisableController
from widowx_controller.srv import EnableController
from widowx_controller.srv import GetCartesianPose
from widowx_controller.srv import GetGripperDesiredState
from widowx_controller.srv import GetState
from widowx_controller.srv import GetVRButtons
from widowx_controller.srv import GotoNeutral
from widowx_controller.srv import MoveToEEP
from widowx_controller.srv import MoveToState
from widowx_controller.srv import OpenGripper
from widowx_controller.srv import SetGripperPosition
from widowx_envs.utils.exceptions import Environment_Exception

from widowx_controller.controller_base import RobotControllerBase

# TODO: abstract class for all controllers:
# with internal ROS init, thus higher level impl doesnt need to interact
# with ROS directly, middleware agnostic

class WidowX_VRContollerClient(RobotControllerBase):
    def __init__(self, print_debug=False):
        rospy.init_node("vr_controller_client")
        logger = logging.getLogger('robot_logger')
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        logger.setLevel(logging.DEBUG)

        log_level = logging.INFO
        if print_debug:
            log_level = logging.DEBUG
        ch = logging.StreamHandler()
        ch.setLevel(log_level)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    def move_to_neutral(self, duration=4):
        rospy.wait_for_service('go_to_neutral')
        try:
            goto_neutral = rospy.ServiceProxy('go_to_neutral', GotoNeutral)
            goto_neutral(duration)
        except rospy.ServiceException as e:
            print("Service call failed: %s" % e)

    def move_to_eep(self, target_pose, duration=1.5):
        rospy.wait_for_service('move_to_eep')
        try:
            service_func = rospy.ServiceProxy('move_to_eep', MoveToEEP)
            service_func(target_pose, duration)
        except rospy.ServiceException as e:
            print("Service call failed: %s" % e)

    def move_to_state(self, startpos, zangle, duration=1.5):
        rospy.wait_for_service('move_to_state')
        try:
            service_func = rospy.ServiceProxy('move_to_state', MoveToState)
            resp = service_func(startpos, zangle, duration)
            if not resp.success:
                raise Environment_Exception
        except rospy.ServiceException as e:
            print("Service call failed: %s" % e)


    def get_cartesian_pose(self, matrix=False):
        assert matrix
        rospy.wait_for_service('get_cartesian_pose')
        try:
            service_func = rospy.ServiceProxy('get_cartesian_pose', GetCartesianPose)
            return np.array(service_func().eep).reshape(4,4)
        except rospy.ServiceException as e:
            print("Service call failed: %s" % e)

    def get_state(self):
        rospy.wait_for_service('get_state')
        try:
            service_func = rospy.ServiceProxy('get_state', GetState)
            response = service_func()
            return np.array(response.joint_angles), np.array(response.joint_velocities), np.array(response.cartesian_pose)
        except rospy.ServiceException as e:
            print("Service call failed: %s" % e)

    def get_vr_buttons(self):
        rospy.wait_for_service('get_vr_buttons')
        try:
            service_func = rospy.ServiceProxy('get_vr_buttons', GetVRButtons)
            response = service_func()
            return {'handle': bool(response.handle), 'A': bool(response.a), 'B': bool(response.b), 'RJ': bool(response.rj)}
        except rospy.ServiceException as e:
            print("Service call failed: %s" % e)


    def enable_controller(self):
        rospy.wait_for_service('enable_controller')
        try:
            service_func = rospy.ServiceProxy('enable_controller', EnableController)
            service_func()
        except rospy.ServiceException as e:
            print("Service call failed: %s" % e)

    def disable_controller(self):
        rospy.wait_for_service('disable_controller')
        try:
            service_func = rospy.ServiceProxy('disable_controller', DisableController)
            service_func()
        except rospy.ServiceException as e:
            print("Service call failed: %s" % e)

    def get_gripper_desired_state(self):
        rospy.wait_for_service('get_gripper_desired_state')
        try:
            service_func = rospy.ServiceProxy('get_gripper_desired_state', GetGripperDesiredState)
            return service_func().des_pos
        except rospy.ServiceException as e:
            print("Service call failed: %s" % e)

    def open_gripper(self, wait=False):
        print('opening gripper')
        rospy.wait_for_service('open_gripper')
        try:
            service_func = rospy.ServiceProxy('open_gripper', OpenGripper)
            service_func()
        except rospy.ServiceException as e:
            print("Service call failed: %s" % e)

    def set_gripper_position(self, position):
        rospy.wait_for_service('set_gripper_position')
        try:
            service_func = rospy.ServiceProxy('set_gripper_position', SetGripperPosition)
            service_func(position)
        except rospy.ServiceException as e:
            print("Service call failed: %s" % e)
