#! /usr/bin/python3

import numpy as np
import rospy
from pyquaternion import Quaternion
from sensor_msgs.msg import JointState
from geometry_msgs.msg import TransformStamped

from threading import Lock
import logging
import os
import time

import tf2_ros
from transformations import quaternion_from_matrix

from interbotix_xs_modules.arm import InterbotixArmXSInterface, InterbotixArmXSInterface, \
    InterbotixRobotXSCore, InterbotixGripperXSInterface

import modern_robotics as mr
try:
    # older version of interbotix sdk
    from interbotix_xs_sdk.msg import JointGroupCommand
except:
    # newer version of interbotix sdk
    from interbotix_xs_msgs.msg import JointGroupCommand

from widowx_envs.utils.exceptions import Environment_Exception
from modern_robotics.core import JacobianSpace, Adjoint, MatrixLog6, se3ToVec, TransInv, FKinSpace
import widowx_envs.utils.transformation_utils as tr

from widowx_controller.custom_gripper_controller import GripperController
from widowx_controller.controller_base import RobotControllerBase

##############################################################################

def publish_transform(transform, name, parent_name='wx250s/base_link'):
    """TODO(YL): it's bad to reinit the broadcaster every time, improve this"""
    translation = transform[:3, 3]

    br = tf2_ros.TransformBroadcaster()
    t = TransformStamped()

    t.header.stamp = rospy.Time.now()
    t.header.frame_id = parent_name
    t.child_frame_id = name
    t.transform.translation.x = translation[0]
    t.transform.translation.y = translation[1]
    t.transform.translation.z = translation[2]

    quat = quaternion_from_matrix(transform)
    t.transform.rotation.w = quat[0]
    t.transform.rotation.x = quat[1]
    t.transform.rotation.y = quat[2]
    t.transform.rotation.z = quat[3]

    # print('publish transofrm', name)
    br.sendTransform(t)

def compute_joint_velocities_from_cartesian(Slist, M, T, thetalist_current):
    """Computes inverse kinematics in the space frame for an open chain robot

    :param Slist: The joint screw axes in the space frame when the
                  manipulator is at the home position, in the format of a
                  matrix with axes as the columns
    :param M: The home configuration of the end-effector
    :param T: The desired end-effector configuration Tsd
    :param thetalist_current: An initial guess of joint angles that are close to
                       satisfying Tsd
    """
    thetalist = np.array(thetalist_current).copy()
    Tsb = FKinSpace(M,Slist, thetalist)
    Vs = np.dot(Adjoint(Tsb),
                se3ToVec(MatrixLog6(np.dot(TransInv(Tsb), T))))
    theta_vel = np.dot(np.linalg.pinv(JacobianSpace(Slist,
                                                    thetalist)), Vs)
    return theta_vel


class ModifiedInterbotixManipulatorXS(object):
    def __init__(self, robot_model, group_name="arm", gripper_name="gripper", robot_name=None, moving_time=2.0, accel_time=0.3, gripper_pressure=0.5, gripper_pressure_lower_limit=150, gripper_pressure_upper_limit=350, init_node=True):
        self.dxl = InterbotixRobotXSCore(robot_model, robot_name, init_node)
        self.arm = ModifiedInterbotixArmXSInterface(self.dxl, robot_model, group_name, moving_time, accel_time)
        if gripper_name is not None:
            self.gripper = InterbotixGripperXSInterface(self.dxl, gripper_name, gripper_pressure, gripper_pressure_lower_limit, gripper_pressure_upper_limit)


class ModifiedInterbotixArmXSInterface(InterbotixArmXSInterface):
    def __init__(self, *args, **kwargs):
        super(ModifiedInterbotixArmXSInterface, self).__init__(*args, **kwargs)
        self.waist_index = self.group_info.joint_names.index("waist")

    def set_ee_pose_matrix_fast(self, T_sd, custom_guess=None, execute=True):
        """
        this version of set_ee_pose_matrix does not set the velocity profile registers in the servos and therefore runs faster
        """
        if (custom_guess is None):
            initial_guesses = self.initial_guesses
        else:
            initial_guesses = [custom_guess]

        for guess in initial_guesses:
            theta_list, success = mr.IKinSpace(self.robot_des.Slist, self.robot_des.M, T_sd, guess, 0.001, 0.001)
            solution_found = True

            # Check to make sure a solution was found and that no joint limits were violated
            if success:
                theta_list = [int(elem * 1000)/1000.0 for elem in theta_list]
                for x in range(self.group_info.num_joints):
                    if not (self.group_info.joint_lower_limits[x] <= theta_list[x] <= self.group_info.joint_upper_limits[x]):
                        solution_found = False
                        break
            else:
                solution_found = False

            if solution_found:
                if execute:
                    self.publish_positions_fast(theta_list)
                    self.T_sb = T_sd
                return theta_list, True
            else:
                rospy.loginfo("Guess failed to converge...")

        rospy.loginfo("No valid pose could be found")
        return theta_list, False
    
    def publish_positions_fast(self, positions):
        self.joint_commands = list(positions)
        joint_commands = JointGroupCommand(self.group_name, self.joint_commands)
        self.core.pub_group.publish(joint_commands)


DEFAULT_ROTATION = np.array([[0 , 0, 1.0],
                             [0, 1.0,  0],
                             [-1.0,  0, 0]])

##############################################################################

class WidowX_Controller(RobotControllerBase):
    def __init__(self, robot_name, print_debug, gripper_params,
                 enable_rotation='6dof',
                 gripper_attached='custom',
                 normal_base_angle=0):
        """
        gripper_attached: either "custom" or "default"
        """
        print('waiting for widowx_controller to be set up...')
        self.bot = ModifiedInterbotixManipulatorXS(robot_model=robot_name)
        # TODO: Why was this call necessary in the visual_foresight? With the new SDK it messes the motor parameters
        # self.bot.dxl.robot_set_operating_modes("group", "arm", "position", profile_type="velocity", profile_velocity=131, profile_acceleration=15)

        if gripper_params is None:
            gripper_params = {}

        self._robot_name = robot_name
        rospy.on_shutdown(self.clean_shutdown)

        logger = logging.getLogger('robot_logger')
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        log_level = logging.WARN
        if print_debug:
            log_level = logging.DEBUG
        ch = logging.StreamHandler()
        ch.setLevel(log_level)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        self._init_gripper(gripper_attached, gripper_params)

        self._joint_lock = Lock()
        self._angles, self._velocities, self._effort = {}, {}, {}
        rospy.Subscriber(f"/{robot_name}/joint_states", JointState, self._joint_callback)
        time.sleep(1)
        self._n_errors = 0

        self._upper_joint_limits = np.array(self.bot.arm.group_info.joint_upper_limits)
        self._lower_joint_limits = np.array(self.bot.arm.group_info.joint_lower_limits)
        self._qn = self.bot.arm.group_info.num_joints

        self.joint_names = self.bot.arm.group_info.joint_names
        self.default_rot = np.dot(tr.eulerAnglesToRotationMatrix([0, 0, normal_base_angle]), DEFAULT_ROTATION)

        # self.neutral_joint_angles = np.zeros(self._qn)
        # self.neutral_joint_angles[0] = normal_base_angle
        # self.neutral_joint_angles[-2] = np.pi / 2
        self.neutral_joint_angles = np.array([-0.13192235, -0.76238847,  0.44485444, -0.01994175,  1.7564081,  -0.15953401])
        self.enable_rotation = enable_rotation
        
        self.low_pos_angles = np.array([-0.05215535, -0.82834965,  0.9372623,  -0.0966408,   1.35450506, -0.09050487])
        # self.low_pos_angles = np.array([-0.01533981, -1.00168943,  0.99248558, -0.00153398,  1.51097107, -0.05215535])
        self.grip_pos_joint_angles = np.array([-0.01994175,  0.15646605, -0.05215535, -0.0398835,   1.24559247,  0.05062137]) #nvm this one
        # self.grip_pos_joint_angles = np.array([-0.01687379,  0.33287385, -0.20248547, -0.00460194,  1.25172842,  0.06902914]) # double again

        # self.grip_pos_joint_angles = np.array([ 0.04908739,  0.18561168, -0.02914564,  0.01227185,  1.1826992,  -0.05215535]) # 55, close

        # self.grip_pos_joint_angles = np.array([-0.0966408,   0.26537868, -0.06749516, -0.00306796,  1.13974774, -0.14572819]) # 100
        # self.grip_pos_joint_angles = np.array([-0.08743691,  0.36815539, -0.24850489,  0.00306796,  1.12747586,  0.02607767])
        # self.grip_pos_joint_angles = np.array([-0.08590293, 0.55069911, -0.47706804, -0.00613592,  1.39745653, -0.06596117])
        # self.grip_pos_joint_angles = np.array([-0.04295146,  0.34974763, -0.37582532, -0.07823303,  1.25172842,  0.30372819]) # 80 traj
        # self.grip_pos_joint_angles = np.array([-0.04141748, 0.44792241, -0.42031074, -0.10737866,  1.34376717,  0.26691267])
        # self.grip_pos_joint_angles = np.array([0.00920388, 0.42491269, -0.52001953, -0.05368933, 1.61221385, 0.04295146])
        
        # self.grip_pos_joint_angles = np.array([-0.00460194, 0.23316509, -0.05368933, 0.11351458, 1.11673808, 0.08436894])
        # self.grip_pos_joint_angles = np.array([-0.0398835, 0.2960583, -0.1518641, 0.09817477, 1.24712646, 0.0])
        # self.grip_pos_joint_angles = np.array([-0.04295146, 0.12271847, 0.13345633, 0.03374758, 1.08299041, -0.08283497])
        # # self.grip_pos_joint_angles = np.array([-0.05141748, 0.30372819, -0.12425245, 0.02454369, 1.08145654, -0.04295146])
        # self.grip_pos_joint_angles = np.array([-0.06596117, 0.37889326, -0.04908739,  0.02300971,  1.08759236, -0.05368933])
    def clean_shutdown(self):
        pid = os.getpid()
        logging.getLogger('robot_logger').info('Exiting example w/ pid: {}'.format(pid))
        logging.shutdown()
        os.kill(pid, 9)

    def move_to_state(self, target_xyz, target_zangle, duration=1.5):
        new_pose = np.eye(4)
        new_pose[:3, -1] = target_xyz
        new_quat = Quaternion(axis=np.array([0.0, 0.0, 1.0]), angle=target_zangle) * Quaternion(matrix=self.default_rot)
        new_pose[:3, :3] = new_quat.rotation_matrix
        self.move_to_eep(new_pose, duration)

    def set_moving_time(self, moving_time):
        self.bot.arm.set_trajectory_time(moving_time=moving_time*1.25, accel_time=moving_time * 0.5)

    def move_to_eep(self, target_pose, duration=1.5, blocking=True, check_effort=True, step=True):
        try:
            if step and not blocking:
                # this is a call from the `step` function so we use a custom faster way to set the ee pose
                solution, success = self.bot.arm.set_ee_pose_matrix_fast(target_pose, custom_guess=self.get_joint_angles(), execute=True)
            else:
                solution, success = self.bot.arm.set_ee_pose_matrix(target_pose, custom_guess=self.get_joint_angles(),
                                        moving_time=duration, accel_time=duration * 0.45, blocking=blocking)
            
            self.des_joint_angles = solution
            
            if not success:
                print('no IK solution found, do nothing')
                # self.open_gripper()
                # self.move_to_neutral()
                # raise Environment_Exception

            if check_effort:
                max_effort_abs_values = np.array([800, 1000, 600.0, 600.0, 600.0, 700.0]) * 1.5 # experimenting with this
                # max_effort_abs_values = np.array([800, 1000, 600.0, 300.0, 600.0, 700.0])
                # max_effort_abs_values = np.array([10000] * 6)
                if np.max(np.abs(self.get_joint_effort()) - max_effort_abs_values) > 10:
                    print('violation ', np.abs(self.get_joint_effort()) - max_effort_abs_values)
                    print('motor number: ', np.argmax(np.abs(self.get_joint_effort()) - max_effort_abs_values))
                    print('max effort reached: ', self.get_joint_effort())
                    print('max effort allowed ', max_effort_abs_values)
                    self.open_gripper()
                    self.move_to_neutral()
                    raise Environment_Exception

        except rospy.service.ServiceException:
            print('stuck during move')
            import pdb; pdb.set_trace()
            self.move_to_neutral()
    
    def set_joint_angles(self, target_positions, duration=4):
        target_positions_to_reach = [target_positions]
        if len(target_positions_to_reach) > 1000:
            print('set_joint_angles failed')
            raise Environment_Exception
        try:
            while target_positions_to_reach:
                target_position = target_positions_to_reach[-1]
                success = self.bot.arm.set_joint_positions(target_position, moving_time=duration)
                if success is False:
                    intermediate_pos = np.mean([self.get_joint_angles(), target_position], axis=0)
                    target_positions_to_reach.append(intermediate_pos)
                else:
                    target_positions_to_reach.pop()
        except rospy.service.ServiceException:
            print('stuck during motion')
            import pdb; pdb.set_trace()

    def check_motor_status_and_reboot(self):
        # print("checking motor status")
        status_codes = self.bot.dxl.robot_get_motor_registers("group", "all", "Hardware_Error_Status")
        # print(status_codes.values)
        # import ipdb; ipdb.set_trace()
        if len(status_codes.values) < 7:
            print("Some motor went wrong!")
            self.bot.dxl.robot_reboot_motors("group", "all", enable=True, smart_reboot=True)
            print("robot rebooted")
            self.move_to_neutral()
            raise Environment_Exception

    def move_to_neutral(self, duration=4):
        duration = 2
        print('moving to neutral..')
        try:
            self.bot.arm.publish_positions(self.neutral_joint_angles, moving_time=duration)
            # print("Error in neutral position", np.linalg.norm(self.neutral_joint_angles - self.get_joint_angles()))
            # import ipdb; ipdb.set_trace()
            if np.linalg.norm(self.neutral_joint_angles - self.get_joint_angles()) > 0.1:
                print("moving to neutral failed!")
                self.check_motor_status_and_reboot()
        except rospy.service.ServiceException:
            print('stuck during reset')
            import pdb; pdb.set_trace()
    
    def move_to_grip(self, duration=4):
        duration = 2
        print('moving to grip positions..')
        try:
            self.bot.arm.publish_positions(self.grip_pos_joint_angles, moving_time=duration)
            # print("Error in neutral position", np.linalg.norm(self.neutral_joint_angles - self.get_joint_angles()))
            # import ipdb; ipdb.set_trace()
            print("moving test")
            print(self.grip_pos_joint_angles)
            if np.linalg.norm(self.grip_pos_joint_angles - self.get_joint_angles()) > 0.1:
                print("moving to grip failed!")
                self.check_motor_status_and_reboot()
        except rospy.service.ServiceException:
            print('stuck during reset')
            import pdb; pdb.set_trace()
    def move_to_neutral_low(self, duration=4):
        duration = 2 
        print('moving to grip positions..')
        try:
            # self.bot.arm.publish_positions(self.low_pos_angles, moving_time=duration)
            self.bot.arm.publish_positions(self.neutral_joint_angles, moving_time=duration) 
            # print("Error in neutral position", np.linalg.norm(self.neutral_joint_angles - self.get_joint_angles()))
            # import ipdb; ipdb.set_trace()
            if np.linalg.norm(self.neutral_joint_angles - self.get_joint_angles()) > 0.1:
                print("moving to neutral through low failed!")
                self.check_motor_status_and_reboot()
        except rospy.service.ServiceException:
            print('stuck during reset')
            import pdb; pdb.set_trace()

    def _joint_callback(self, msg):
        with self._joint_lock:
            for name, position, velocity, effort  in zip(msg.name, msg.position, msg.velocity, msg.effort):
                self._angles[name] = position
                self._velocities[name] = velocity
                self._effort[name] = effort

    def get_joint_angles(self):
        '''
        Returns current joint angles
        '''
        with self._joint_lock:
            try:
                return np.array([self._angles[k] for k in self.joint_names])
            except KeyError:
                return None

    def get_joint_effort(self):
        '''
        Returns current joint angles
        '''
        with self._joint_lock:
            try:
                return np.array([self._effort[k] for k in self.joint_names])
            except KeyError:
                return None

    def get_joint_angles_velocity(self):
        '''
        Returns velocities for joints
        '''
        with self._joint_lock:
            try:
                return np.array([self._velocities[k] for k in self.joint_names])
            except KeyError:
                return None

    def get_cartesian_pose(self, matrix=False):
        # Returns cartesian end-effector pose
        joint_positions = list(self.bot.dxl.joint_states.position[self.bot.arm.waist_index:(self._qn + self.bot.arm.waist_index)])
        pose = mr.FKinSpace(self.bot.arm.robot_des.M, self.bot.arm.robot_des.Slist, joint_positions)
        if matrix:
            return pose
        else:
            return np.concatenate([pose[:3, -1], np.array(Quaternion(matrix=pose[:3, :3]).elements)])

    def _init_gripper(self, gripper_attached, gripper_params):
        if gripper_attached == 'custom':
            self._gripper = GripperController(robot_name=self.bot.dxl.robot_name, des_pos_max=gripper_params.des_pos_max, des_pos_min=gripper_params.des_pos_min)
            self.custom_gripper_controller = True
        elif gripper_attached == 'custom_narrow':
            self._gripper = GripperController(robot_name=self.bot.dxl.robot_name, upper_limit=0.022, des_pos_max=gripper_params.des_pos_max, des_pos_min=gripper_params.des_pos_min)
            self.custom_gripper_controller = True
        elif gripper_attached == 'default':
            self.custom_gripper_controller = False
        else:
            raise ValueError("gripper_attached value has to be either 'custom', 'custom_narrow' or 'default'")

    def get_gripper_desired_position(self):
        if self.custom_gripper_controller:
            return self._gripper.get_gripper_target_position()
        else:
            return self.des_gripper_state

    def set_continuous_gripper_position(self, target):
        assert self.custom_gripper_controller
        self._gripper.set_continuous_position(target)

    def get_gripper_position(self):
        assert self.custom_gripper_controller  # we need the joint_states subscriber to keep track of gripper position
        # TODO: for standard gripper we could use WidowX_Controller self._angles to return the current position similarly as custom_gripper_controller
        return self.get_continuous_gripper_position()

    def get_continuous_gripper_position(self):
        assert self.custom_gripper_controller
        return self._gripper.get_continuous_position()

    def wait_until_gripper_position_reached(self):
        if self.custom_gripper_controller:
            goal_reached = self.get_gripper_desired_position() - 0.075 < self.get_continuous_gripper_position() < self.get_gripper_desired_position() + 0.075
            if goal_reached:
                # don't wait if we are already at the target
                return
            # otherwise wait until gripper stopped moving
            still_moving = True
            previous_position = self.get_continuous_gripper_position()
            while still_moving:
                time.sleep(0.05)
                current_position = self.get_continuous_gripper_position()
                still_moving = not (previous_position - 0.01 < current_position < previous_position + 0.01)
                previous_position = current_position
            # wait a bit to give time for gripper to exert force on the object
            time.sleep(0.1)

    def open_gripper(self, wait=False):
        if self.custom_gripper_controller:
            self._gripper.open()
        else:
            self.des_gripper_state = np.array([1])
            self.bot.gripper.open()

    def close_gripper(self, wait=False):
        if self.custom_gripper_controller:
            self._gripper.close()
        else:
            self.des_gripper_state = np.array([0])
            self.bot.gripper.close()


##############################################################################

class WidowXVelocityController(WidowX_Controller):
    # NOTE (YL) THIS CONTROLLER IS not working
    def __init__(self, *args, **kwargs):
        super(WidowXVelocityController, self).__init__(*args, **kwargs)
        self.bot.dxl.robot_set_operating_modes("group", "arm", "velocity")
        self.bot.arm.set_trajectory_time(moving_time=0.2, accel_time=0.05)

        # TODO: do we need the SpaceMouseRemoteReader?
        # TODO(YL): WARNING: This package is not avail, fix this
        from visual_mpc.envs.util.teleop.server import SpaceMouseRemoteReader
        self.space_mouse = SpaceMouseRemoteReader()
        rospy.Timer(rospy.Duration(0.02), self.update_robot_cmds)
        self.last_update_cmd = time.time()
        self.enable_cmd_thread = False
        self.do_reset = False
        self.task_stage = 0
        self.num_task_stages = 1e9

    def update_robot_cmds(self, event):
        reading = self.space_mouse.get_reading()
        if reading is not None and self.enable_cmd_thread:
            # print('delta t cmd update, ', time.time() - self.last_update_cmd)
            self.last_update_cmd = time.time()
            if reading['left'] and reading['right'] or reading['left_and_right']:
                self.task_stage += 1
                self.task_stage = np.clip(self.task_stage, 0, self.num_task_stages)
                if self.task_stage == self.num_task_stages:
                    print('resetting!')
                    self.do_reset = True
                rospy.sleep(1.0)
            # t0 = time.time()
            self.apply_spacemouse_action(reading)
            # print('apply action time', time.time() - t0)

    def apply_spacemouse_action(self, readings):
        if readings is None:
            print('readings are None!')
            return
        if self.custom_gripper_controller:
            if readings['left']:
                self._gripper.open()
            if readings['right']:
                self._gripper.close()
        else:
            if readings['left']:
                self.bot.open_gripper()
            if readings['right']:
                self.bot.close_gripper()

        if self.enable_rotation:
            # t1 = time.time()
            pose = self.get_cartesian_pose(matrix=True)
            # print('get pose time', time.time() - t1)

            current_quat = Quaternion(matrix=pose[:3, :3])
            translation_scale = 0.1
            commanded_translation_velocity = readings['xyz'] * translation_scale
            new_pos = pose[:3, 3] + commanded_translation_velocity

            # rotation_scale = 0.3
            rotation_scale = 0.4
            commanded_rotation_velocity = readings['rot'] * rotation_scale
            if self.enable_rotation == '4dof':
                commanded_rotation_velocity = commanded_rotation_velocity[2]
                new_rot = Quaternion(axis=[0, 0, 1], angle=commanded_rotation_velocity) * current_quat
            elif self.enable_rotation == '6dof':
                new_rot = Quaternion(axis=[1, 0, 0], angle=commanded_rotation_velocity[0]) * \
                          Quaternion(axis=[0, 1, 0], angle=commanded_rotation_velocity[1]) * \
                          Quaternion(axis=[0, 0, 1], angle=commanded_rotation_velocity[2]) * current_quat
            else:
                raise NotImplementedError

            new_transform = np.eye(4)
            new_transform[:3, :3] = new_rot.rotation_matrix
            new_transform[:3, 3] = new_pos
        else:
            new_transform = self.get_cartesian_pose(matrix=True)
            new_transform[:3, 3] += readings['xyz'] * 0.1

        # t2 = time.time()
        joint_velocities = compute_joint_velocities_from_cartesian(self.bot.robot_des.Slist, self.bot.robot_des.M,
                                                                   new_transform, self.get_joint_angles())
        # print('compute joint vel time', time.time() - t2)
        self.cap_joint_limits(joint_velocities)
        try:
            joint_commands = JointCommands(joint_velocities)
            self.bot.core.pub_group.publish(joint_commands)
        except:
            print('could not set joint velocity!')

    def stop_motors(self):
        joint_commands = JointGroupCommand(self.bot.arm.group_name, )
        self.bot.core.pub_group.publish(joint_commands)
        self.bot.core.pub_group.publish(JointCommands())

    def move_to_state(self, target_xyz, target_zangle, duration=2):
        pose = np.eye(4)
        rot = Quaternion(axis=[0, 0, 1], angle=target_zangle) * Quaternion(matrix=self.default_rot)
        pose[:3, :3] = rot.rotation_matrix
        pose[:3, 3] = target_xyz
        joint_pos, success = self.bot.set_ee_pose_matrix(pose, custom_guess=self.get_joint_angles(), moving_time=2,
                                                         execute=False)
        if success:
            self.move_to_pos_with_velocity_ctrl(joint_pos)
            return True
        else:
            print('no kinematics solution found!')
            raise Environment_Exception

    def cap_joint_limits(self, ctrl):
        for i in range(self.qn):
            if self.get_joint_angles()[i] < self._lower_joint_limits[i]:
                print('ctrl', ctrl)
                print('ja', self.get_joint_angles())
                print('limit', self._lower_joint_limits)
                print('lower joint angle limit violated for j{}'.format(i + 1))
                if ctrl[i] < 0:
                    print('setting to zero')
                    ctrl[i] = 0
            if self.get_joint_angles()[i] > self._upper_joint_limits[i]:
                print('ctrl', ctrl)
                print('ja', self.get_joint_angles())
                print('limit', self._lower_joint_limits)
                print('upper joint angle limit violated for j{}'.format(i + 1))
                if ctrl[i] > 0:
                    print('setting to zero')
                    ctrl[i] = 0

    def move_to_neutral(self, duration=4):
        self.move_to_pos_with_velocity_ctrl(self.neutral_joint_angles, duration=duration)

    def move_to_pos_with_velocity_ctrl(self, des, duration=3):
        nsteps = 30
        per_step = float(duration)/nsteps

        tstart = time.time()
        current = self.get_joint_angles()
        error = des - current
        while (time.time() - tstart) < duration and np.linalg.norm(error) > 0.15:
            current = self.get_joint_angles()
            error = des - current
            ctrl = error * 0.8
            max_speed = 0.5
            ctrl = np.clip(ctrl, -max_speed, max_speed)
            self.cap_joint_limits(ctrl)
            self.bot.core.pub_group.publish(JointCommands(ctrl))
            self._last_healthy_tstamp = rospy.get_time()
            rospy.sleep(per_step)
        self.bot.core.pub_group.publish(JointCommands(np.zeros(self._qn)))


##############################################################################

if __name__ == '__main__':
    dir = '/mount/harddrive/spt/trainingdata/realworld/can_pushing_line/2020-09-04_09-28-29/raw/traj_group0/traj2'
    dict = pkl.load(open(dir + '/policy_out.pkl', "rb"))
    actions = np.stack([d['actions'] for d in dict], axis=0)
    dict = pkl.load(open(dir + '/obs_dict.pkl', "rb"))
    states = dict['raw_state']

    # TODO: check if renaming is required for widowx to widowx_controller
    controller = WidowXVelocityController('widowx', True)

    rospy.sleep(2)
    controller.move_to_neutral()
    # controller.move_to_state(states[0, :3], target_zangle=states[0, 3])
    controller.move_to_state(states[0, :3], target_zangle=0.)

    prev_eef = controller.get_cartesian_pose()[:3]
    for t in range(20):
        # low_bound = np.array([1.12455181e-01, 8.52311223e-05, 3.23975718e-02, -2.02, -0.55]) + np.array([0, 0, 0.05, 0, 0])
        # high_bound = np.array([0.29880695, 0.22598613, 0.15609235, 1.52631092, 1.39])
        # x, y, z, theta = np.random.uniform(low_bound[:4], high_bound[:4])
        controller.apply_endeffector_velocity(actions[t]/0.2)


        new_eef = controller.get_cartesian_pose()[:3]
        print("current eef pos", new_eef[:3])
        print('desired eef pos', states[t, :3])
        print('delta', states[t, :3] - new_eef[:3])
        rospy.sleep(0.2)
