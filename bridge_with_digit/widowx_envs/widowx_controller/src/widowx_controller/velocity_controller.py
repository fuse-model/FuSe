"""
NOTE (YL) This VelocityController is not working,
mainly serves as a backup from the original code.

This is moved here from the original widowx_controller.py file, to
make the code more readable and cleaner.
"""

from __future__ import print_function
import numpy as np
import rospy
import time
import pickle as pkl
from pyquaternion import Quaternion

try:
    # older version of interbotix sdk
    from interbotix_xs_sdk.msg import JointGroupCommand
except:
    # newer version of interbotix sdk
    from interbotix_xs_msgs.msg import JointGroupCommand

from widowx_controller.widowx_controller import WidowX_Controller
from visual_mpc.envs.util.teleop.server import SpaceMouseRemoteReader

from widowx_envs.utils.exceptions import Environment_Exception

from modern_robotics.core import \
    JacobianSpace, Adjoint, MatrixLog6, se3ToVec, TransInv, FKinSpace

##############################################################################


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
    Tsb = FKinSpace(M, Slist, thetalist)
    Vs = np.dot(Adjoint(Tsb),
                se3ToVec(MatrixLog6(np.dot(TransInv(Tsb), T))))
    theta_vel = np.dot(np.linalg.pinv(JacobianSpace(Slist,
                                                    thetalist)), Vs)
    return theta_vel

##############################################################################


class WidowXVelocityController(WidowX_Controller):
    def __init__(self, *args, **kwargs):
        super(WidowXVelocityController, self).__init__(*args, **kwargs)
        self.bot.dxl.robot_set_operating_modes("group", "arm", "velocity")
        self.bot.arm.set_trajectory_time(moving_time=0.2, accel_time=0.05)

        # TODO: do we need the SpaceMouseRemoteReader?
        # TODO(YL): WARNING: This package is not avail, fix this
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
        joint_velocities = compute_joint_velocities_from_cartesian(
            self.bot.robot_des.Slist, self.bot.robot_des.M,
            new_transform, self.get_joint_angles()
        )
        # print('compute joint vel time', time.time() - t2)
        self.cap_joint_limits(joint_velocities)
        try:
            # TODO(YL): where joint_commands is defined? cant find it
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
        joint_pos, success = self.bot.set_ee_pose_matrix(
            pose,
            custom_guess=self.get_joint_angles(),
            moving_time=2,
            execute=False
        )
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
