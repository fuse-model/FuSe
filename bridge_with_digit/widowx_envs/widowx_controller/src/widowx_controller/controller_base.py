# /usr/bin/env python3

from abc import ABC, abstractmethod
from typing import Optional

##############################################################################

class GripperControllerBase(ABC):

    @abstractmethod
    def __init__(self, robot_name):
        pass

    @property
    def des_pos(self) -> Optional[float]:
        return None

    @des_pos.setter
    @abstractmethod
    def des_pos(self, value):
        pass

    @abstractmethod
    def get_gripper_pos(self):
        pass

    @abstractmethod
    def open(self):
        pass

    @abstractmethod
    def close(self):
        pass

    @abstractmethod
    def set_continuous_position(self, target):
        pass

    @abstractmethod
    def get_continuous_position(self):
        pass

    @abstractmethod
    def is_moving(self):
        pass

    @abstractmethod
    def get_gripper_target_position(self):
        pass


##############################################################################

class RobotControllerBase(ABC):

    @abstractmethod
    def __init__(self, robot_name, print_debug):
        pass

    @abstractmethod
    def move_to_state(self, target_xyz, target_zangle, duration=1.5):
        pass

    @abstractmethod
    def set_moving_time(self, moving_time):
        pass

    @abstractmethod
    def move_to_eep(self, target_pose, duration=1.5):
        """
        :param target_pose: Cartesian pose (x,y,z, quat).
        :param duration: Total time trajectory will take before ending
        """
        pass

    @abstractmethod
    def set_joint_angles(self, target_positions, duration=4):
        pass

    @abstractmethod
    def move_to_neutral(self, duration=4):
        pass

    @abstractmethod
    def get_joint_angles(self):
        """Get current joint angles"""
        pass

    @abstractmethod
    def get_joint_effort(self):
        """Get current joint efforts"""
        pass

    @abstractmethod
    def get_joint_angles_velocity(self):
        """Get current joint angle velocities"""
        pass

    @abstractmethod
    def get_cartesian_pose(self, matrix=False):
        pass

    def get_state(self):
        """Get a tuple of (joint_angles, joint_angles_velocity, cartesian_pose)"""
        return self.get_joint_angles(), \
            self.get_joint_angles_velocity(), \
            self.get_cartesian_pose()

    @abstractmethod
    def open_gripper(self, wait=False):
        pass

    @abstractmethod
    def close_gripper(self, wait=False):
        pass
