import rospy
import numpy as np
from threading import Lock
from sensor_msgs.msg import JointState
import time

from std_msgs.msg import Float64
from widowx_controller.srv import OpenGripper, OpenGripperResponse
from widowx_controller.srv import GetGripperDesiredState, GetGripperDesiredStateResponse

try:
    # older version of interbotix sdk
    from interbotix_xs_sdk.msg import JointSingleCommand
except:
    # newer version of interbotix sdk
    from interbotix_xs_msgs.msg import JointSingleCommand

from widowx_controller.controller_base import GripperControllerBase


class GripperController(GripperControllerBase):
    def __init__(self, robot_name, create_node=False, upper_limit=0.035, lower_limit=0.010, des_pos_max=1, des_pos_min=0):
        if create_node:
            rospy.init_node('gripper_controller')
        assert des_pos_max >= des_pos_min, "gripper des_pos_max has to be >= des_pos_min"
        self.des_pos_max = des_pos_max
        self.des_pos_min = des_pos_min
        self._upper_limit = upper_limit
        self._lower_limit = lower_limit
        assert self._upper_limit > self._lower_limit
        if not create_node:
            rospy.Timer(rospy.Duration(0.02), self.update_gripper_pwm)

        self._joint_lock = Lock()
        self._des_pos_lock = Lock()

        self._angles = {}
        self._velocities = {}
        self._pub_gripper_command = rospy.Publisher(f"/{robot_name}/commands/joint_single", JointSingleCommand, queue_size=3)
        rospy.Subscriber(f"/{robot_name}/joint_states", JointState, self._joint_callback)

        self._moving = False
        self._time_movement_started = None
        self._grace_period_until_can_be_marked_as_stopped = 0.1
        self._des_pos = None
        self.des_pos = self._upper_limit

    @property
    def des_pos(self):
        return self._des_pos

    @des_pos.setter
    def des_pos(self, value):
        if value != self._des_pos:
            with self._des_pos_lock:
                self._moving = True
                self._time_movement_started = time.time()
                self._des_pos = value

    def get_gripper_pos(self):
        with self._joint_lock:
            if 'left_finger' not in self._angles:
                print(" WARNING!! left_finger should be avail in gripper")
                return 0.0
            return self._angles['left_finger']

    def _joint_callback(self, msg):
        with self._joint_lock:
            for name, position, velocity in zip(msg.name, msg.position, msg.velocity):
                self._angles[name] = position
                self._velocities[name] = velocity

    def open(self):
        self.des_pos = self._upper_limit

    def close(self):
        self.des_pos = self._lower_limit

    def set_continuous_position(self, target):
        target_clipped = np.clip(target, self.des_pos_min, self.des_pos_max)
        if target != target_clipped:
            print('Warning target gripper pos outside of range', target)
        self.des_pos = self.denormalize(target_clipped)

    def get_continuous_position(self):
        gripper_pos = self.get_gripper_pos()
        return self.normalize(gripper_pos)

    def normalize(self, x):
        return (self.des_pos_max - self.des_pos_min) * (x - self._lower_limit) / (self._upper_limit - self._lower_limit) + self.des_pos_min

    def denormalize(self, x):
        return (x - self.des_pos_min) * (self._upper_limit - self._lower_limit) / (self.des_pos_max - self.des_pos_min) + self._lower_limit

    def is_moving(self):
        return self._moving

    def get_gripper_target_position(self):
        des_pos_normed = self.normalize(self.des_pos)
        assert des_pos_normed <= self.des_pos_max and des_pos_normed >= self.des_pos_min
        return des_pos_normed

    def update_gripper_pwm(self, event):
        with self._des_pos_lock:
            moving = self._moving
            des_pos = self.des_pos

        if moving:
            gripper_pos = self.get_gripper_pos()
            ctrl = (des_pos - gripper_pos)*300
            pwm = self.get_gripper_pwm(ctrl)
            gripper_command = JointSingleCommand('gripper', pwm)
            self._pub_gripper_command.publish(gripper_command)

    def get_gripper_pwm(self, pressure):
        """
        :param pressure: range -1, 1
        :return: pwm
        """
        pressure = np.clip(pressure, -1, 1)
        # offset = 150
        offset = 0
        if pressure < 0:
            gripper_pwm = -(offset + int(-pressure * 350))
        if pressure >= 0:
            gripper_pwm = offset + int(pressure * 350)
        time_since_movements_started = time.time() - self._time_movement_started
        # if abs(self._velocities['gripper']) == 0.0 and time_since_movements_started > self._grace_period_until_can_be_marked_as_stopped:
        #     # set 0 to stop sending commands by setting self.moving = False
        #     gripper_pwm = 0
        #     self._moving = False
        #     self._time_movement_started = None
        return gripper_pwm


class GripperControllerServer(GripperController):
    def __init__(self, robot_name, create_node=True, upper_limit=0.037, lower_limit=0.010, des_pos_max=1, des_pos_min=0):
        super(GripperControllerServer, self).__init__(robot_name, create_node, upper_limit, lower_limit, des_pos_max, des_pos_min)

        rospy.Service('open_gripper', OpenGripper, self.open_gripper_service)
        # rospy.Service('set_gripper_position', SetGripperPosition, self.set_gripper_position_service)
        rospy.Service('get_gripper_desired_state', GetGripperDesiredState, self.get_gripper_desired_state_service)
        rospy.Subscriber("/gripper_despos", Float64, self._gripper_despos_callback)

    def _gripper_despos_callback(self, msg):
        self.set_continuous_position(msg.data)

    def open_gripper_service(self, req):
        self.open()
        return OpenGripperResponse()

    def get_gripper_desired_state_service(self, req):
        return GetGripperDesiredStateResponse(self.get_gripper_target_position())


def run():
    controller = GripperControllerServer(robot_name='wx250s')
    rospy.sleep(0.1)
    r = rospy.Rate(50)
    while not rospy.is_shutdown():
        controller.update_gripper_pwm(None)
        r.sleep()

if __name__ == '__main__':
    run()
