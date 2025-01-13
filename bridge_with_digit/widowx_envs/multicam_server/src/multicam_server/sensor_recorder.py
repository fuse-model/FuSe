import threading

import numpy as np
import rospy
import os
from threading import Lock, Semaphore
import cv2
from std_msgs.msg import Float32MultiArray
import copy
import hashlib
import logging
import time

from sensor_msgs.msg import JointState


class IMUObservation(object):
    def __init__(self):
        self.euler = None 
        self.linear_acceleration = None 
        self.gyro = None 
        self.timestep = None
        self.mutex = Lock()


class IMURecorder: 
    MAX_REPEATS = 50
    def __init__(self, topic_name):
        """
        :param topic_name:
        """

        self.errored = False 
        self._obs_dim = 10 # length of observation from single timestep: 3 for euler/accel/gyro each, one for timestep 
        self._latest_obs = IMUObservation()

        self._is_first_status, self._status_sem = True, Semaphore(value=0)
        self._last_hash, self._num_repeats = None, 0
        self._last_hash_get_obs = None

        self._topic_name = topic_name

        rospy.Subscriber(topic_name, Float32MultiArray, self.store_latest_obs)
        logger = logging.getLogger('robot_logger')
        logger.debug('downing sema on topic: {}'.format(topic_name))
        success = self._status_sem.acquire(timeout=5)

        if not success:
            print('Still waiting for an data to arrive at IMU recorder... Topic name:', self._topic_name)
            self._status_sem.acquire()
        logger.info(
            f"IMU at {topic_name} subscribed")
    
    def get_reading(self, arg=None):
        self._latest_obs.mutex.acquire()
        current_hash = hashlib.sha256(self._data.tostring()).hexdigest()
        if self._last_hash_get_obs is not None:
            if current_hash == self._last_hash_get_obs:
                print('Repeated measurements for IMU get_reading method!')
        self._last_hash_get_obs = current_hash
        reading = { 
            "euler": self._latest_obs.euler, 
            "acceleration": self._latest_obs.linear_acceleration, 
            "gyro": self._latest_obs.gyro, 
            "timestep": self._latest_obs.timestep
        }
        self._latest_obs.mutex.release()
        return reading
    

    def _proc_obs(self, latest_obsv, data):
        # data format: [euler,  timestep] * buffer_size
        data = np.array(data.data).copy() 
        self._data = data
        assert len(data) % self._obs_dim == 0
        num_splits = len(data) // self._obs_dim 
        split_data = np.split(self._data, num_splits) # row corresponds to each obs (euler + timestep), column to type (e.g. euler_1 or timestep)
        split_data = np.stack(split_data)
        latest_obsv.euler = split_data[:, 0:3] # nsteps x 3 
        latest_obsv.linear_acceleration = split_data[:, 3:6]
        latest_obsv.gyro = split_data[:, 6:9]
        latest_obsv.timestep = split_data[:, 9]

    @property
    def topic_name(self):
        return self._topic_name

    def store_latest_obs(self, data):
        # data = np.array(data)
        self._latest_obs.mutex.acquire()

        t0 = rospy.get_time()
        self._proc_obs(self._latest_obs, data)

        current_hash = hashlib.sha256(self._data.tostring()).hexdigest()
        if self._is_first_status:
            self._is_first_status = False
            self._status_sem.release()

        elif self._last_hash == current_hash:
            if self._num_repeats < self.MAX_REPEATS:
                self._num_repeats += 1
            else:
                logging.getLogger('robot_logger').error(f'Too many repeated measurements.\
                        Check IMU topic {self._topic_name}!')
                self.errored = True 
                # go to neutral instead 
                # rospy.signal_shutdown('Too many repeated images. Check IMU!')
        else:
            self._num_repeats = 0

        self._last_hash = current_hash

        self._latest_obs.mutex.release()
        

class MicObservation(object):
    def __init__(self):
        self.audio = None 
        self.mutex = Lock()


class MicRecorder: 
    MAX_REPEATS = 100
    def __init__(self, topic_name):
        """
        :param topic_name:
        """

        self._latest_obs = IMUObservation()

        self._is_first_status, self._status_sem = True, Semaphore(value=0)
        self._last_hash, self._num_repeats = None, 0
        self._last_hash_get_obs = None

        self._topic_name = topic_name

        rospy.Subscriber(topic_name, Float32MultiArray, self.store_latest_obs)
        logger = logging.getLogger('robot_logger')
        logger.debug('downing sema on topic: {}'.format(topic_name))
        success = self._status_sem.acquire(timeout=5)

        if not success:
            print('Still waiting for an data to arrive at mic recorder... Topic name:', self._topic_name)
            self._status_sem.acquire()
        logger.info(
            f"Mic at {topic_name} subscribed")
    
    def get_reading(self, arg=None):
        self._latest_obs.mutex.acquire()
        current_hash = hashlib.sha256(self._data.tostring()).hexdigest()
        if self._last_hash_get_obs is not None:
            if current_hash == self._last_hash_get_obs:
                print('Repeated measurements for mic get_reading method!')
        self._last_hash_get_obs = current_hash
        audio = self._latest_obs.audio.copy()
        self._latest_obs.mutex.release()
        return audio
    

    def _proc_obs(self, latest_obsv, data):
        data = np.array(data.data).copy() 
        self._data = data 
        latest_obsv.audio = data 

    @property
    def topic_name(self):
        return self._topic_name

    def store_latest_obs(self, data):
        # print(data.data)
        # print()
        self._latest_obs.mutex.acquire()

        t0 = rospy.get_time()
        self._proc_obs(self._latest_obs, data)

        current_hash = hashlib.sha256(self._data.tostring()).hexdigest()
        if self._is_first_status:
            self._is_first_status = False
            self._status_sem.release()

        elif self._last_hash == current_hash:
            if self._num_repeats < self.MAX_REPEATS:
                self._num_repeats += 1
            else:
                logging.getLogger('robot_logger').error(f'Too many repeated measurements.\
                    Check mic topic {self._topic_name}!')
                rospy.signal_shutdown('Too many repeated images. Check IMU!')
        else:
            self._num_repeats = 0

        self._last_hash = current_hash

        self._latest_obs.mutex.release()

class JointObservation(object):
    def __init__(self):
        self.pos = None
        self.mutex = Lock()

class JointPosRecorder: 
    MAX_REPEATS = 100
    PRINT_AFTER = 20
    def __init__(self, topic_name):
        """
        :param topic_name:
        """
        self._get_repeats = 0
        self._obs_dim = 9  
        self._latest_obs = JointObservation()

        self._is_first_status, self._status_sem = True, Semaphore(value=0)
        self._last_hash, self._num_repeats = None, 0
        self._last_hash_get_obs = None

        self._topic_name = topic_name
        rospy.Subscriber(topic_name, JointState, self.store_latest_obs)
        logger = logging.getLogger('robot_logger')
        logger.debug('downing sema on topic: {}'.format(topic_name))
        success = self._status_sem.acquire(timeout=5)

        if not success:
            print('Still waiting for an data to arrive at joint recorder... Topic name:', self._topic_name)
            self._status_sem.acquire()
        logger.info(
            f"jointpos at {topic_name} subscribed")
    
    def get_reading(self, arg=None):
        self._latest_obs.mutex.acquire()
        current_hash = hashlib.sha256(self._data.tostring()).hexdigest()
        if self._last_hash_get_obs is not None:
            if current_hash == self._last_hash_get_obs:
                self._get_repeats += 1 
                if self._get_repeats >= self.PRINT_AFTER: 
                    print(f'{self._get_repeats} repeated measurements for joint pos get_reading method!')
            else: 
                self._get_repeats = 0 
                
        self._last_hash_get_obs = current_hash
        reading = self._latest_obs.pos
        self._latest_obs.mutex.release()
        return reading
    

    def _proc_obs(self, latest_obsv, data):
        data = np.array(data.position).copy() 
        self._data = data
        assert len(data) % self._obs_dim == 0
        latest_obsv.pos = data 

    @property
    def topic_name(self):
        return self._topic_name

    def store_latest_obs(self, data):
        # data = np.array(data)
        self._latest_obs.mutex.acquire()

        t0 = rospy.get_time()
        self._proc_obs(self._latest_obs, data)

        current_hash = hashlib.sha256(self._data.tostring()).hexdigest()
        if self._is_first_status:
            self._is_first_status = False
            self._status_sem.release()

        elif self._last_hash == current_hash:
            if self._num_repeats < self.MAX_REPEATS:
                self._num_repeats += 1
            else:
                logging.getLogger('robot_logger').error(f'Too many repeated measurements.\
                        Check IMU topic {self._topic_name}!')
                self.errored = True 
                # go to neutral instead 
                # rospy.signal_shutdown('Too many repeated images. Check IMU!')
        else:
            self._num_repeats = 0

        self._last_hash = current_hash

        self._latest_obs.mutex.release()


class AngleObservation(object):
    def __init__(self):
        self.tstamp = 0
        self.angles = None 
        self.mutex = Lock()

class AngleRecorder: 
    MAX_REPEATS = 100
    PRINT_AFTER = 20
    def __init__(self, topic_name):
        """
        :param topic_name:
        """
        self._finished_setup = False 
        self._get_repeats = 0
        self._obs_dim = 3 
        self._latest_obs = AngleObservation()

        self._is_first_status, self._status_sem = True, Semaphore(value=0)
        self._last_hash, self._num_repeats = None, 0
        self._last_hash_get_obs = None

        self._topic_name = topic_name
        rospy.Subscriber(topic_name, Float32MultiArray, self.store_latest_obs)
        logger = logging.getLogger('robot_logger')
        logger.debug('downing sema on topic: {}'.format(topic_name))
        success = self._status_sem.acquire(timeout=5)

        if not success:
            print('Still waiting for an data to arrive at angle recorder... Topic name:', self._topic_name)
            self._status_sem.acquire()
        logger.info(
            f"angle recorder at at {topic_name} subscribed")

        self._finished_setup = True 

    def valid_state(self): 
        if not self._finished_setup: 
            return False 
        with self._latest_obs.mutex: 
            most_recent_tstamp = self._latest_obs.tstamp
        return rospy.get_time() - most_recent_tstamp < 0.3
    
    def get_reading(self, arg=None):
        self._latest_obs.mutex.acquire()
        current_hash = hashlib.sha256(self._data.tostring()).hexdigest()
        if self._last_hash_get_obs is not None:
            if current_hash == self._last_hash_get_obs:
                self._get_repeats += 1 
                if self._get_repeats >= self.PRINT_AFTER: 
                    print(f'{self._get_repeats} repeated measurements for angle get_reading method!')
            else: 
                self._get_repeats = 0 
                
        self._last_hash_get_obs = current_hash
        reading = self._latest_obs.angles
        self._latest_obs.mutex.release()
        return reading
    

    def _proc_obs(self, latest_obsv, data):
        data = np.array(data.data).copy() 
        self._data = data
        assert len(data) % (self._obs_dim + 1) == 0
        latest_obsv.angles = data[:-1]
        latest_obsv.tstamp = data[-1] 

    @property
    def topic_name(self):
        return self._topic_name

    def store_latest_obs(self, data):
        # data = np.array(data)
        self._latest_obs.mutex.acquire()

        t0 = rospy.get_time()
        self._proc_obs(self._latest_obs, data)

        current_hash = hashlib.sha256(self._data.tostring()).hexdigest()
        if self._is_first_status:
            self._is_first_status = False
            self._status_sem.release()

        elif self._last_hash == current_hash:
            if self._num_repeats < self.MAX_REPEATS:
                self._num_repeats += 1
            else:
                logging.getLogger('robot_logger').error(f'Too many repeated measurements.\
                        Check angle topic {self._topic_name}!')
                self.errored = True 
                # go to neutral instead 
                # rospy.signal_shutdown('Too many repeated images. Check IMU!')
        else:
            self._num_repeats = 0

        self._last_hash = current_hash

        self._latest_obs.mutex.release()

        


if __name__ == '__main__':
    # from multicam_server.topic_utils import IMTopic

    rospy.init_node("camera_rec_test")

    rec = IMURecorder("/imu/imu_raw")

    r = rospy.Rate(2)  # 10hz
    start_time = rospy.get_time()
    for t in range(10):
        print('t{} before get image {}'.format(t, rospy.get_time() - start_time))
        t0 = rospy.get_time()
        reading = rec.get_reading()
        print(reading)
        print()
        # print('get image took', rospy.get_time() - t0)

        # im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        # t1 = rospy.get_time()
        # cv2.imwrite(os.environ['EXP'] + '/test_image_t{}_{}.jpg'.format(t, rospy.get_time() - start_time), im)
        # # print('save took ', rospy.get_time() - t1)

        r.sleep()
