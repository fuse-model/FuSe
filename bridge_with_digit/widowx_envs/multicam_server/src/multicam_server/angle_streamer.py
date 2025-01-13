#! /usr/bin/env python3
# best working version 
import cv2
import rospy
import threading
import os
from std_msgs.msg import Float32MultiArray
from start_streamers import get_param
import adafruit_bno055
import serial
import time
import numpy as np 
from numpy_ringbuffer import RingBuffer
import rospy 
from functools import partial


##############################################################################


class AngleStreamer:
    def __init__(self, env, publish_freq=40., read_freq=40.):
        self.env = env 
        self._publish_freq = publish_freq 
        self._read_freq = read_freq
        self._obs_dim = 3

        self._most_recent_reading = None 
        self._lock = threading.Lock()
        self._angle_lock = threading.Lock() 
        self._angle_sema = threading.Semaphore(0)
        self.start_capture()
        self.publisher = rospy.Publisher('/joints/rpy', Float32MultiArray, queue_size=1)
        self.start_publishing()

    def start_capture(self):
        self._capture_thread = threading.Thread(target=self.capture)
        self._capture_thread.start()
    
    def capture(self):
        rate = rospy.Rate(self._read_freq)
        while not rospy.is_shutdown(): 
            full_state = self.env.get_full_state() 
            rpy = full_state[3:6]
            with self._lock: 
                self._most_recent_reading = np.concatenate((rpy, np.array([rospy.get_time()])))
            rate.sleep()
            
    def publish_reading(self, data): 
        msg = Float32MultiArray(data=data)
        self.publisher.publish(msg)
    


    def start_publishing(self):
        self._publishing_thread = threading.Thread(target=self.publishing)
        self._publishing_thread.start()

    def publishing(self):
        rate = rospy.Rate(self._publish_freq)
        while not rospy.is_shutdown():
            reading = None
            with self._lock:
               reading = self._most_recent_reading
            if reading is not None:
                self.publish_reading(reading)        
            rate.sleep()




def main():
    # rospy.init_node('Angle_streamer')
    streamer = AngleStreamer()
    # rospy.spin()
