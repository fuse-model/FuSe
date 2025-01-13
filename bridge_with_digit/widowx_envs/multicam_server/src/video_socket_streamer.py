#! /usr/bin/env python3

from base64 import encode
from ctypes import Union
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
import struct 
import socket 
import pickle 
from multicam_server.camera_recorder import CameraRecorder
from multicam_server.sensor_recorder import JointPosRecorder, AngleRecorder
from multicam_server.topic_utils import IMTopic


############################################################################## 
from socket_config import HOST, VIDEO_PORT
PORT = VIDEO_PORT # The port used by the server

IMTOPIC_ARG_NAMES = ['name', 'is_python_node'] 

############################################################################## 

class VideoSocketStreamer:
    def __init__(self, 
        socket: socket.socket, 
        # im_topic: IMTopic, 
        publish_freq: float = 30.0
    ):

        self.thread_exception = None
        self.socket = socket 
        
        imtopic_args = {key: get_param(f'~{key}') for key in IMTOPIC_ARG_NAMES}
        
        self._topic = IMTopic(**imtopic_args)

        self.recorder = CameraRecorder(self._topic)

        self.joint_recorder = JointPosRecorder('/wx250s/joint_states')

        self._most_recent_angle = None 
        self._angle_setup_success = False 
        self._angle_recorder = None 

        self._publish_freq = publish_freq
        self._lock = threading.Lock()
        self._most_recent_msg = None 
        self.run_all()
    
 

    def run_all(self): 
        rospy.loginfo(f'[video streamer] Listening for client requests...')
        prev_timeout = self.socket.gettimeout()
        self.socket.settimeout(None)
        self.client, address = self.socket.accept()
        self.client.settimeout(None)
        # self.socket.settimeout(prev_timeout)
        self.thread_exception = None 
        self.publish_message_size()
        rospy.loginfo(f'[video streamer] socket set up to send {self._msg_len}-byte messages.')
        self.start_capturing()
        self.start_capturing_angle()
        self.start_publishing()

    def get_image(self): 
        return self.recorder.get_image()[1]  

    def get_joint_state(self): 
        return self.joint_recorder.get_reading() 
    
    def angle_valid_state(self): 
        return self._angle_setup_success #  and self._angle_recorder.valid_state()
    
    def get_angle(self): 
        angle = None 
        if self.angle_valid_state(): 
            angle =  self._angle_recorder.get_reading() 
            angles_valid = 1 
        else:
            angle =  np.zeros(3,)
            angles_valid = 0
        angle_and_metadata = np.concatenate((angle, np.array([angles_valid])))
        return angle_and_metadata

    def get_observation(self): 
        return (self.get_image(), self.get_joint_state(), self.get_angle())

    def publish_message_size(self):
        message = self.get_observation()
        encoded_message = self.encode(message)
        self._msg_len = len(encoded_message) 
        msg = struct.pack('>I', self._msg_len)
        self.client.sendall(msg)

    def publish_observation(self, encoded_message):
        try: 
            self.client.sendall(encoded_message)
        except (BrokenPipeError, ConnectionResetError) as e:
            self.thread_exception = e 

    def encode(self, image): 
        return pickle.dumps(image)

    def start_capturing(self): 
        self._capture_thread = threading.Thread(target=self.capture, daemon=True)
        self._capture_thread.start()
    
    def capture(self): 
        rate = rospy.Rate(self._publish_freq)
        try: 
            while not rospy.is_shutdown() and self.thread_exception is None: 
                encoded_message = self.encode(self.get_observation())
                with self._lock: 
                    self._most_recent_msg = encoded_message
                rate.sleep()
        except TypeError as e: 
            with self._lock: 
                self.thread_exception = e
        except KeyboardInterrupt as e: 
            with self._lock: 
                self.thread_exception = e

    def start_capturing_angle(self): 
        self._angle_capture_thread = threading.Thread(target=self.setup_and_capture_angle, daemon=True)
        self._angle_capture_thread.start()

    def setup_and_capture_angle(self): 
        self._angle_recorder = AngleRecorder('/joints/rpy')
        self._angle_setup_success = True




    def start_publishing(self):
        # self._publishing_thread = threading.Thread(target=self.publishing, daemon=True)
        # self._publishing_thread.start()
        self.publishing()

    def publishing(self):
        rate = rospy.Rate(self._publish_freq)
        try: 
            while not rospy.is_shutdown() and self.thread_exception is None:
                msg = None 
                with self._lock:
                    if self._most_recent_msg is not None: 
                        msg = self._most_recent_msg
                if msg is not None:
                    self.publish_observation(msg)
                rate.sleep()
        except KeyboardInterrupt as e: 
            with self._lock: 
                self.thread_exception = e 



def main():
    rospy.init_node('video_streamer')
    while not rospy.is_shutdown(): 
        data_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        found_unused_port = False 
        port = PORT 
        time.sleep(1)
        data_socket.bind((HOST, port))
        rospy.loginfo(f'[video_streamer] Transmitting on PORT={port}')
        data_socket.listen()
        streamer = VideoSocketStreamer(data_socket)
        data_socket.close()
        del data_socket
        rospy.logerr(f'[video_streamer] Streaming pipe has broken. Attempting to reinitialize...')

##############################################################################

if __name__ == '__main__':
    main()
