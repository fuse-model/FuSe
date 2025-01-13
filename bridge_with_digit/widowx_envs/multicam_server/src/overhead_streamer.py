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
from multicam_server.topic_utils import IMTopic
import sys 
from PIL import Image
import socket 


############################################################################## 
from socket_config import HOST, CHECK_PORT
DEFAULT_PORT = CHECK_PORT # The port used by the server
############################################################################## 


class CheckStreamer: 
    def __init__(self,
        socket, 
        publish_freq = 30. 
    ): 
        self._topic = IMTopic('/blue/image_raw')
        # self._topic = IMTopic('/wrist/image_raw')
        self.socket = socket 
        self.thread_exception = None
        self.recorder = CameraRecorder(self._topic)
        self._publish_freq = publish_freq
        self._lock = threading.Lock()
        self._most_recent_msg = None 
        self.run_all()

    
    def run_all(self): 
        rospy.loginfo(f'[setup checker] Listening for client requests...')
        prev_timeout = self.socket.gettimeout()
        self.socket.settimeout(None)
        self.client, address = self.socket.accept()
        self.socket.settimeout(prev_timeout)
        self.thread_exception = None 
        self.publish_message_size()
        rospy.loginfo(f'[setup checker] socket set up to send {self._msg_len}-byte messages.')
        self.start_capturing()
        self.start_publishing()

    def get_image(self): 
        return self.recorder.get_image()[1]  

    def get_observation(self): 
        return self.get_image()

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
    port = DEFAULT_PORT
    rospy.init_node('check_streamer')
    while not rospy.is_shutdown(): 
        data_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        data_socket.bind((HOST, port))
        rospy.loginfo(f'[check_streamer] Transmitting on PORT={port}')
        data_socket.listen()
        streamer = CheckStreamer(data_socket)
        data_socket.close()
        del data_socket
        rospy.logerr(f'[check_streamer] Streaming pipe has broken. Attempting to reinitialize...')
    

if __name__ == '__main__': 
    main()

        
