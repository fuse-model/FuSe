#! /usr/bin/env python3

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


############################################################################## 
from socket_config import HOST, IMU_PORT
PORT = IMU_PORT

############################################################################## 

class IMUSocketServer:
    def __init__(self, socket):
        self.thread_exception = None
        self.socket = socket 
        self._publish_freq = 60.
        self._lock = threading.Lock()
        self.publisher = rospy.Publisher('/imu/imu_raw', Float32MultiArray, queue_size=1)
        self._buffer = None 
        self.run_all()
 

    def run_all(self): 
        rospy.loginfo(f'[IMU streamer] Listening for client requests...')
        self.client, address = self.socket.accept()
        self.thread_exception = None 
        raw_msglen = self._recvall(4)
        self._msg_len = struct.unpack('>I', raw_msglen)[0]
        if self._msg_len == 0: # exit_signal 
            rospy.loginfo(f'[IMU streamer] Received msg_len = 0 (termination signal). Killig main process')
        else: 
            rospy.loginfo(f'[IMU streamer] socket set up to receive {self._msg_len}-byte messages.')
            self.start_capturing()
            self.start_publishing()

    def start_capturing(self): 
        self._capture_thread = threading.Thread(target=self.capture, daemon=True)
        self._capture_thread.start()
    
    def capture(self): 
        rate = rospy.Rate(self._publish_freq)
        try: 
            while not rospy.is_shutdown(): 
                buf = self.receive_buffer()
                if self.thread_exception is None: 
                    with self._lock: 
                        self._buffer = buf 
                    rate.sleep()
                else: 
                    break 
        except TypeError as e: 
            with self._lock: 
                self.thread_exception = e
        except KeyboardInterrupt as e: 
            with self._lock: 
                self.thread_exception = e


    def receive_buffer(self): 
        buffer = self._recvall(self._msg_len)
        return pickle.loads(buffer)
    
    def _recvall(self, n):
        data = bytearray()
        while len(data) < n:
            packet = None
            try: 
                packet = self.client.recv(n - len(data))
            except (BrokenPipeError, ConnectionResetError) as e:
                self.thread_exception = e 
                break 
                
            if not packet:
                return None
            data.extend(packet)
        return data

    def publish_reading(self, data):
        msg = Float32MultiArray(data=np.array(data))
        self.publisher.publish(msg)

    def start_publishing(self):
        # self._publishing_thread = threading.Thread(target=self.publishing, daemon=True)
        # self._publishing_thread.start()
        self.publishing()

    def publishing(self):
        rate = rospy.Rate(self._publish_freq)
        try: 
            while not rospy.is_shutdown() and self.thread_exception is None:
                reading = None
                with self._lock:
                    if self._buffer is not None: 
                        reading = self._buffer.copy() 
                if reading is not None:
                    self.publish_reading(reading)
                rate.sleep()
        except KeyboardInterrupt as e: 
            with self._lock: 
                self.thread_exception = e 



def main():
    rospy.init_node('IMU_streamer')

    # data_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # data_socket.bind((HOST, PORT))
    # data_socket.listen()

    # rospy.loginfo('[imu] listening for client signaler requests')
    # # client, address = signal_socket.accept()
    # streamer = IMUSocketServer(data_socket)
    # rospy.spin()
    

    while not rospy.is_shutdown(): 
        data_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        data_socket.bind((HOST, PORT))
        rospy.loginfo(f'[IMU streamer] Transmitting on PORT={PORT}')
        data_socket.listen()
        streamer = IMUSocketServer(data_socket)
        data_socket.close()
        del data_socket
        time.sleep(5)
        rospy.logerr(f'[IMU streamer] Streaming pipe has broken. Attempting to reinitialize...')

##############################################################################

if __name__ == '__main__':
    main()
