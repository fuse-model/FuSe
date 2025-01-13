#! /usr/bin/env python3

import cv2
import rospy
import threading
import os
from std_msgs.msg import Float32MultiArray
from start_streamers import get_param
import sounddevice as sd 
import time
import numpy as np 
from numpy_ringbuffer import RingBuffer



##############################################################################


class MicStreamer:
    def __init__(self):
        self.parse_rosparam()
        self._buffer = RingBuffer(capacity=self._buffer_queue_size, dtype=float)
        self._lock = threading.Lock()
        self.start_capture()
        self.publisher = rospy.Publisher('/mic/mic_raw', Float32MultiArray, queue_size=1) 
        self.start_publishing()
    
    def setup_buffer(self): 
        self._buffer = RingBuffer(capacity = self._buffer_queue_size, dtype=float)
        self._buffer.append(0) # can't extend empty buffer for some reason
        self._buffer.extend(np.zeros(self._buffer_queue_size)) 


    def setup_capture_device(self):
        found_device = False 
        for device in sd.query_devices(): 
            if self._partial_name in device['name']: 
                found_device = True 
                break 
        success = found_device 
        if success: 
            sd.default.device = device['index']
        else: 
            raise ValueError
        return success

    def parse_rosparam(self):
        self._partial_name = get_param("~partial_name")
        self._sample_freq = get_param("~sample_freq") 
        self._block_time = get_param("~block_time")
        self._block_size = int(self._block_time * self._sample_freq)
        self._buffer_time = 0.3 # get_param("~buffer_time")
        self._buffer_queue_size = int(self._buffer_time * self._sample_freq)
        self._publish_freq = get_param("~publish_freq")

    def fill_buffer(
        self, indata: np.ndarray, frames: int, time, status
    ): 
        with self._lock: 
            self._buffer.extend(indata[:, 0])

    def start_capture(self):
        self._capture_thread = threading.Thread(target=self.setup_and_capture)
        self._capture_thread.start()

    def setup_and_capture(self):
        self._setup_success = self.setup_capture_device()
        if self._setup_success: 
            rospy.loginfo(f"[mic] mic set up successfully")
        self._stream = sd.InputStream(
            samplerate=self._sample_freq,
            blocksize=self._block_size,
            callback=self.fill_buffer
        )
        self.capture()

    def capture(self):
        rate = rospy.Rate(1)
        self._stream.start() 
        while not rospy.is_shutdown():
            rate.sleep()
         
    def publish_reading(self, reading):
        msg = Float32MultiArray(data=reading)
        self.publisher.publish(msg)

    def start_publishing(self):
        self._publishing_thread = threading.Thread(target=self.publishing)
        self._publishing_thread.start()

    def publishing(self):
        rate = rospy.Rate(self._publish_freq)
        while not rospy.is_shutdown():
            reading = None
            with self._lock:
                reading = np.array(self._buffer)
            self.publish_reading(reading)
            rate.sleep()


def main():
    rospy.init_node('streamer', anonymous=True)
    streamer = MicStreamer()
    rospy.spin()


##############################################################################

if __name__ == '__main__':
    main()
