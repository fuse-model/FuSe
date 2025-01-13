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


class IMUStreamer:
    def __init__(self):
        self.num_readings = 0
        self._num_errors = 0 
        self._reset_every = 10 
        self._obs_dim = 4
        self._print_every = 1000 # print FPS stats to logger 
        self.parse_rosparam()
        self.setup_buffer()
        self._lock = threading.Lock()
        self._start_time = time.time() 
        self.start_capture()
        self.publisher = rospy.Publisher('/imu/imu_raw', Float32MultiArray, queue_size=1)
        self.start_publishing()
        self._max_time_diff = self._sample_freq * self._buffer_queue_size
        
    
    def setup_buffer(self): 
        self._buffer = RingBuffer(capacity = self._obs_dim * self._buffer_queue_size, dtype=float)
        self._buffer.append(0)
        self._buffer.extend(np.zeros(self._obs_dim * self._buffer_queue_size))

    def do_setup(self): 
        try: 
            found_valid_serial = False 
            for id in self._uart_ids: 
                try: 
                    uart = serial.Serial(id,  baudrate=115200)
                    self.port = uart 
                    self.sensor = adafruit_bno055.BNO055_UART(uart)
                    found_valid_serial = True 
                    break
                except serial.serialutil.SerialException: # wrong ID 
                    continue 
            if not found_valid_serial: 
                rospy.logerr('IMU serial not found, try plugging in again.')
            reading = self.sensor.temperature
            success = True 
            self._num_errors = 0
        except Exception as e: 
            success = False 
            rospy.logerr(f"Failed to set up IMU with error: {e}")
            time.sleep(0.1)
        return success

    def setup_capture_device(self):
        setup_ctr = 0
        setup_success = False
        while not setup_success: 
            setup_success = self.do_setup()
            if not setup_success: 
                rospy.loginfo(f"[imu] IMU setup failed on attempt {setup_ctr + 1}")
                setup_ctr += 1 
        rospy.loginfo(f"[imu] IMU setup complete. Took {setup_ctr+1} tries.")
        return setup_success

    def parse_rosparam(self):
        self._uart_ids = ['/dev/ttyUSB1', '/dev/ttyUSB2']
        self._sample_freq = 50. # overestimate, fix in post
        self._buffer_time = 0.3
        self._buffer_queue_size = int(self._buffer_time * self._sample_freq) # buffer_time is length of data record, in s
        self._publish_freq = 30. # get_param("~publish_freq")
        self.port = None 

    # def extend_buffer(self): 
    #     while not rospy.is_shutdown(): 
    #         with self._lock: 
    #             self._buffer.

    def start_capture(self):
        self._capture_thread = threading.Thread(target=self.setup_and_capture)
        self._capture_thread.start()


    def setup_and_capture(self):
        self._setup_success = self.setup_capture_device()
        self.capture()

    def get_reading(self):
        try: 
            # euler =  np.array(self.sensor.euler)
            # gyro = np.array(self.sensor.gyro)
            # linear_acceleration = np.array(self.sensor.linear_acceleration)
            # euler = np.array(self.sensor.euler)
            euler = self.sensor.euler
            time_reading = time.time() % 100000
            time_reading = (time_reading,)
            # magnetic = np.array(self.sensor.magnetic)
            # temperature = np.array([self.sensor.temperature for _ in range(3)])
            # timestamp = np.array([time.time_ns() for _ in range(3)])
            # euler = np.array(self.sensor.euler)
            # gyro = np.array(self.sensor.gyro)
            # linear_acceleration = np.array(self.sensor.linear_acceleration)
            # magnetic = np.array(self.sensor.magnetic)
            # temperature = np.array([self.sensor.temperature for _ in range(3)])
            # timestamp = np.array([time.time_ns() for _ in range(3)])
            self.num_readings += 1 
        except Exception as e: # UART read error
            rospy.logerr(f"[imu] Error during attempted reading:   {e}")
            rospy.logerr(f'{self._num_errors} errors.')
            self._num_errors += 1
            if self._num_errors == 20: 
                self.port.close() 
                time.sleep(0.1)
                rospy.logerr("\n\n[imu] Retrying setup...\n\n")
                self.setup_capture_device()
            return self.get_reading()
        # reading = np.concatenate([euler, gyro, linear_acceleration, magnetic, temperature, timestamp])
        # reading = np.zeros(18,)
        reading = np.array(euler + time_reading)
        return reading 

    def capture(self):
        # time_between_readings = 1.0 / (2 * self._sample_freq)
        time_between_readings = 1.0 / 120.0
        most_recent_reading_time = time.time() 
        num_captures = 0 
        num_adds = 0
        start_time = time.time() 
        with self._lock: 
            cycle_time = start_time 
        cycle_time = start_time

        num_caps_in_last_cycle = 0 
        while not rospy.is_shutdown():
            reading = self.get_reading()
            curr_time = time.time() 
            # if curr_time - most_recent_reading_time > time_between_readings:  
            with self._lock: 
                self._buffer.extend(reading)
            num_adds += 1 
            most_recent_reading_time = curr_time
            num_captures += 1 
            if num_captures % 200 == 0: 
                dt = time.time() - start_time 
                print(f'Capture rate: {num_captures/(dt)}     extend rate:   {num_adds / dt}')
            


    def publish_reading(self, data):
        msg = Float32MultiArray(data=np.array(data))
        self.publisher.publish(msg)

    def start_publishing(self):
        self._publishing_thread = threading.Thread(target=self.publishing)
        self._publishing_thread.start()

    def publishing(self):
        rate = rospy.Rate(self._publish_freq)
        print_every = 100
        print_ctr = 0 
        reading = np.zeros(self._buffer_queue_size)
        curr_first_time = 0
        while not rospy.is_shutdown():
            reading = None
            with self._lock:
               reading = np.array(self._buffer)
            if reading is not None:
                self.publish_reading(reading)
                if (print_ctr + 1) % print_every == 0: 
                    curr_time = time.time() 
                    delta_time = curr_time - self._start_time
                    delta_buffer = reading[-1] - reading[3]
                    rospy.loginfo(f'[imu] {self.num_readings} readings taken over {delta_time} time, for a true rate of {self.num_readings/delta_time}.')
                    rospy.loginfo(f'[imu] Time diff between current time and time at start of buffer = {delta_buffer}\n')
                print_ctr = (print_ctr + 1) % print_every
        
            rate.sleep()
        rospy.loginfo('[imu] closing IMU port...')
        self.port.close() 


def main():
    rospy.init_node('IMU_streamer')
    streamer = IMUStreamer()
    rospy.spin()
    # id = '/dev/ttyUSB1'
    # uart = serial.Serial(id,  baudrate=115200)
    # sensor = adafruit_bno055.BNO055_UART(uart)
    # start_time = time.time()
    # for i in range(2000): 
    #     e = sensor.euler
    #     a = sensor.linear_acceleration 
    #     g = sensor.gyro
    # end_time = time.time() 
    # print(f'delta t = {end_time - start_time}, freq = {2000/(end_time - start_time)}')



##############################################################################

if __name__ == '__main__':
    main()


