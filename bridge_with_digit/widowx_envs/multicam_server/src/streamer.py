#! /usr/bin/env python3

import cv2
import rospy
import threading
import os
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from cv_bridge import CvBridge
from multicam_server.camera_recorder import GlitchChecker
from multicam_server.topic_utils import IMTopic
from start_streamers import process_camera_connector_chart, get_param
from functools import partial 


##############################################################################


class Streamer:
    def __init__(self):
        self.parse_rosparam()
        self._buffer = []
        self.bridge = CvBridge()
        self._lock = threading.Lock()
        self._reset_flag = False
        self._setup_success = False
        self._in_reset = False
        self._setup_sema = threading.Semaphore(value=0)
        self.start_capture()
        self._setup_sema.acquire()
        if not self._setup_success:
            return
        self.publisher = rospy.Publisher(self._topic_name + '/image_raw', Image, queue_size=1)
        self.start_publishing()
        if self._is_digit:
            self.start_checking_reset()

    def get_provider_name(self):
        provider = None
        providers, topic_names = process_camera_connector_chart()
        if self._topic_name in topic_names:
            provider = providers[topic_names.index(self._topic_name)]
        return provider

    def setup_capture_device(self):
        self._in_reset = True
        success = True
        self._video_stream_provider = self.get_provider_name()
        if self._video_stream_provider:
            self.full_resource_path = "/dev/video" + str(self._video_stream_provider)
        else:
            success = False
            rospy.logerr(f"[{self._topic_name}] Could not find device corresponding to {self._topic_name}")

        if success:
            success = os.path.exists(self.full_resource_path)
            if not success:
                rospy.logerr(f"[{self._topic_name}] Device '{self.full_resource_path}' does not exist.")

        if success:
            rospy.loginfo(f"[{self._topic_name}] Trying to open resource: '{self.full_resource_path}' "
                          f"for topic '{self._topic_name}'")
            self.cap = cv2.VideoCapture(self.full_resource_path)
            if not self.cap.isOpened():
                rospy.logerr(f"[{self._topic_name}] Error opening resource: {self.full_resource_path}")
                rospy.loginfo("opencv VideoCapture can't open it.")
                rospy.loginfo(f"The device '{self.full_resource_path}' is possibly in use. You could try reconnecting "
                              f"the camera.")
                success = False
            else: 
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
                self.cap.set(cv2.CAP_PROP_FPS, self._fps)


        if success:
            ret, frame = self.cap.read()
            if not ret or ('wrist' in self._topic_name and not frame.any()):
                rospy.logerr(f"[{self._topic_name}] Failed to read image, despite capture being open. Please "
                             f"disconnect the device then try again.")
                success = False


        if success and self._is_digit:
            for _ in range(300):
                ret, frame = self.cap.read()
            if not ret:
                success = False
            else:
                pass 
                # if GlitchChecker.is_glitched(frame):  # Likely glitched error bars at bottom of display
                #     rospy.logerr(f"[{self._topic_name}] Discontinuous glitch detected. Attempting reset.")
                #     read_success = False
                #     for i in range(self._num_glitch_reset_tries):
                #         self.cap.release()
                #         rospy.sleep(1.)
                #         self.cap = cv2.VideoCapture(self.full_resource_path)
                #         open_success = self.cap.isOpened()
                #         if open_success:
                #             for _ in range(300):
                #                 ret, frame = self.cap.read()
                #         else:
                #             ret, frame = False, None
                #         open_success = ret
                #         if not open_success:
                #             rospy.logerr(
                #                 f"[{self._topic_name}] Failed to read image. Please "
                #                 f"disconnect the device then try again.")
                #             break
                #         read_success = not GlitchChecker.is_glitched(frame)
                #         if read_success:
                #             rospy.loginfo(f"[{self._topic_name}] DIGIT successfully initialized after {i + 2} tries.")
                #             break
                #     if not read_success:
                #         success = False
                #         rospy.logerr(f"[{self._topic_name}] Unable to reset DIGIT digitally. Please disconnect the "
                #                      f"device, then try again.")
        if success:
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            rospy.loginfo(f"[{self._topic_name}] Setup successful. Camera running at {fps} fps "
                          f"at resolution {width}x{height}")

        if not success and self._retry_on_fail:
            self.cap.release()
            rospy.loginfo(f"[{self._topic_name}] Setup failed. Please unplug and then plug in again.")
            rospy.sleep(3.)
            rospy.loginfo(f"[{self._topic_name}] Retrying setup for {self._topic_name}")
            success = self.setup_capture_device()
        
        self._in_reset = False
        return success

    def parse_rosparam(self):
        self._fps = get_param("~fps")
        self._width = get_param("~width")
        self._height = get_param("~height")
        self._frame_id = get_param("~frame_id")
        self._retry_on_fail = get_param("~retry_on_fail")
        self._buffer_queue_size = get_param("~buffer_queue_size")
        self._topic_name = get_param("~camera_name")
        self._video_stream_provider = get_param("~video_stream_provider")
        self._is_digit = get_param("~is_digit")
        self._num_glitch_reset_tries = get_param("~num_glitch_reset_tries")

        if self._is_digit: 
            if get_param("~digit_format") == "VGA": 
                self._fps = 30 
                self._width = 640 
                self._height = 480 
            else: 
                self._fps = 30 
                self._width = 320 
                self._height = 480

    def start_capture(self):
        self._capture_thread = threading.Thread(target=self.setup_and_capture)
        self._capture_thread.start()

    def setup_and_capture(self):
        self._setup_success = self.setup_capture_device()
        while self._retry_on_fail and not self._setup_success:
            self.setup_capture_device()
        self._setup_sema.release()
        self.capture()

    def capture(self):
        # running at full speed the camera allows
        while not rospy.is_shutdown():
            if not self._reset_flag:
                rval, frame = self.cap.read()
                if not rval:
                    rospy.logwarn(f"[{self._topic_name}] The frame has not been captured. You could try reconnecting "
                                  f"the camera resource {self.full_resource_path}.")
                    rospy.sleep(3)
                    if self._retry_on_fail:
                        rospy.loginfo(f"[{self._topic_name}] Retrying setup...")
                        self.setup_capture_device()
                else:
                    reading = [frame, rospy.Time(secs=rospy.get_time())]
                    with self._lock:
                        while (len(self._buffer) > self._buffer_queue_size):
                            self._buffer.pop(0)
                        self._buffer.append(reading)
            else:
                self.cap.release()
                self.setup_capture_device()
                self._setup_sema.release()
                self._reset_flag = False

        self.cap.release()

    def publish_image(self, reading):
        img = reading[0]
        time = reading[1]
        imgmsg = self.bridge.cv2_to_imgmsg(img, 'bgr8')
        imgmsg.header.frame_id = self._frame_id
        imgmsg.header.stamp = time
        self.publisher.publish(imgmsg)

    def start_publishing(self):
        self._publishing_thread = threading.Thread(target=self.publishing)
        self._publishing_thread.start()

    def publishing(self):
        rate = rospy.Rate(self._fps)
        while not rospy.is_shutdown():
            reading = None
            with self._lock:
                if self._buffer:
                    reading = self._buffer[0]
                    self._buffer.pop(0)
            if reading is not None:
                self.publish_image(reading)
            rate.sleep()

    def start_checking_reset(self):
        self._checking_thread = threading.Thread(target=self.check_reset)
        self._checking_thread.start()

    def check_reset(self):
        self._reset_subscriber = rospy.Subscriber(
            self._topic_name + '/image_raw/reset_flag',
            Bool,
            self.reset_connection,
            queue_size=1,
        )
        topic = IMTopic(self._topic_name + '/image_raw/', is_python_node=True)
        # self._recorder = GlitchChecker(topic, self)

    def reset_connection(self, should_reset):
        if should_reset and not self._reset_flag:
            rospy.logerr(f"[{self._topic_name}] Resetting camera {self._topic_name}...")
            self._setup_sema = threading.Semaphore(value=0)
            self._reset_flag = True
            self._setup_sema.acquire()
            self._reset_flag = False

    def in_reset(self):
        return self._reset_flag or self._in_reset

    @property
    def topic_name(self):
        return self._topic_name



def main():
    rospy.init_node('streamer', anonymous=True)
    streamer = Streamer()
    rospy.spin()

##############################################################################

if __name__ == '__main__':
    main()