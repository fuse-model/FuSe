import threading

import numpy as np
import rospy
import os
from threading import Lock, Semaphore
from cv_bridge import CvBridge
import cv2
from sensor_msgs.msg import Image as Image_msg
from sensor_msgs.msg import CameraInfo
import copy
import hashlib
import logging
import time


class LatestObservation(object):
    def __init__(self, create_tracker=False, save_buffer=False):
        self.img_cv2 = None
        self.tstamp_img = None
        self.img_msg = None
        self.mutex = Lock()
        if save_buffer:
            self.reset_saver()
        if create_tracker:
            self.reset_tracker()

    def reset_tracker(self):
        self.cv2_tracker = cv2.TrackerMIL_create()
        self.bbox = None
        self.track_itr = 0

    def reset_saver(self):
        self.save_itr = 0


class CameraRecorder:
    TRACK_SKIP = 2  # the camera publisher works at 60 FPS but camera itself only goes at 30
    MAX_REPEATS = 100  # camera errors after 10 repeated frames in a row

    def __init__(self, topic_data, opencv_tracking=False, save_videos=False):
        """
        :param topic_data:
        :param opencv_tracking:
        :param save_videos:
        :param server_id: if not None run in server mode. If the inference code requires python3 the camera server
        can still be launched in python2.
        """
        self._tracking_enabled, self._save_vides = opencv_tracking, save_videos

        self._latest_image = LatestObservation(self._tracking_enabled, self._save_vides)

        self._is_tracking = False
        if self._tracking_enabled:
            self.box_height = 80

        self.bridge = CvBridge()
        self._is_first_status, self._status_sem = True, Semaphore(value=0)
        self._cam_height, self._cam_width = None, None
        self._last_hash, self._num_repeats = None, 0
        self._last_hash_get_image = None
        if self._save_vides:
            self._buffers = []
            self._saving = False

        self._topic_data = topic_data
        self._image_dtype = topic_data.dtype
        self._topic_name = topic_data.name

        if topic_data.info_name is not None:
            self._camera_info_name = topic_data.info_name
        else:
            # Trys to guess the correct camera info name based on ROS conventions.
            # If this code is hanging, manually set the info_name in the conf file when defining the IMTopic
            self._camera_info_name = os.path.join(os.path.split(self._topic_name)[0], "camera_info")

        if topic_data.is_python_node:
            # FIXME: DIGIT must use the python streamer node, which currently does not publish info
            self._camera_info = None
        else:
            print("Trying to read camera info from ", self._camera_info_name)
            self._camera_info = rospy.wait_for_message(self._camera_info_name, CameraInfo)
            print("Successfully read camera info")

        rospy.Subscriber(topic_data.name, Image_msg, self.store_latest_im)
        logger = logging.getLogger('robot_logger')
        logger.debug('downing sema on topic: {}'.format(topic_data.name))
        success = self._status_sem.acquire(timeout=5)
        if not success:
            print('Still waiting for an image to arrive at CameraRecorder... Topic name:', self._topic_name)
            self._status_sem.acquire()
        logger.info(
            "Cameras {} subscribed: stream is {}x{}".format(self._topic_data.name, self._cam_width, self._cam_height))

    def _cam_start_tracking(self, lt_ob, point):
        lt_ob.reset_tracker()
        lt_ob.bbox = np.array([int(point[1] - self.box_height / 2.),
                               int(point[0] - self.box_height / 2.),
                               self.box_height, self.box_height]).astype(np.int64)

        lt_ob.cv2_tracker.init(lt_ob.img_cv2, tuple(lt_ob.bbox))
        lt_ob.track_itr = 0

    def start_tracking(self, start_points):
        assert self._tracking_enabled
        n_desig, xy_dim = start_points.shape
        if n_desig != 1:
            raise NotImplementedError("opencv_tracking requires 1 designated pixel")
        if xy_dim != 2:
            raise ValueError("Requires XY pixel location")

        self._latest_image.mutex.acquire()
        self._cam_start_tracking(self._latest_image, start_points[0])
        self._is_tracking = True
        self._latest_image.mutex.release()
        rospy.sleep(2)  # sleep a bit for first few messages to initialize tracker

        logging.getLogger('robot_logger').info("TRACKING INITIALIZED")

    def end_tracking(self):
        self._latest_image.mutex.acquire()
        self._is_tracking = False
        self._latest_image.reset_tracker()
        self._latest_image.mutex.release()

    def _bbox2point(self, bbox):
        point = np.array([int(bbox[1]), int(bbox[0])]) \
                + np.array([self.box_height / 2, self.box_height / 2])
        return point.astype(np.int32)

    def get_track(self):
        assert self._tracking_enabled, "OPENCV TRACKING IS NOT ENABLED"
        assert self._is_tracking, "RECORDER IS NOT TRACKING"

        points = np.zeros((1, 2), dtype=np.int64)
        self._latest_image.mutex.acquire()
        points[0] = self._bbox2point(self._latest_image.bbox)
        self._latest_image.mutex.release()

        return points.astype(np.int64)

    def get_image(self, arg=None):
        self._latest_image.mutex.acquire()
        time_stamp, img_cv2 = self._latest_image.tstamp_img, self._latest_image.img_cv2
        current_hash = hashlib.sha256(img_cv2.tostring()).hexdigest()
        if self._last_hash_get_image is not None:
            if current_hash == self._last_hash_get_image:
                # logging.getLogger('robot_logger').error('Repeated images, camera queried too fast!')
                # rospy.signal_shutdown('shutting down.')
                print(f'[{self._topic_name}] repeated images for get_image method!')
        self._last_hash_get_image = current_hash
        self._latest_image.mutex.release()
        return time_stamp, img_cv2

    def start_recording(self, reset_buffer=False):
        assert self._save_vides, "Video saving not enabled!"

        self._latest_image.mutex.acquire()
        if reset_buffer:
            self.reset_recording()
        self._saving = True
        self._latest_image.mutex.release()

    def stop_recording(self):
        assert self._save_vides, "Video saving not enabled!"
        self._latest_image.mutex.acquire()
        self._saving = False
        self._latest_image.mutex.release()

    def reset_recording(self):
        assert self._save_vides, "Video saving not enabled!"
        assert not self._saving, "Can't reset while saving (run stop_recording first)"

        old_buffers = self._buffers
        self._buffers = []
        self._latest_image.reset_saver()
        return old_buffers

    def _proc_image(self, latest_obsv, data):
        latest_obsv.img_msg = data
        latest_obsv.tstamp_img = data.header.stamp

        cv_image = self.bridge.imgmsg_to_cv2(data, self._image_dtype)
        if len(cv_image.shape) > 2 and cv_image.shape[2] > 3:
            cv_image = cv_image[:, :, :3]
        latest_obsv.img_cv2 = copy.deepcopy(self._topic_data.process_image(cv_image))

        if self._tracking_enabled and self._is_tracking:
            if latest_obsv.track_itr % self.TRACK_SKIP == 0:
                _, bbox = latest_obsv.cv2_tracker.update(latest_obsv.img_cv2)
                latest_obsv.bbox = np.array(bbox).astype(np.int32).reshape(-1)
            latest_obsv.track_itr += 1

    @property
    def topic_name(self):
        return self._topic_name

    def store_latest_im(self, data):
        self._latest_image.mutex.acquire()
        # if self._latest_image.tstamp_img is not None:
        #     print('{} delta t image {}'.format(self._topic_name, rospy.get_time() - self._latest_image.tstamp_img))

        t0 = rospy.get_time()
        self._proc_image(self._latest_image, data)

        current_hash = hashlib.sha256(self._latest_image.img_cv2.tostring()).hexdigest()
        if self._is_first_status:
            self._cam_height, self._cam_width = self._latest_image.img_cv2.shape[:2]
            self._is_first_status = False
            self._status_sem.release()
        elif self._last_hash == current_hash:
            if self._num_repeats < self.MAX_REPEATS:
                self._num_repeats += 1
            else:
                logging.getLogger('robot_logger').error(f'Too many repeated images.\
                    Check camera with topic {self._topic_name}!')
                rospy.signal_shutdown('Too many repeated images. Check camera!')
        else:
            self._num_repeats = 0
        self._last_hash = current_hash

        if self._save_vides and self._saving:
            if self._latest_image.save_itr % self.TRACK_SKIP == 0:
                self._buffers.append(copy.deepcopy(self._latest_image.img_cv2)[:, :, ::-1])
            self._latest_image.save_itr += 1
        self._latest_image.mutex.release()
        # print('time process image', rospy.get_time() - t0)

    @property
    def img_width(self):
        return self._cam_width

    @property
    def img_height(self):
        return self._cam_height

    @property
    def camera_info(self):
        return self._camera_info


class GlitchChecker(CameraRecorder):
    def __init__(self, topic_data, streamer, opencv_tracking=False, save_videos=False):
        super().__init__(topic_data, opencv_tracking, save_videos)
        self.streamer = streamer
        self._log_glitches = True
        self._glitch_log_dir = "/home/robonet/trainingdata/robonetv2/bridge_data_v2/glitches"

    def store_latest_im(self, data):
        super().store_latest_im(data)
        curr_image = self.get_image()[1]

        if (GlitchChecker.is_glitched(curr_image)
                and hasattr(self, "streamer")
                and not self.streamer.in_reset()):
            rospy.logerr(f"[{self.streamer.topic_name}] Glitch detected by watcher thread.")
            if self._log_glitches:
                self._glitch_name = f"{int(time.time() * 100)}.jpg"
                self._curr_image = curr_image
                self._save_sema = threading.Semaphore(value=0)
                save_thread = threading.Thread(target=self.save_image)
                save_thread.run()
            self.streamer.reset_connection(True)
            if self._log_glitches:
                self._save_sema.acquire()
            rospy.sleep(10)

    def save_image(self):
        cv2.imwrite(
            f"{self._glitch_log_dir}/{self._glitch_name}",
            self._curr_image[:, :, ::-1]
        )
        rospy.loginfo(
            f"[{self.streamer.topic_name}] Logged detected glitch to {self._glitch_log_dir}/{self._glitch_name}")
        self._save_sema.release()

    @classmethod
    def is_glitched(cls, img):
        grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        vert_gradient = cv2.Sobel(grayscale_img, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=3)
        vert_gradient = cv2.convertScaleAbs(vert_gradient)
        return (vert_gradient > 50).sum() > 50


if __name__ == '__main__':
    from multicam_server.topic_utils import IMTopic

    rospy.init_node("camera_rec_test")
    imtopic = IMTopic('/cam1/image_raw')
    rec = CameraRecorder(imtopic)

    r = rospy.Rate(5)  # 10hz
    start_time = rospy.get_time()
    for t in range(20):
        print('t{} before get image {}'.format(t, rospy.get_time() - start_time))
        t0 = rospy.get_time()
        tstamp, im = rec.get_image()
        # print('get image took', rospy.get_time() - t0)

        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        t1 = rospy.get_time()
        cv2.imwrite(os.environ['EXP'] + '/test_image_t{}_{}.jpg'.format(t, rospy.get_time() - start_time), im)
        # print('save took ', rospy.get_time() - t1)

        r.sleep()
