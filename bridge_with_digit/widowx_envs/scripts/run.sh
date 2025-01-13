#!/bin/bash

bash $(dirname "$0")/setup.sh || exit 1

python_node_string='python_node:=false'
camera_string='realsense:=false'

source /opt/ros/noetic/setup.bash
source ~/interbotix_ws/devel/setup.bash
source ~/myenv/bin/activate

# using 'exec' here is very important because roslaunch needs to do some cleanup after it exits
# so when the container is killed the SIGTERM needs to be passed to roslaunch
exec roslaunch widowx_controller widowx_rs.launch \
    ${video_stream_provider_string} camera_connector_chart:=/tmp/camera_connector_chart \
    serial_no_camera1:=${REALSENSE_SERIAL} \
    python_node:=false realsense:=true
