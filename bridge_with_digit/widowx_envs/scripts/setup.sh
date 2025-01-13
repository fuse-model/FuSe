#!/bin/bash
set -e
if [[ -z "${ROBONETV2_ARM}" ]]; then
    echo 'Env variable "ROBONETV2_ARM" is not set. Please define it based on https://github.com/Interbotix/interbotix_ros_manipulators/tree/main/interbotix_ros_xsarms'
    echo 'For instance in case of WidowX 250 Robot Arm 6DOF use:'
    echo 'echo "export ROBONETV2_ARM=wx250s" >> ~/.bashrc && source ~/.bashrc'
    exit 1
fi

cd
if [ ! -f ".built" ]; then
    cd ~/interbotix_ws && catkin_make && touch ~/.built
fi
