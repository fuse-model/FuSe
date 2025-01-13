#!/bin/bash -eu

# The BSD License
# Copyright (c) 2018 PickNik Consulting
# Copyright (c) 2014 OROCA and ROS Korea Users Group

#set -x

function usage {
    # Print out usage of this script.
    echo >&2 "usage: $0"
    echo >&2 "[-h|--help] Print help message."
    exit 0
}

# Parse command line. If the number of argument differs from what is expected, call `usage` function.
OPT=`getopt -o h -l help -- $*`
eval set -- $OPT
while [ -n "$1" ] ; do
    case $1 in
        -h|--help) usage ;;
        --) shift; break;;
        *) echo "Unknown option($1)"; usage;;
    esac
done

if ! command -v lsb_release &> /dev/null
then
    sudo apt-get install lsb-release
fi

version=`lsb_release -sc`
echo ""
echo "INSTALLING ROS --------------------------------"
echo ""
echo "Checking the Ubuntu version"
case $version in
  "trusty" | "xenial" | "bionic" | "focal")
  ;;
  *)
    echo "ERROR: This script will only work on Trusty / Xenial / Bionic / Focal. Exit."
    exit 0
esac

case $version in
  "trusty")
  ROS_DISTRO="indigo"
  ;;
  "xenial")
  ROS_DISTRO="kinetic"
  ;;
  "bionic")
  ROS_DISTRO="melodic"
  ;;
  "focal")
  ROS_DISTRO="noetic"
  ;;
esac

relesenum=`grep DISTRIB_DESCRIPTION /etc/*-release | awk -F 'Ubuntu ' '{print $2}' | awk -F ' LTS' '{print $1}'`
if [ "$relesenum" = "14.04.2" ]
then
  echo "Your ubuntu version is $relesenum"
  echo "Install the libgl1-mesa-dev-lts-utopic package to solve the dependency issues for the ROS installation specifically on $relesenum"
  sudo apt-get install -y libgl1-mesa-dev-lts-utopic
else
  echo "Your ubuntu version is $relesenum"
fi

echo "Add the ROS repository"
if [ ! -e /etc/apt/sources.list.d/ros-latest.list ]; then
  sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
fi

echo "Download the ROS keys"
sudo apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654

sudo apt update

echo "Installing ROS $ROS_DISTRO"

# Support for Python 3 in Noetic
if [ "$ROS_DISTRO" = "noetic" ]
then
  sudo apt install -y \
  python3-rosdep \
  python3-rosinstall \
  python3-bloom \
  python3-rosclean \
  python3-wstool \
  python3-pip \
  python3-catkin-lint \
  python3-catkin-tools \
  python3-rosinstall \
  ros-$ROS_DISTRO-desktop-full
else
  sudo apt install -y \
  python-rosdep \
  python-rosinstall \
  python-bloom \
  python-rosclean \
  python-wstool \
  python-pip \
  python-catkin-lint \
  python-catkin-tools \
  python-rosinstall \
  ros-$ROS_DISTRO-desktop-full
fi

# Only init if it has not already been done before
if [ ! -e /etc/ros/rosdep/sources.list.d/20-default.list ]; then
  sudo rosdep init
fi
rosdep update

if ! command -v roscore &> /dev/null
then
  echo "source /opt/ros/$ROS_DISTRO/setup.bash" >> ~/.bashrc
fi

echo "Done installing ROS"

exit 0
