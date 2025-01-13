FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8 TZ=America/Los_Angeles
ARG USER_ID=robonet
ARG DEBIAN_FRONTEND=noninteractive

SHELL ["/bin/bash", "-c"]

RUN apt-get update -y && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    python3-pip \
    python3-dev \
    vim \
    wget \
    curl \
    lsb-release \
    sudo \
    android-tools-adb \
    libglew-dev \
    patchelf \
    libosmesa6-dev \
    python3-venv \
    python3-cffi \
    v4l-utils \
    keyboard-configuration \
    tzdata \
    unzip \
    ffmpeg \
    git-lfs \
    alsa-base \
    alsa-utils \
    pulseaudio \
    libasound-dev \
    libportaudio2 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*


# Installs ROS
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
RUN apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654
RUN apt-get update -y && apt-get install -y \
    ros-noetic-ros-base \
    ros-noetic-realsense2-camera \
    ros-noetic-video-stream-opencv \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# based on https://askubuntu.com/a/1168971: enable running sudo commands as $USER_ID
RUN adduser --disabled-password --gecos '' ${USER_ID}
RUN adduser ${USER_ID} sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

# from now on run build commands as user $USER_ID
USER ${USER_ID}

# install interbotic_ros_arms repo
RUN curl 'https://raw.githubusercontent.com/Interbotix/interbotix_ros_manipulators/834edcbe6b955497d10ebc20ea9242c3f61e8fd1/interbotix_ros_xsarms/install/amd64/xsarm_amd64_install.sh' \
    > /tmp/xsarm_amd64_install.sh \
    && chmod +x /tmp/xsarm_amd64_install.sh \
    && echo "n" | /tmp/xsarm_amd64_install.sh \
    && rm /tmp/xsarm_amd64_install.sh
ENV PYTHONPATH="${PYTHONPATH}:~/interbotix_ws/src/interbotix_ros_toolboxes/interbotix_xs_toolbox"

# install python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN source /opt/ros/noetic/setup.bash && source ~/interbotix_ws/devel/setup.bash && python3 -m venv --system-site-packages ~/myenv
RUN source ~/myenv/bin/activate && pip install wheel && pip install -r /tmp/requirements.txt

# add widowx_envs code for build step
COPY . /home/${USER_ID}/widowx_envs
RUN ln -s ~/widowx_envs ~/interbotix_ws/src/
ENV PYTHONPATH="${PYTHONPATH}:/home/${USER_ID}/interbotix_ws/src/widowx_envs"

ENV ROBONETV2_ARM=wx250s
ENV DATA=/home/robonet/trainingdata
ENV EXP=/home/robonet/experiments
ENV REALSENSE_SERIAL=920312072629

# build interbox ros packages
RUN source /opt/ros/noetic/setup.bash \
    && source ~/interbotix_ws/devel/setup.bash \
    && cd ~/interbotix_ws \
    && catkin_make \
    && touch ~/.built

# add utility functions
RUN sudo ln -s ~/widowx_envs/scripts/go_to_sleep_pose.py /usr/local/bin/go_sleep
RUN sudo ln -s ~/widowx_envs/scripts/go_to_neutral_pose.py /usr/local/bin/go_neutral

RUN echo 'source ~/myenv/bin/activate' >> ~/.bashrc
RUN echo 'source /opt/ros/noetic/setup.bash' >> ~/.bashrc
RUN echo 'source ~/interbotix_ws/devel/setup.bash' >> ~/.bashrc

WORKDIR /home/${USER_ID}

# Clone and install the edgeml library
RUN git clone https://github.com/youliangtan/edgeml && \
    cd edgeml && \
    pip3 install -e .

# Create alias to python3 widowx_envs/widowx_envs/widowx_env_service.py
# This is a workaround for pip3 install -e .
RUN echo 'alias widowx_env_service="python3 ~/widowx_envs/widowx_envs/widowx_env_service.py"' >> ~/.bashrc
