#!/bin/bash

wget -q https://raw.githubusercontent.com/Interbotix/interbotix_ros_core/main/interbotix_ros_xseries/interbotix_xs_sdk/99-interbotix-udev.rules -O /tmp/99-interbotix-udev.rules || (echo "interbotix-udev.rules download failed" && exit 1)
sudo cp /tmp/99-interbotix-udev.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules && sudo udevadm trigger
