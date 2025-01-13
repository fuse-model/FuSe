#!/usr/bin/env bash


read -r -p "Install ROS? [Y/n] " response
case "$response" in
  [nN][oO]|[nN])
    install_ros=false
    ;;
  *)
    install_ros=true
    ;;
esac

read -r -p "Install WidowX drivers? [Y/n] " response
case "$response" in
  [nN][oO]|[nN])
    install_widowx=false
    ;;
  *)
    install_widowx=true
    ;;
esac

read -r -p "Install Docker? [Y/n] " response
case "$response" in
  [nN][oO]|[nN])
    install_docker=false
    ;;
  *)
    install_docker=true
    read -r -p "Do you want to use Docker as a non-root user (without sudo)? [Y/n] " response
    case "$response" in
      [nN][oO]|[nN])
        docker_without_sudo=false
        ;;
      *)
        docker_without_sudo=true
        echo "Remember to log out and back in for this to take effect after the script completes!"
        if [[ $EUID -eq 0 ]]; then
           echo "If you want to set up Docker as a non-root user run this script $0 as non-root user." 
           exit 1
        fi
        ;;
    esac
    ;;
esac


read -r -p "Install nvidia-docker? [Y/n] " response
case "$response" in
  [nN][oO]|[nN])
    install_nv_docker=false
    ;;
  *)
    install_nv_docker=true
    echo "If you want to use Nvidia GPU in docker, please make sure that nvidia drivers are already installed. https://github.com/NVIDIA/nvidia-docker/wiki/Frequently-Asked-Questions#how-do-i-install-the-nvidia-driver"
    read -r -p "Do you want to continue? [Y/n] " response
    case "$response" in
      [nN][oO]|[nN])
        exit 0
    esac
    ;;
esac

read -r -p "Install docker-compose? [Y/n] " response
case "$response" in
  [nN][oO]|[nN])
    install_docker_compose=false
    ;;
  *)
    install_docker_compose=true
    ;;
esac

full_path=$(realpath $0)
dir_path=$(dirname $full_path)/host_install_scripts

if [ "$install_ros" = "true" ]; then
    $dir_path/ros_install.sh
fi
if [ "$install_widowx" = "true" ]; then
    $dir_path/widowx_install.sh
fi
if [ "$install_docker" = "true" ]; then
    $dir_path/docker_install.sh $docker_without_sudo
fi
if [ "$install_nv_docker" = "true" ]; then
    $dir_path/nvidia_docker_install.sh
fi
if [ "$install_docker_compose" = "true" ]; then
    $dir_path/docker_compose_install.sh
fi

echo "All done!"
