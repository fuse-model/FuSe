#!/bin/bash

docker_without_sudo=$1
curl -fsSL https://get.docker.com -o /tmp/get-docker.sh || (echo "get-docker.sh download failed" && exit 1)
sudo sh /tmp/get-docker.sh || (echo "Docker install failed" && exit 1)
sudo systemctl --now enable docker || (echo "Docker startup failed" && exit 1)

if [ "$docker_without_sudo" = "true" ]; then
    if [[ $EUID -eq 0 ]]; then
           echo "If you want to set up Docker as a non-root user run this script $0 as non-root user." 
           exit 1
    fi
    sudo usermod -aG docker $USER
fi