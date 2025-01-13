# Bridge Data Robot

Code for controlling Trossen WidowX robot arms.

- `widowx_envs`: contains the `widowx_envs` Python package with all of the WidowX controller code.
- `docker_compose.yml`: contains all of the docker-compose services that will be used to run the robot.

## Setup

First, install the dependencies on the host machine by running `./host_install.sh`.

Next, we need to build and launch the ROS service that communicates with the robot. This service is defined by the `robonet` entry in `docker-compose.yml`. It uses the `robonet-base` image which is built from the `widowx_envs` directory (see `widowx_envs/Dockerfile`). To build and run the `robonet-base` service, run:

```bash
# first generate the usb config file
./generate_usb_config.sh

# build and run the robonet service
USB_CONNECTOR_CHART=$(pwd)/usb_connector_chart.yml docker compose up --build robonet
```

This builds the `robonet-base` image, which contains all of the ROS dependencies and the Python controller code from the `widowx_envs` directory. The USB connector chart is required to start the camera stream. You can get the USB device IDs by running `v4l2-ctl --list-devices`, `./generate_usb_config.sh` automatically generates the config file for you.

Once this is running, you can execute commands in the running container like so:

```bash
docker compose exec robonet bash -lic "go_sleep"
```

Explanation:
- `docker compose exec`: execute a command in a running container
- `robonet`: the service name (as specified in `docker-compose.yml`)
- `bash`: the executable to run inside the container
- `-l`: tells bash to open a login shell, sourcing `~/.bashrc` inside the container, which is required to set up a few ROS things and the correct Python virtual environment
- `-i`: makes the shell interactive, in case you want to run an interactive command (like `python`)
- `-c`: tells bash to execute the next argument
- `go_sleep`: the string to be executed by bash; in this case, it's a utility script that is built in to the `robonet-base` image that moves the arm to the sleep position

If you really want to, you can also attach a bash shell interactively using `docker compose exec robonet bash`.

### Using RealSense cameras

The RealSense cameras require different drivers than RGB cameras.  If you are using RealSenses, change the `camera_string` in `widowx_envs/scripts/run.sh` to `realsense:=true`.

You will also need to update the device serial number in `widowx_envs/widowx_controller/launch/widowx_rs.launch` to match your cameras.

## Data collection

First, make sure the `robonet-base` container is running using the above command. Then, run the following commands:

```bash
# first create an empty directory to store the data
mkdir -p $HOME/widowx_data

# give sudo write access to the container
# we can check the id by running `id` in the container
sudo chown -R 1000:1002 $HOME/widowx_data

# access the container
docker compose exec robonet bash

# start the data collection script
python widowx_envs/widowx_envs/run_data_collection.py widowx_envs/experiments/bridge_data_v2/conf.py
```

You can specify a different directory to save the data in `docker-compose.yml`.

At this point, the data_collection script will start initializing, and then throw the error:
```bash
Device not found. Make sure that device is running and is connected over USB
Run `adb devices` to verify that the device is visible.
```

This is expected, as our data collection requires the use of a Meta Quest VR headset to control the widowx arm. Turn on and connect the VR headset to the computer via USB. Make sure USB debugging is enabled and answer "Yes/Agree" to any prompts that appear in the headset. 

## Evaluating policies

There are two ways to interface a policy with the robot: (1) the docker compose service method or (2) the (newer) server-client method. In general, we recommend the server-client method.

### Docker compose service method

The first method is to install the neccesary code and dependencies for running the policy in the same environment as the robot dependencies. This is done by creating a new docker compose service in `docker-compose.yml`. 

See `docker-compose.yml` for an example with the [bridge_data_v2](https://github.com/rail-berkeley/bridge_data_v2) codebase. Each codebase also needs a minimal Dockerfile that builds on top of the `robonet-base` image. The Dockerfile for the `bridge_data_v2` codebase looks like:

```Dockerfile
FROM robonet-base:latest

COPY requirements.txt /tmp/requirements.txt
RUN ~/myenv/bin/pip install -r /tmp/requirements.txt
RUN ~/myenv/bin/pip install --upgrade "jax[cuda11_pip]==0.4.13" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# this path will get mounted as a volume (see `docker-compose.yml`)
ENV PYTHONPATH=${PYTHONPATH}:/home/robonet/code/bridge_data_v2

# modify packages to work with python 3.8 (ros noetic needs python 3.8)
# to avoid orbax checkpoint error, downgrade flax 
RUN ~/myenv/bin/pip install flax==0.6.11
# to avoid typing errors, upgrade distrax
RUN ~/myenv/bin/pip install distrax==0.1.3

# avoid git safe directory errors
RUN git config --global --add safe.directory /home/robonet/code/bridge_data_v2

WORKDIR /home/robonet/code/bridge_data_v2
```

To run the service, clone the [bridge_data_v2](https://github.com/rail-berkeley/bridge_data_v2) repo into `bridge_data_robot/code`.

Then, build the new service/container:

```bash
docker compose build bridge_data_v2
```

Now, we can run commands in this container similar to the previous section. We just need to make sure the`robonet-base` container is running in the background (see the setup section).

For instance, to run the `bridge_data_v2` evaluation script:

```bash
docker compose run bridge_data_v2 bash -lic "python experiments/eval_gc.py ..."
```

To execute commands in the docker container with an interactive shell, you can again do `docker compose run bridge_data_v2 bash`.

### Server-client method

With this setup, the robot is run as a server that receives actions and the policy acts as a client that sends actions. This "server-client" architecture allows us to both isolate robot controller and policy dependencies, as well as perform inference on a separate machine from the one used to control the robot (though in the simplest case, both the robot and policy can run on the same machine).

First, run the server on the robot:

```bash
docker compose exec robonet bash -lic "widowx_env_service --server"
```

Then, we'll create an environment to run the client. This can be on the same machine as the robot or a different machine. First, create a new python environment (e.g conda) with the policy dependencies (e.g [bridge_data_v2](https://github.com/rail-berkeley/bridge_data_v2)). Additionally, install the [edgeml](https://github.com/youliangtan/edgeml) library. Finally, install the `widowx_envs` package from this repo:

```bash
cd widowx_envs
pip install -e .
```

Now, we are ready to communicate with the server. To verify that everything is set up correctly run:

```bash
python widowx_envs/widowx_env_service.py --client
```

This command assumes the server is running on `localhost` (i.e the same machine). If you're running the client on a different machine from the server, use `--ip` to specify the ip address of the server.

The robot should execute a few predefined movements. See the `bridge_data_v2` evaluation script [here](https://github.com/rail-berkeley/bridge_data_v2/blob/main/experiments/eval.py) for an example of how to send actions to the server. 

Extra Util: To try teleop the robot arm remotely
```bash
python3 widowx_envs/widowx_envs/teleop.py --ip <IP_ADDRESS>
```

## Troubleshooting

##### Permission errors

If you run into following errors:

```bash
Traceback (most recent call last):
  File "urllib3/connectionpool.py", line 677, in urlopen
  File "urllib3/connectionpool.py", line 392, in _make_request
  File "http/client.py", line 1277, in request
  File "http/client.py", line 1323, in _send_request
  File "http/client.py", line 1272, in endheaders
  File "http/client.py", line 1032, in _send_output
  File "http/client.py", line 972, in send
  File "docker/transport/unixconn.py", line 43, in connect
PermissionError: [Errno 13] Permission denied
```
that can be fixed by running the following commands and subsequently restarting the PC (the log out and log back in is sometimes not sufficient):

```bash
sudo groupadd docker
sudo usermod -aG docker $USER
```
