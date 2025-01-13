#! /usr/bin/env python3


import os
import yaml
import re
import shutil
import subprocess
import rospy

##############################################################################

def get_param(parameter_name):
    if not rospy.has_param(parameter_name):
        rospy.logerr('Parameter %s not provided. Exiting.', parameter_name)
        exit(1)
    return rospy.get_param(parameter_name)


def load_connector_chart():
    config_path = get_param("~camera_connector_chart")
    if not os.path.exists(config_path):
        rospy.logerr(
            f"The usb connector chart in path {config_path} does not exist. You can use the example usb_connector_chart_example.yml as a template.")
        rospy.logerr("you can find the usb outlets of the webcams by running $v4l2-ctl --list-devices")
        exit(1)
    return yaml.load(open(config_path, 'r'), Loader=yaml.CLoader)


def get_dev(output_string, usb_id):
    lines = output_string.decode().split('\n')
    for i, line in enumerate(lines):
        if usb_id in line:
            return re.search('video(\d+)', lines[i + 1]).group(1)
    rospy.logerr('usb_id {} not found!'.format(usb_id))
    return None


def reset_usb(reset_names):
    if shutil.which('usbreset') is None:
        current_dir = os.path.dirname(os.path.realpath(__file__))
        res = subprocess.call(f'gcc {current_dir}/usbreset.c -o /usr/local/bin/usbreset')
        if not res == 0:
            rospy.logerr(f'usbreset install exit code: {res}')
            raise ValueError('could not install usbreset !')
    res = subprocess.run(['lsusb'], stdout=subprocess.PIPE)
    lines = res.stdout.decode().split('\n')
    for line in lines:
        for name in reset_names:
            if name in line:
                numbers = re.findall(r'(\d\d\d)', line)[:2]
                rospy.loginfo('resetting usb with lsusb line: {}'.format(line))
                cmd = 'sudo usbreset /dev/bus/usb/{}/{}'.format(*numbers)
                res = subprocess.call(cmd.split())
                if not res == 0:
                    rospy.logerr(f'exit code: {res}')
                    raise ValueError('could not reset !')


def process_camera_connector_chart():
    connector_chart_dict = load_connector_chart()

    res = subprocess.run(['v4l2-ctl', '--list-devices'], stdout=subprocess.PIPE)
    output_string = res.stdout

    providers = []
    topic_names = []
    for topic_name, usb_id in connector_chart_dict.items():
        if usb_id[:4] != 'usb-': 
            continue # either mic or IMU, handle separately 
        dev_number = get_dev(output_string, usb_id)
        providers.append(dev_number)
        topic_names.append(topic_name)
    return providers, topic_names


def populate_params():
    params = {}
    params['fps'] = get_param("~fps")
    params['frame_id'] = get_param("~frame_id")
    params['retry_on_fail'] = get_param("~retry_on_fail")
    params['buffer_queue_size'] = get_param("~buffer_queue_size")
    params['python_node'] = get_param("~python_node")
    params['camera_connector_chart'] = get_param("~camera_connector_chart")
    return params


def populate_digit_params():
    params = {}
    params['python_node'] = True
    params['is_digit'] = True
    params['retry_on_fail'] = True
    return params

def populate_wrist_params(): 
    params = {}
    params['python_node'] = True
    params['retry_on_fail'] = True
    return params

def populate_imu_params(): 
    params = {} 
    params['sample_freq'] = get_param('~imu_sample_freq')
    params['buffer_time'] = get_param('~imu_buffer_time')
    params['publish_freq'] = get_param('~imu_publish_freq')
    params['node_name'] = 'imu_streamer'
    return params 
    

def populate_mic_params(): 
    params = {} 
    params['publish_freq'] = get_param('~mic_publish_freq')
    params['sample_freq'] = get_param('~mic_sample_freq')
    params['block_time'] = get_param('~mic_block_time')
    params['buffer_time'] = get_param('~mic_buffer_time')
    params['node_name'] = 'mic_streamer'
    return params 


##############################################################################

def main():
    base_call = "roslaunch multicam_server streamer.launch"
    rospy.init_node('start_streamers', anonymous=True)  # node name is provided in the roslaunch file
    topic_names = []
    if get_param("~camera_connector_chart"):
        video_stream_providers, topic_names = process_camera_connector_chart()
    else:
        video_stream_provider = get_param("~video_stream_provider")
        parsed_video_stream_provider = eval(video_stream_provider)
        if isinstance(parsed_video_stream_provider, list):
            video_stream_providers = parsed_video_stream_provider
        elif isinstance(parsed_video_stream_provider, int):
            video_stream_providers = [parsed_video_stream_provider]
        else:
            rospy.logerr("Pass either list or an integer as video_stream_provider to video_stream_opencv node.")
            rospy.loginfo(f"Arguments provided: {video_stream_provider}")
        for i in range(len(video_stream_providers)):
            topic_names.append(f'camera{i}')

    processes = []
    for index, [video_stream_provider, topic_name] in enumerate(zip(video_stream_providers, topic_names)):
        full_params = {'video_stream_provider': video_stream_provider, 'camera_name': topic_name,
                       'node_name': f'streamer_{index}'}
        full_params.update(populate_params())
        if "digit" in topic_name.lower():
            full_params.update(populate_digit_params())
        elif "wrist" in topic_name.lower(): 
            full_params.update(populate_wrist_params())
        appended_string = ''
        for key, val in full_params.items():
            appended_string += key + ':=' + str(val) + ' '
        proc = subprocess.Popen((base_call + ' ' + appended_string).split())
        processes.append(proc)

    connector_chart_dict = load_connector_chart()
    # start IMU streamer 
    if 'imu' in connector_chart_dict: 
        full_params = {'uart_id': connector_chart_dict['imu']}
        full_params.update(populate_imu_params())
        base_imu_call = "roslaunch multicam_server imu_streamer.launch"
        appended_string = ''
        for key, val in full_params.items():
            appended_string += key + ':=' + str(val) + ' '
        proc = subprocess.Popen((base_imu_call + ' ' + appended_string).split())
        processes.append(proc)

    # start mic streamer 
    if 'mic' in connector_chart_dict: 
        full_params = {'partial_name': connector_chart_dict['mic']}
        full_params.update(populate_mic_params())
        base_mic_call = "roslaunch multicam_server mic_streamer.launch"
        appended_string = ''
        for key, val in full_params.items():
            appended_string += key + ':=' + str(val) + ' '
        proc = subprocess.Popen((base_mic_call + ' ' + appended_string).split())
        processes.append(proc)

    # stream video over socket 
    if get_param('~stream_video'): 
        full_params = {} # TODO: add support for arguments
        base_video_call = "roslaunch multicam_server video_streamer.launch"
        appended_string = ''
        for key, val in full_params.items():
            appended_string += key + ':=' + str(val) + ' '
        proc = subprocess.Popen((base_video_call + ' ' + appended_string).split())
        processes.append(proc)

        full_params = {} # TODO: add support for arguments
        base_video_call = "roslaunch multicam_server overhead_streamer.launch"
        appended_string = ''
        for key, val in full_params.items():
            appended_string += key + ':=' + str(val) + ' '
        proc = subprocess.Popen((base_video_call + ' ' + appended_string).split())
        processes.append(proc)


    while not rospy.is_shutdown():
        rospy.sleep(0.1)
    for proc in processes:
        proc.kill()
        proc.communicate()

if __name__ == '__main__':
    main()
