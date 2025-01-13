import subprocess 
import sys



if __name__ == '__main__': 
    if len(sys.argv) == 1: 
        dev_name = 'enp114s0'
    else: 
        dev_name = sys.argv[1]
    ifconfig_out = str(subprocess.check_output(['ifconfig']))
    dev_only = ifconfig_out[ifconfig_out.index(dev_name):]
    inet_str = 'inet '
    dev_only = dev_only[dev_only.index(inet_str) + len(inet_str):]
    dev_only = dev_only[:dev_only.index(' ')]
    
    full_str = f"CURR_IP = '{dev_only}'"
    write_path = '/home/paulzhou/bridge_with_digit/widowx_envs/curr_ip.py'
    with open(write_path, 'w') as file: 
        file.write(full_str)
