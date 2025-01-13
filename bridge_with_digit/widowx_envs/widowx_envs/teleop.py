#!/usr/bin/env python3

import argparse
import numpy as np
import cv2
from widowx_envs.widowx_env_service import WidowXClient, WidowXConfigs

def show_video(client, full_image=True):
    """
    This shows the video from the camera for a given duration.
    Full image is the image before resized to default 256x256.
    """
    res = client.get_observation()
    if res is None:
        print("No observation available... waiting")
        return

    if full_image:
        img = res["full_image"]
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    else:
        img = res["image"]
        img = (img.reshape(3, 256, 256).transpose(1, 2, 0) * 255).astype(np.uint8)
    cv2.imshow("Robot Camera", img)
    cv2.waitKey(20)  # 20 ms

print_yellow = lambda x: print("\033[93m {}\033[00m" .format(x))

def print_help():
    print_yellow("  Teleop Controls:")
    print_yellow("    w, s : move forward/backward")
    print_yellow("    a, d : move left/right")
    print_yellow("    z, c : move up/down")
    print_yellow("    i, k:  rotate yaw")
    print_yellow("    j, l:  rotate pitch")
    print_yellow("    n  m:  rotate roll")
    print_yellow("    space: toggle gripper")
    print_yellow("    r: reset robot")
    print_yellow("    q: quit")

def main():
    parser = argparse.ArgumentParser(description='Teleoperation for WidowX Robot')
    parser.add_argument('--ip', type=str, default='localhost')
    parser.add_argument('--port', type=int, default=5556)
    args = parser.parse_args()

    client = WidowXClient(host=args.ip, port=args.port)
    client.init(WidowXConfigs.DefaultEnvParams, image_size=256)

    print_help()
    cv2.namedWindow("Robot Camera")
    is_open = 1
    running = True
    while running:
        # Check for key press
        key = cv2.waitKey(100) & 0xFF

        # escape key to quit
        if key == ord('q'):
            print("Quitting teleoperation.")
            running = False
            continue

        # Handle key press for robot control
        # translation
        if key == ord('w'):
            client.step_action(np.array([0.01, 0, 0, 0, 0, 0, is_open]))
        elif key == ord('s'):
            client.step_action(np.array([-0.01, 0, 0, 0, 0, 0, is_open]))
        elif key == ord('a'):
            client.step_action(np.array([0, 0.01, 0, 0, 0, 0, is_open]))
        elif key == ord('d'):
            client.step_action(np.array([0, -0.01, 0, 0, 0, 0, is_open]))
        elif key == ord('z'):
            client.step_action(np.array([0, 0, 0.01, 0, 0, 0, is_open]))
        elif key == ord('c'):
            client.step_action(np.array([0, 0, -0.01, 0, 0, 0, is_open]))
        
        # rotation
        elif key == ord('i'):
            client.step_action(np.array([0, 0, 0, 0.01, 0, 0, is_open]))
        elif key == ord('k'):
            client.step_action(np.array([0, 0, 0, -0.01, 0, 0, is_open]))
        elif key == ord('j'):
            client.step_action(np.array([0, 0, 0, 0, 0.01, 0, is_open]))
        elif key == ord('l'):
            client.step_action(np.array([0, 0, 0, 0, -0.01, 0, is_open]))
        elif key == ord('n'):
            client.step_action(np.array([0, 0, 0, 0, 0, 0.01, is_open]))
        elif key == ord('m'):
            client.step_action(np.array([0, 0, 0, 0, 0, -0.01, is_open]))    
        
        # space bar to change gripper state
        elif key == ord(' '):
            is_open = 1 - is_open
            print("Gripper is now: ", is_open)
            client.step_action(np.array([0, 0, 0, 0, 0, 0, is_open]))
        elif key == ord('r'):
            print("Resetting robot...")
            client.reset()
            print_help()

        show_video(client)

    client.stop()  # Properly stop the client
    cv2.destroyAllWindows()
    print("Teleoperation ended.")

if __name__ == "__main__":
    main()
