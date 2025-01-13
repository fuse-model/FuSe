from sklearn.linear_model import LinearRegression
import numpy as np
import time
from sklearn.preprocessing import PolynomialFeatures


def compute_robot_transformation_matrix(a, b):
    lr = LinearRegression(fit_intercept=False).fit(a, b)
    return lr.coef_.T


def convert_obs_to_image(obs, transpose=False):
    print("taking picture...")
    image = np.uint8(np.reshape(obs['image'] * 255, (3, 64, 64)))
    if transpose: image = np.transpose(image, (1, 2, 0))
    # print("image.shape", image.shape)
    return image


def rgb_to_robot_coords(rgb_coords, transmatrix):
    # add vector of 1s as feature to the pc_coords.
    assert len(rgb_coords.shape) <= 2
    if len(rgb_coords.shape) == 1:
        rgb_coords = np.array(rgb_coords[None])
    poly = PolynomialFeatures(2)
    rgb_coords = poly.fit_transform(rgb_coords)

    if transmatrix is not None:
        robot_coords = rgb_coords @ transmatrix
        return np.squeeze(robot_coords)

def get_image_obs(env, image_xyz=None, skip_move_to_neutral=False):
    joint_angles = env._controller.get_joint_angles()
    if image_xyz is None:
        if not skip_move_to_neutral:
            env.move_to_neutral(0.5)
        # else:
        #     env.reset()
    else:
        env.move_to_state(image_xyz, target_zangle=0, duration=0.5)
    time.sleep(0.2)  # wait for camera to catch up
    obs = env.current_obs()
    env._controller.set_joint_angles(joint_angles, 0.5)
    return obs


def get_image(env, transpose=True, image_xyz=None, skip_move_to_neutral=False):
    obs = get_image_obs(env, image_xyz, skip_move_to_neutral)
    return convert_obs_to_image(obs, transpose=transpose)


def execute_reach(env, reach_policy, reachpoint, noise=0.0):
    reach_policy.reset(reach_point=reachpoint)
    for i in range(6):
        action, _ = reach_policy.get_action()

        # noise
        noise_dims = 2
        noise_stds = [noise] * noise_dims + [0] * (len(action) - noise_dims)
        action = np.random.normal(loc=action, scale=noise_stds)
        action = np.clip(action, -1.0, 1.0)
        # import ipdb; ipdb.set_trace()
        obs, _, _, _ = env.step(action)
    return obs
