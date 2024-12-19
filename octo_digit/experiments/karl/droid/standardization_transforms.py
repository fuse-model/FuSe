"""Episode transforms for custom (non-OXE) RLDS datasets to canonical dataset definition."""
from typing import Any, Dict

import json
import os
import tensorflow as tf
import numpy as np
import tensorflow_graphics.geometry.transformation as tfg

try:
    # load scene IDs for scene filtering
    with tf.io.gfile.GFile("gs://rail-orca-central2/r2_d2/r2d2_metadata_240125.json", "r") as F:
        metadata = json.load(F)
except:
    pass


def rmat_to_euler(rot_mat):
    return tfg.euler.from_rotation_matrix(rot_mat)


def euler_to_rmat(euler):
    return tfg.rotation_matrix_3d.from_euler(euler)


def invert_rmat(rot_mat):
    return tfg.rotation_matrix_3d.inverse(rot_mat)


def mat_to_rot6d(mat):
    r6 = mat[..., :2, :]
    r6_0, r6_1 = r6[..., 0, :], r6[..., 1, :]
    r6_flat = tf.concat([r6_0, r6_1], axis=-1)
    return r6_flat


def change_velocity_act_frame(velocity, frame):
    R_frame = euler_to_rmat(frame[:, 3:6])
    R_frame_inv = invert_rmat(R_frame)

    # world to wrist: dT_pi = R^-1 dT_rbt
    vel_t = (R_frame_inv @ velocity[:, :3][..., None])[..., 0]

    # world to wrist: dR_pi = R^-1 dR_rbt R
    dR = euler_to_rmat(velocity[:,3:6])
    dR = R_frame_inv @ (dR @ R_frame)
    dR_r6 = mat_to_rot6d(dR)
    return tf.concat([vel_t, dR_r6], axis=-1)


def change_state_act_frame(pos_acs, frame):
    xyz_0 = frame[...,:3]
    R_0 = euler_to_rmat(frame[...,3:6])
    R_0_inv = invert_rmat(R_0)

    delta_xyz = (pos_acs[...,:3] - xyz_0)[..., None]
    xyz = (R_0_inv @ delta_xyz)[..., 0]
    R = R_0_inv @ euler_to_rmat(pos_acs[...,3:6])
    R6 = mat_to_rot6d(R)
    return tf.concat([xyz, R6], axis=-1)


@tf.py_function(Tout=tf.bool)
def is_not_swapped(tf_file_path):
    file_path = tf_file_path.numpy().decode()
    key = os.path.join(*file_path.split('/')[-4:])
    extrinsics_1 = metadata[key]["ext1_cam_extrinsics"]
    extrinsics_2 = metadata[key]["ext2_cam_extrinsics"]
    return extrinsics_1[1] < extrinsics_2[1]


def maybe_swap_exterior_images(img1, img2, file_path):
    return tf.cond(
        is_not_swapped(
            file_path
        ),
        lambda: (img1, img2),
        lambda: (img2, img1)
    )


def droid_baseact_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # every input feature is batched, ie has leading batch dimension
    dt = trajectory["action_dict"]["cartesian_velocity"][:, :3]
    dR = mat_to_rot6d(euler_to_rmat(trajectory["action_dict"]["cartesian_velocity"][:, 3:6]))
    trajectory["action"] = tf.concat(
        (
            dt,
            dR,
            trajectory["action_dict"]["gripper_position"],
        ),
        axis=-1,
    )
    trajectory["observation"]["exterior_image_1_left"], trajectory["observation"]["exterior_image_2_left"] = (
        maybe_swap_exterior_images(
            trajectory["observation"]["exterior_image_1_left"],
            trajectory["observation"]["exterior_image_2_left"],
            trajectory["traj_metadata"]["episode_metadata"]["file_path"][0]
        )
    )
    trajectory["observation"]["proprio"] = tf.concat(
        (
            trajectory["observation"]["cartesian_position"],
            trajectory["observation"]["gripper_position"],
        ),
        axis=-1,
    )
    return trajectory


def droid_wristact_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # every input feature is batched, ie has leading batch dimension
    wrist_act = change_velocity_act_frame(
        trajectory["action_dict"]["cartesian_velocity"],
        trajectory["observation"]["cartesian_position"]
    )
    trajectory["action"] = tf.concat(
        (
            wrist_act,
            trajectory["action_dict"]["gripper_position"],
        ),
        axis=-1,
    )
    trajectory["observation"]["exterior_image_1_left"], trajectory["observation"]["exterior_image_2_left"] = (
        maybe_swap_exterior_images(
            trajectory["observation"]["exterior_image_1_left"],
            trajectory["observation"]["exterior_image_2_left"],
            trajectory["traj_metadata"]["episode_metadata"]["file_path"][0]
        )
    )
    trajectory["observation"]["proprio"] = tf.concat(
        (
            trajectory["observation"]["cartesian_position"],
            trajectory["observation"]["gripper_position"],
        ),
        axis=-1,
    )
    return trajectory


def droid_cumulative_wristact_transform(
        trajectory: Dict[str, Any],
        action_horizon: int
) -> Dict[str, Any]:
    # chunk input actions
    actions = tf.concat(
        (
            trajectory["action_dict"]["cartesian_position"],
            trajectory["action_dict"]["gripper_position"],
        ),
        axis=-1,
    )
    traj_len = tf.shape(actions)[0]
    action_chunk_indices = tf.range(traj_len)[:, None] + tf.range(
        action_horizon
    )  # [traj_len, action_horizon]
    # repeat the last action at the end of the trajectory rather than going out of bounds
    action_chunk_indices = tf.minimum(action_chunk_indices, traj_len - 1)
    # gather
    actions = tf.gather(
        actions, action_chunk_indices
    )  # [traj_len, action_horizon, action_dim]
    cartesian_pos_actions, gripper_actions = actions[..., :6], actions[..., 6:]

    # compute cumulative-delta actions in wrist frame for non-gripper dimensions
    wrist_act = change_state_act_frame(
        cartesian_pos_actions,
        trajectory["observation"]["cartesian_position"][:, None]
    )
    trajectory["action"] = tf.concat(
        (
            wrist_act,
            gripper_actions,
        ),
        axis=-1,
    )
    trajectory["observation"]["exterior_image_1_left"], trajectory["observation"]["exterior_image_2_left"] = (
        maybe_swap_exterior_images(
            trajectory["observation"]["exterior_image_1_left"],
            trajectory["observation"]["exterior_image_2_left"],
            trajectory["traj_metadata"]["episode_metadata"]["file_path"][0]
        )
    )
    trajectory["observation"]["proprio"] = tf.concat(
        (
            trajectory["observation"]["cartesian_position"],
            trajectory["observation"]["gripper_position"],
        ),
        axis=-1,
    )
    return trajectory
