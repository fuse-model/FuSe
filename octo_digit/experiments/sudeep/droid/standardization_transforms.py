from typing import Any, Dict

import tensorflow as tf
import tensorflow_graphics.geometry.transformation as tfg


def euler_to_rmat(euler):
    return tfg.rotation_matrix_3d.from_euler(euler)


def mat_to_rot6d(mat):
    r6 = mat[..., :2, :]
    r6_0, r6_1 = r6[..., 0, :], r6[..., 1, :]
    r6_flat = tf.concat([r6_0, r6_1], axis=-1)
    return r6_flat


def droid_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    T = trajectory["action_dict"]["cartesian_position"][:, :3]
    R = mat_to_rot6d(euler_to_rmat(trajectory["action_dict"]["cartesian_position"][:, 3:6]))
    trajectory["action"] = tf.concat(
        (
            T,
            R,
            trajectory["action_dict"]["gripper_position"],
        ),
        axis=-1,
    )
    trajectory["observation"]["proprio"] = tf.concat(
        (
            trajectory["observation"]["cartesian_position"],
            trajectory["observation"]["gripper_position"],
        ),
        axis=-1,
    )
    return trajectory


def droid_rel_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    dT = trajectory["action_dict"]["cartesian_velocity"][:, :3]
    dR = mat_to_rot6d(euler_to_rmat(trajectory["action_dict"]["cartesian_velocity"][:, 3:6]))
    trajectory["action"] = tf.concat(
        (
            dT,
            dR,
            trajectory["action_dict"]["gripper_position"],
        ),
        axis=-1,
    )
    trajectory["observation"]["proprio"] = tf.concat(
        (
            trajectory["observation"]["cartesian_position"],
            trajectory["observation"]["gripper_position"],
        ),
        axis=-1,
    )
    return trajectory
