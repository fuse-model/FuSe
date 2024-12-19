"""Filter functions for R2D2 curation."""
from typing import Any, Dict

import tensorflow as tf
import numpy as np
import json
import os
from collections import Counter

try:
    # load array of loss per trajectory index averaged across multiple pre-trained models
    # we will use this for filtering
    LOSS_PER_TRAJ = np.load("/nfs/nfs2/users/karl/code/orca/avg_losses.npy")
except:
    pass

try:
    # load scene IDs for scene filtering
    with tf.io.gfile.GFile("gs://rail-orca-central2/r2_d2/r2d2_metadata_240125.json", "r") as F:
        metadata = json.load(F)
    scene_ids = []
    for sample in metadata:
        scene_ids.append(metadata[sample]["scene_id"])
    c = Counter(scene_ids)
    most_common_ids = [e for e, _ in c.most_common()]
except:
    pass

try:
    # load verb_objects for task filtering
    with tf.io.gfile.GFile("gs://rail-orca-central2/r2_d2/r2d2_verb_object.json", "r") as F:
        verb_object_data = json.load(F)
    verb_objects = []
    for sample in verb_object_data:
        if (
            verb_object_data[sample]["verb"] is None
            or verb_object_data[sample]["object"] is None
        ):
            continue
        verb_objects.append(
            verb_object_data[sample]["verb"] + "_" + verb_object_data[sample]["object"]
        )
    cvo = Counter(verb_objects)
    most_common_verb_objects = [e for e, _ in cvo.most_common()]
except:
    pass


def filter_success(trajectory: Dict[str, Any]):
    # only keep trajectories that have "success" in the file path
    return tf.strings.regex_full_match(
        trajectory['traj_metadata']['episode_metadata']['file_path'][0],
        ".*/success/.*"
    )


def filter_loss_q09(trajectory: Dict[str, Any]):
    traj_index = trajectory["_traj_index"][0]
    traj_index = tf.minimum(traj_index, LOSS_PER_TRAJ.shape[0] - 1)
    return tf.gather(LOSS_PER_TRAJ, traj_index) < np.quantile(LOSS_PER_TRAJ, 0.9)


def filter_loss_q07(trajectory: Dict[str, Any]):
    traj_index = trajectory["_traj_index"][0]
    traj_index = tf.minimum(traj_index, LOSS_PER_TRAJ.shape[0] - 1)
    return tf.gather(LOSS_PER_TRAJ, traj_index) < np.quantile(LOSS_PER_TRAJ, 0.7)


def filter_scene_02(trajectory: Dict[str, Any]):

    @tf.py_function(Tout=tf.bool)
    def is_valid_traj(tf_file_path):
        file_path = tf_file_path.numpy().decode()
        key = os.path.join(*file_path.split('/')[-4:])
        scene_id = metadata[key]["scene_id"]
        # filter -1 IDs and only keep data from most common 50% of scenes
        return scene_id >= 0 and most_common_ids.index(scene_id) < int(0.2 * len(most_common_ids))

    return is_valid_traj(trajectory["traj_metadata"]["episode_metadata"]["file_path"][0])


def filter_task_50(trajectory: Dict[str, Any]):

    @tf.py_function(Tout=tf.bool)
    def is_valid_traj(tf_language_instruction):
        language_instruction = tf_language_instruction.numpy().decode()
        if not language_instruction or language_instruction not in verb_object_data:
            return False
        verb_obj = verb_object_data[language_instruction]
        if verb_obj["verb"] is None or verb_obj["object"] is None:
            return False
        task = verb_obj["verb"] + '_' + verb_obj["object"]
        return task in most_common_verb_objects[:50]

    return is_valid_traj(
        trajectory["language_instruction"][0])


def filter_task_100(trajectory: Dict[str, Any]):

    @tf.py_function(Tout=tf.bool)
    def is_valid_traj(tf_language_instruction):
        language_instruction = tf_language_instruction.numpy().decode()
        if not language_instruction or language_instruction not in verb_object_data:
            return False
        verb_obj = verb_object_data[language_instruction]
        if verb_obj["verb"] is None or verb_obj["object"] is None:
            return False
        task = verb_obj["verb"] + '_' + verb_obj["object"]
        return task in most_common_verb_objects[:100]

    return is_valid_traj(
        trajectory["language_instruction"][0])


def filter_task_200(trajectory: Dict[str, Any]):

    @tf.py_function(Tout=tf.bool)
    def is_valid_traj(tf_language_instruction):
        language_instruction = tf_language_instruction.numpy().decode()
        if not language_instruction or language_instruction not in verb_object_data:
            return False
        verb_obj = verb_object_data[language_instruction]
        if verb_obj["verb"] is None or verb_obj["object"] is None:
            return False
        task = verb_obj["verb"] + '_' + verb_obj["object"]
        return task in most_common_verb_objects[:200]

    return is_valid_traj(
        trajectory["language_instruction"][0])


def filter_skill_8(trajectory: Dict[str, Any]):
    KEY_SKILLS = ['put', 'move', 'pick', 'remove', 'take', 'place', 'open', 'close']

    @tf.py_function(Tout=tf.bool)
    def is_valid_traj(tf_language_instruction):
        language_instruction = tf_language_instruction.numpy().decode()
        if not language_instruction or language_instruction not in verb_object_data:
            return False
        verb_obj = verb_object_data[language_instruction]
        if verb_obj["verb"] is None or verb_obj["verb"] not in KEY_SKILLS:
            return False
        return True

    return is_valid_traj(
        trajectory["language_instruction"][0])


def filter_viewpoint_10k(trajectory: Dict[str, Any]):
    CENTER = np.array([-0.08976049, -0.35126469,  0.41580593])
    MAX_DIST = 0.16405667178427544

    ROT_CENTER = np.array([-1.73736128, -0.07302387, -1.31426051])
    ROT_MAX_DIST = 0.5705628578515318

    @tf.py_function(Tout=tf.bool)
    def is_valid_traj(tf_file_path):
        file_path = tf_file_path.numpy().decode()
        key = os.path.join(*file_path.split('/')[-4:])
        extrinsics_1 = metadata[key]["ext1_cam_extrinsics"]
        extrinsics_2 = metadata[key]["ext2_cam_extrinsics"]
        # filter anything that's not close enough to the defined target camera location
        return (
            (
                np.linalg.norm(extrinsics_1[:3] - CENTER) < MAX_DIST
                and np.linalg.norm(extrinsics_1[3:] - ROT_CENTER) < ROT_MAX_DIST
            ) or (
                np.linalg.norm(extrinsics_2[:3] - CENTER) < MAX_DIST
                and np.linalg.norm(extrinsics_2[3:] - ROT_CENTER) < ROT_MAX_DIST
            )
        )

    return is_valid_traj(trajectory["traj_metadata"]["episode_metadata"]["file_path"][0])
