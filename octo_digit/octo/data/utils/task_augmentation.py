"""
Contains basic logic for randomly zero-ing out keys in the task specification.
"""

import pickle

import tensorflow as tf

from octo.data.utils.data_utils import to_padding
from octo.utils.fuse_utils import FuseRephraser


def delete_and_rephrase(
    traj, pickle_file_path: str, rephrase_prob: float, keep_image_prob: float
):
    traj = rephrase_instruction(traj, pickle_file_path, rephrase_prob)
    traj = delete_task_conditioning(traj, keep_image_prob)
    return traj


class Rephraser:
    def create_static_hash_table(self, dictionary):
        """Takes a python dictionary with string keys and values and creates a tf static hash table"""
        keys = list(dictionary.keys())
        values = list(dictionary.values())
        initializer = tf.lookup.KeyValueTensorInitializer(
            keys, values, key_dtype=tf.string, value_dtype=tf.string
        )
        hash_table = tf.lookup.StaticHashTable(initializer, default_value="")
        return hash_table

    def __init__(self, pickle_file_path: str):
        if isinstance(pickle_file_path, str):
            with tf.io.gfile.GFile(pickle_file_path, "rb") as file:
                lang_paraphrases = pickle.load(file)
                # Create StaticHashTable
                self.rephrase_lookup = self.create_static_hash_table(lang_paraphrases)


def rephrase_instruction(
    traj: dict, pickle_file_path: str, rephrase_prob: float
) -> dict:
    """Randomly rephrases language instructions with precomputed paraphrases
    Args:
       traj: A dictionary containing trajectory data. Should have a "task" key.
       pickle_file_path: The path to the pickle file containing the paraphrases.
       rephrase_prob: The probability of augmenting the language instruction. The probability of keeping the language
           instruction is 1 - rephrase_prob.
    """
    rephraser = Rephraser(pickle_file_path)

    if "language_instruction" not in traj["task"]:
        return traj
    original_language = traj["task"]["language_instruction"]
    # check the language key is not empty
    string_is_not_empty = tf.reduce_all(tf.strings.length(original_language) > 0)
    # check dict is not empty
    dict_is_not_empty = bool(rephraser.rephrase_lookup)
    if dict_is_not_empty and string_is_not_empty:
        rephrased_instruction = rephraser.rephrase_lookup.lookup(original_language[0])
        rephrased_instruction = tf.where(
            tf.strings.length(rephrased_instruction) > 0,
            original_language[0] + "." + rephrased_instruction,
            original_language[0],
        )
        split_tensor = tf.strings.split(rephrased_instruction, sep=".")
        num_strings = tf.cast(tf.shape(split_tensor)[0], tf.int32)
        random_index = tf.random.uniform(
            (tf.shape(original_language)[0],),
            minval=0,
            maxval=num_strings,
            dtype=tf.int32,
        )
        sampled_language = tf.gather(split_tensor, random_index)
        rand = tf.random.uniform(shape=(), minval=0, maxval=1, dtype=tf.float32)
        sampled_language = tf.where(
            rand < rephrase_prob,
            sampled_language,
            original_language,
        )
        traj["task"]["language_instruction"] = sampled_language
    return traj


def add_fuse_modal_commands(
    traj: dict, json_file_path: str
):
    """
    Adds fuse modal commands (e.g. "Grab the green frog" -> ["Grab the green frog", "", "Grab the object that looks green", "Grab the object that feels soft", ...] 
    to the language instruction based on the provided JSON file mapping simple commands to |-separated lists of modal commands. 
        0: simple command 
        1: empty command (kept for legacy reasons)
        2: visual command
        3: tactile command
        4: auditory command
        5: visual-tactile command
        6: visual-auditory command
        7: tactile-auditory command
        8: visual-tactile-auditory command
    
    For some tasks, the language instruction is not valid,
    e.g. a tactile command for a button-pressing task. Therefore, also creates a language validity mask
    in the trajectory, based on the information in the loaded file.

    Args:
        traj: A dictionary containing trajectory data. Should have a "task" key.
        json_file_path: The path to the JSON file that contains both the modal commands and the language validity
        mappings
    """
    expander = FuseRephraser.create(json_file_path, ("rephrase",))
    validity_checker = FuseRephraser.create(json_file_path, ("valid",), value_dtype=tf.int32, default_value=0)
    if "language_instruction" not in traj["task"]:
        return traj
    original_language = traj["task"]["language_instruction"]
    traj_len = tf.shape(original_language)[0]
    # don't check that language key is nonempty, or that dict is not empty - want to error instead,
    # since these are required for fuse 
    
    joined_modal_commands = expander.hash_table.lookup(original_language[0])
    validity_decimal = validity_checker.hash_table.lookup(original_language[0])
    validity = tf.math.floormod(tf.bitwise.right_shift(tf.expand_dims(validity_decimal,0), tf.range(9)), 2)
    modal_strings = tf.strings.split(joined_modal_commands, sep="|")
    traj["task"]["modal_commands"] = tf.repeat(tf.expand_dims(modal_strings, axis=0), traj_len, axis=0)
    traj["task"]["language_validity"] = tf.repeat(tf.expand_dims(validity, axis=0), traj_len, axis=0)

    return traj


def rephrase_fuse_modal_commands(
    traj: dict, rephrase_prob: float, json_file_path: str, num_rephrases_per_modal_command: int = 20
):
    """
    Rephrases each of the fuse modal commands, each with a probability of `rephrase_prob`.

    Args:
        traj: A dictionary containing trajectory data. Should have a "task" key.
        rephrase_prob: The probability of rephrasing each of the fuse modal commands.
        json_file_path: The path to the JSON file that contains a mapping of modal base commands to |-separated lists of rephrased commands.
    """

    rephraser = FuseRephraser.create(json_file_path, default_value=" | " * (num_rephrases_per_modal_command-1))

    if "language_instruction" not in traj["task"]:
        return traj
    original_language = traj["task"]["language_instruction"]
    traj_len = tf.shape(original_language)[0]
    modal_commands = traj["task"]["modal_commands"][0]
    rephrased_instruction = rephraser.hash_table.lookup(modal_commands)
    split_tensor = tf.strings.split(rephrased_instruction, sep="|")
    num_modalities = tf.cast(tf.shape(split_tensor)[0], tf.int32)
    num_strings = tf.cast(tf.shape(split_tensor[0])[0], tf.int32)
    random_index = tf.random.uniform(
        (traj_len, num_modalities),
        minval=0,
        maxval=num_strings,
        dtype=tf.int32,
    )
    split_tensor = tf.repeat(tf.expand_dims(split_tensor.to_tensor(), axis=0), traj_len, axis=0)
    sampled_language = tf.gather(split_tensor, random_index, batch_dims=2)
    should_rephrase = tf.random.uniform(
        shape=(traj_len, num_modalities),
        minval=0,
        maxval=1,
        dtype=tf.float32
    )
    sampled_language = tf.where(
        should_rephrase < rephrase_prob,
        sampled_language,
        traj["task"]["modal_commands"],
    )
    traj["task"]["modal_commands"] = sampled_language
    return traj
    

def select_fuse_modal_command(
    traj: dict, 
):
    traj_len = tf.shape(traj["task"]["language_instruction"])[0]
    modal_strings=  traj["task"]["modal_commands"][0]
    modal_index = tf.squeeze(tf.random.categorical(
        logits=tf.math.log([[1.0/8 if i != 1 else 0.0 for i in range(9)]]),
        num_samples=traj_len,
        dtype=tf.int32
    ), axis=0)
    traj["observation"]["modality_idx"] = modal_index
    traj["observation"]["pad_mask_dict"]["modality_idx"] = tf.ones([traj_len], dtype=tf.bool)
    traj["task"]["language_instruction"] = tf.gather(modal_strings, modal_index)
    return traj


def fuse_augmentation(
    traj: dict, 
    modal_file_path: str,
    rephrase_file_path: str,
    rephrase_prob: float = 0.5,
):
    traj = add_fuse_modal_commands(traj, modal_file_path)
    traj = rephrase_fuse_modal_commands(traj, rephrase_prob, rephrase_file_path)
    traj = select_fuse_modal_command(traj)
    return traj


def delete_task_conditioning(
    traj: dict,
    keep_image_prob: float,
):
    """
    Randomly drops out either the goal images or the language instruction. Only does something if both of
    these are present.

    Args:
        traj: A dictionary containing trajectory data. Should have a "task" key.
        keep_image_prob: The probability of keeping the goal images. The probability of keeping the language
            instruction is 1 - keep_image_prob.
    """
    if "language_instruction" not in traj["task"]:
        return traj

    image_keys = {
        key
        for key in traj["task"].keys()
        if key.startswith("image_") or key.startswith("depth_")
    }
    if not image_keys:
        return traj

    traj_len = tf.shape(traj["action"])[0]
    should_keep_images = tf.random.uniform([traj_len]) < keep_image_prob
    should_keep_images |= ~traj["task"]["pad_mask_dict"]["language_instruction"]

    for key in image_keys | {"language_instruction"}:
        should_keep = should_keep_images if key in image_keys else ~should_keep_images
        # pad out the key
        traj["task"][key] = tf.where(
            should_keep,
            traj["task"][key],
            to_padding(traj["task"][key]),
        )
        # zero out the pad mask dict for the key
        traj["task"]["pad_mask_dict"][key] = tf.where(
            should_keep,
            traj["task"]["pad_mask_dict"][key],
            tf.zeros_like(traj["task"]["pad_mask_dict"][key]),
        )

    # when no goal images are present, the goal timestep becomes the final timestep
    traj["task"]["timestep"] = tf.where(
        should_keep_images,
        traj["task"]["timestep"],
        traj_len - 1,
    )

    return traj
