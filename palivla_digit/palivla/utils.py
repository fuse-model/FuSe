import re

import flax
import jax
from jax.experimental import multihost_utils
import numpy as np
from palivla.types import Params
import tensorflow as tf


def freeze_structure(structure):
    return jax.tree_util.tree_map(
        lambda x: tuple(freeze_structure(y) for y in x) if isinstance(x, list) else x,
        structure,
        is_leaf=lambda x: isinstance(x, list),
    )


def key_string(path, separator="/") -> str:
    def _component_to_string(component) -> str:
        if isinstance(component, jax.tree_util.SequenceKey):
            return str(component.idx)
        elif isinstance(component, jax.tree_util.DictKey):
            return str(component.key)
        elif isinstance(component, jax.tree_util.GetAttrKey):
            return str(component.name)
        elif isinstance(component, jax.tree_util.FlattenedIndexKey):
            return str(component.key)
        else:
            return str(component)

    return separator.join(_component_to_string(component) for component in path)


def host_broadcast_str(x: str | None) -> str:
    """
    Broadcast_one_to_all, but with a string.

    Works by padding the string to the length of the longest string and then
    broadcasting the result, then stripping the padding.

    Note: this will remove the padding from the end of the string.
    """
    if x is None:
        x = ""

    max_len = multihost_utils.broadcast_one_to_all(len(x))
    padded = x.ljust(max_len)

    encoded = np.array([ord(c) for c in padded], dtype=np.uint8)[:max_len]
    encoded = multihost_utils.broadcast_one_to_all(encoded)
    decoded = "".join([chr(u) for u in encoded])

    return decoded.rstrip()


def load_tvl_weights(pretrained_path: str) -> dict[tuple, np.ndarray]:
    with tf.io.gfile.GFile(pretrained_path, "rb") as f:
        ckpt_dict = np.load(f, allow_pickle=False)
    keys, values = zip(*list(ckpt_dict.items()))
    return {tuple(k.split("|")): v for k, v in zip(keys, values)}


def merge_params(init_params: Params, pretrained_params: Params) -> Params:
    def _merge(possible_param1, possible_param2):
        if possible_param2 is not None:
            return possible_param2
        return possible_param1

    flat_init_params = flax.traverse_util.flatten_dict(init_params)
    flat_pretrained_params = flax.traverse_util.flatten_dict(pretrained_params)
    params = {
        k: _merge(v_init, flat_pretrained_params.get(k, None))
        for k, v_init in flat_init_params.items()
    }
    return flax.traverse_util.unflatten_dict(params)
