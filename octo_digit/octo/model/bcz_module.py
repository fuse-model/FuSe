# Written by Dibya
from dataclasses import field
from functools import partial
import logging
from multiprocessing.sharedctypes import Value
from typing import Dict, Optional

import flax.linen as nn
import jax
import jax.numpy as jnp
from octo.utils.spec import ModuleSpec
from octo.utils.typing import Data, Sequence


def expand_dims_to_shape(array: jax.Array, shape: tuple):
    num_to_add = len(shape) - len(array.shape)
    new_axes = tuple(range(-1, -1 - num_to_add, -1))
    return jnp.expand_dims(array, axis=new_axes)


def masked_gap(image_encoding: jax.Array, mask: jax.Array):
    mask = expand_dims_to_shape(mask, image_encoding.shape)
    mask = jnp.broadcast_to(mask, image_encoding.shape)
    return jnp.mean(image_encoding * mask, axis=-2) / jnp.clip(
        jnp.mean(mask, axis=-2), a_min=1e-5
    )


class BczEncoder(nn.Module):
    observation_tokenizers: Dict[str, nn.Module]
    max_horizon: int

    def __call__(
        self,
        observations: Data,
        tasks: Data,
        timestep_pad_mask: jax.Array,
        verbose: bool = False,
        train: bool = False,
    ):
        viz_features = [
            masked_gap(tok(observations, tasks).tokens, timestep_pad_mask)
            for _, tok in self.observation_tokenizers.items()  # b t * f
        ]
        viz_features = jnp.concatenate(viz_features, axis=-1)
        return viz_features


class BczModule(nn.Module):
    """
    Bundles encoder and action, language heads.
    """

    bcz_encoder: BczEncoder
    heads: Dict[str, nn.Module]

    def __call__(
        self,
        observations,
        tasks,
        timestep_pad_mask,
        train=True,
        verbose=False,
    ):
        """Run transformer and the main method for all heads. Useful for init.

        Args:
            observations: A dictionary containing observation data
                where each element has shape (batch, horizon, *).
            tasks: A dictionary containing task data
                where each element has shape (batch, *).
            timestep_pad_mask: A boolean mask of shape (batch, horizon) where False indicates a padded timestep.
            verbose: If True, prints out the structure of the OctoTransformer (useful for debugging!)

        Returns:
            encoded_features: (*, encoding_dim) feature output
            head_outputs: dictionary of outputs from heads {head_name: output}
        """
        encoded_features = self.bcz_encoder(
            observations, tasks, timestep_pad_mask, verbose=verbose
        )
        head_outputs = {}
        for head_name, head in self.heads.items():
            head_outputs[head_name] = head(encoded_features, train=train)
        return encoded_features, head_outputs

    @classmethod
    def create(
        cls,
        observation_tokenizers: Dict[str, ModuleSpec],
        heads: Dict[str, ModuleSpec],
        max_horizon: int,
    ) -> "BczModule":
        """
        Canonical way to create an OctoModule from configuration.

        Args:
            example_batch: used to infer shapes of positional embeddings
            observation_tokenizers: dict of {tokenizer_name: tokenizer_spec} (see tokenizers.py)
            task_tokenizers: dict of {tokenizer_name: tokenizer_spec} (see tokenizers.py)
            heads: dict of {head_name: head_spec} (see heads.py)
            readouts: dict of {readout_name (str): n_tokens_for_readout (int)}
            token_embedding_size (int): The latent dimension of the token embeddings
            max_horizon (int): Sets the size of positional embeddings, and provides an upper limit on the
                maximum horizon of the model
            repeat_task_tokens (bool): If true, repeats the task tokens at each observation timestep.
            transformer_kwargs: additional kwargs to forward to the transformer, which include:
                num_layers (int): number of layers
                mlp_dim (int): hidden dimension of the MLPs
                num_heads (int): Number of heads in nn.MultiHeadDotProductAttention
                dropout_rate (float): dropout rate.
                attention_dropout_rate (float): dropout rate in self attention.
        """
        observation_tokenizer_defs = {
            k: ModuleSpec.instantiate(spec)()
            for k, spec in observation_tokenizers.items()
        }

        def sort_dict(d: dict) -> dict:
            dict_keys = sorted(d.keys())
            return {key: d[key] for key in dict_keys}

        observation_tokenizer_defs = sort_dict(observation_tokenizer_defs)

        head_defs = {k: ModuleSpec.instantiate(spec)() for k, spec in heads.items()}

        bcz_encoder = BczEncoder(
            observation_tokenizers=observation_tokenizer_defs,
            max_horizon=max_horizon,
        )
        return cls(
            bcz_encoder=bcz_encoder,
            heads=head_defs,
        )
