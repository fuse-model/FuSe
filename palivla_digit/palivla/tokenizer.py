import dataclasses
from functools import partial
from typing import Callable, Literal, Optional

from einops import rearrange
from flax import linen as nn
from flax import struct
from flax.core.frozen_dict import FrozenDict
import jax
from jax.experimental import jax2tf
import jax.numpy as jnp
from ml_collections import ConfigDict
import numpy as np
from palivla.utils import freeze_structure
import tensorflow as tf
from tensorflow_text import SentencepieceTokenizer


class ActionTokenizer(nn.Module):
    def tokenize(self, data, obs=None):
        ...

    def detokenize(self, tokens, obs=None):
        ...


class BinActionTokenizer(ActionTokenizer):
    min_action_value: jax.Array = struct.field(pytree_node=True)
    max_action_value: jax.Array = struct.field(pytree_node=True)
    action_dim: int | None = struct.field(pytree_node=False)
    action_vocab_size: int = struct.field(pytree_node=False)
    action_horizon: int = struct.field(pytree_node=False)

    @property
    def num_tokens(self):
        return self.action_horizon * self.action_dim

    @property
    def vocab_size(self):
        return self.action_vocab_size

    def tokenize(self, data, obs=None):
        # Assume normalization and clipping to [-1, 1]
        data = jnp.clip(data, self.min_action_value, self.max_action_value)
        data = (data - self.min_action_value) / (
            self.max_action_value - self.min_action_value
        )
        data = rearrange(data, "... p a -> ... (p a)")
        return jnp.clip(
            (data * self.vocab_size).astype(jnp.int32),
            0,
            self.vocab_size - 1,
        )

    def detokenize(self, tokens, obs=None, action_dim: int | None = None):
        values = tokens / self.vocab_size
        values = jnp.where((values < 0) | (values > 1), jnp.nan, values)
        data = (
            values * (self.max_action_value - self.min_action_value)
            + self.min_action_value
        )
        data = rearrange(data, "... (p a) -> ... p a", a=action_dim or self.action_dim)
        return data


@struct.dataclass
class Tokenizer:
    @struct.dataclass
    class TokenizerConfig:
        action_vocab_size: int = struct.field(pytree_node=False)
        action_vocab_offset: int = struct.field(pytree_node=False)
        vocab_size: int = struct.field(pytree_node=False)
        num_action_tokens: int = struct.field(pytree_node=False)
        bos_token: int = struct.field(pytree_node=True)
        eos_token: int = struct.field(pytree_node=True)
        pad_token: int = struct.field(pytree_node=True)
        begin_of_action_token: int = struct.field(pytree_node=True)
        max_pad_length: int = struct.field(pytree_node=False)
        min_action_value: float = struct.field(pytree_node=True)
        max_action_value: float = struct.field(pytree_node=True)
        prompt_autoregressive: bool = struct.field(pytree_node=True)

        @classmethod
        def create(
            cls,
            action_tokenizer: ActionTokenizer,
            language_tokenizer: SentencepieceTokenizer,
            prompt_autoregressive: bool = False,
        ):
            return cls(
                action_vocab_size=action_tokenizer.vocab_size,
                action_vocab_offset=256000,
                vocab_size=language_tokenizer.vocab_size,
                num_action_tokens=action_tokenizer.num_tokens,
                bos_token=language_tokenizer.string_to_id("<bos>").numpy().item(),
                eos_token=language_tokenizer.string_to_id("<eos>").numpy().item(),
                pad_token=language_tokenizer.string_to_id("<pad>").numpy().item(),
                begin_of_action_token=language_tokenizer.string_to_id("\n")
                .numpy()
                .item(),
                max_pad_length=60,
                min_action_value=getattr(action_tokenizer, "min_action_value", None),
                max_action_value=getattr(action_tokenizer, "max_action_value", None),
                prompt_autoregressive=prompt_autoregressive,
            )

    config: TokenizerConfig = struct.field(pytree_node=True)
    language_tokenizer: SentencepieceTokenizer = struct.field(pytree_node=False)
    token_structure: FrozenDict = struct.field(pytree_node=False)

    action_tokenizer: ActionTokenizer = struct.field(pytree_node=False)
    action_tokenizer_params: dict = struct.field(pytree_node=True)

    _tf_action_tokenize_fn: Optional[tf.Module] = struct.field(pytree_node=False)
    _jax_action_tokenize_fn: Optional[Callable] = struct.field(pytree_node=False)
    _jax_action_detokenize_fn: Optional[Callable] = struct.field(pytree_node=False)

    @classmethod
    def from_components(
        cls,
        language_tokenizer: SentencepieceTokenizer,
        action_tokenizer: ActionTokenizer,
        action_tokenizer_params: dict = None,
        *,
        prompt_autoregressive: bool = False,
        config: TokenizerConfig | None = None,
    ):
        if config is None:
            config = cls.TokenizerConfig.create(
                action_tokenizer, language_tokenizer, prompt_autoregressive
            )

        pad_token = language_tokenizer.string_to_id("<pad>").numpy().item()

        @tf.function(autograph=False)
        def _tokenize(params, actions, obs):
            toks = (
                jax2tf.convert(
                    partial(action_tokenizer.apply, method="tokenize"),
                    native_serialization=True,
                    native_serialization_platforms=["cpu"],
                )({"params": params}, actions[None], obs=obs),
            )
            if isinstance(toks, tuple):
                toks = toks[0]
            return tf.squeeze(toks, axis=0)

        return cls(
            config=cls.TokenizerConfig.create(
                action_tokenizer, language_tokenizer, prompt_autoregressive
            ),
            language_tokenizer=language_tokenizer,
            token_structure=FrozenDict(
                freeze_structure(
                    {
                        "prefix": [
                            [config.bos_token],
                            "prompt",
                            [config.begin_of_action_token],
                        ],
                        "causal": [
                            "action",
                        ],
                        "pad": [
                            [pad_token] * config.max_pad_length,
                        ],
                    }
                )
            ),
            action_tokenizer=action_tokenizer,
            action_tokenizer_params=action_tokenizer_params,
            _tf_action_tokenize_fn=_tokenize,
            _jax_action_detokenize_fn=jax.jit(
                partial(action_tokenizer.apply, method="detokenize")
            ),
            _jax_action_tokenize_fn=jax.jit(
                partial(action_tokenizer.apply, method="tokenize")
            ),
        )

    def compose_token_structure(self, tokens, include_keys=["prefix", "causal", "pad"]):
        def _extract_tokens(ids_or_str):
            if isinstance(ids_or_str, str):
                return tokens[ids_or_str]
            else:
                return tf.constant(ids_or_str, dtype=tf.int32)

        tokens_by_name = {
            k: (
                tf.concat([_extract_tokens(token) for token in v], axis=0)
                if k in include_keys
                else tf.zeros((0,), dtype=tf.int32)
            )
            for k, v in self.token_structure.items()
        }

        tokens = tf.concat(
            [tokens_by_name["prefix"], tokens_by_name["causal"], tokens_by_name["pad"]],
            axis=0,
        )[: self.config.max_pad_length]
        include_prefix_mask = (
            tf.ones_like(tokens_by_name["prefix"], dtype=tf.bool)
            if self.config.prompt_autoregressive
            else tf.zeros_like(tokens_by_name["prefix"], dtype=tf.bool)
        )
        mask_ar = tf.concat(
            [
                include_prefix_mask,
                tf.ones_like(tokens_by_name["causal"], dtype=tf.bool),
                tf.ones_like(tokens_by_name["pad"], dtype=tf.bool),
            ],
            axis=0,
        )[: self.config.max_pad_length]
        mask_loss = tf.concat(
            [
                include_prefix_mask,
                tf.ones_like(tokens_by_name["causal"], dtype=tf.bool),
                tf.zeros_like(tokens_by_name["pad"], dtype=tf.bool),
            ],
            axis=0,
        )[: self.config.max_pad_length]

        always_include_prefix_mask = tf.ones_like(
            tokens_by_name["prefix"], dtype=tf.bool
        )
        mask_ar_fuse = tf.concat(
            [
                always_include_prefix_mask,
                tf.ones_like(tokens_by_name["causal"], dtype=tf.bool),
                tf.ones_like(tokens_by_name["pad"], dtype=tf.bool),
            ],
            axis=0,
        )[: self.config.max_pad_length]
        mask_loss_fuse = tf.concat(
            [
                always_include_prefix_mask,
                tf.zeros_like(tokens_by_name["causal"], dtype=tf.bool),
                tf.zeros_like(tokens_by_name["pad"], dtype=tf.bool),
            ],
            axis=0,
        )[: self.config.max_pad_length]

        return tokens, mask_ar, mask_loss, mask_ar_fuse, mask_loss_fuse

    def tokenize_language_instruction(self, data):
        instruction = data["task"]["language_instruction"]
        instruction = tf.strings.lower(instruction)
        instruction = tf.strings.regex_replace(instruction, "[.?!]", "")
        instruction = tf.strings.regex_replace(instruction, "\n", " ")
        instruction = tf.strings.strip(instruction)
        instruction = tf.strings.join([tf.constant("act "), instruction])

        return self.language_tokenizer.tokenize(instruction)

    def tokenize_action(self, data, obs=None):
        is_single_sample = data.ndim == 1
        if is_single_sample:
            data = data[None]
        tokens = (
            self._jax_action_tokenize_fn(
                {"params": self.action_tokenizer_params}, data, obs=obs
            )
            + self.config.action_vocab_offset
        )
        if is_single_sample:
            tokens = tokens[0]
        return tokens

    def detokenize_action(self, tokens, obs=None):
        is_single_sample = tokens.ndim == 1
        if is_single_sample:
            tokens = tokens[None]
        with jax.profiler.TraceAnnotation("detokenize"):
            recon = self._jax_action_detokenize_fn(
                {"params": self.action_tokenizer_params},
                tokens - self.config.action_vocab_offset,
                obs=obs,
            )

        if is_single_sample:
            recon = recon[0]
        return recon

    def prepare_tokens_for_training(self, data, language_token_instructions):
        tokens = {
            "prompt": language_token_instructions[: self.config.max_pad_length - 10],
            "action": self._tf_action_tokenize_fn(
                self.action_tokenizer_params, data["action"][-1], None
            )
            + self.config.action_vocab_offset,
        }

        (
            tokens,
            mask_ar,
            mask_loss,
            mask_ar_fuse,
            mask_loss_fuse,
        ) = self.compose_token_structure(tokens)

        return {
            "tokens": tokens,
            "mask_ar": mask_ar,
            "mask_loss": mask_loss,
            "mask_ar_fuse": mask_ar_fuse,
            "mask_loss_fuse": mask_loss_fuse,
            "mask_input": tokens != self.config.pad_token,
        }

    def prepare_tokens_for_generation(self, data, language_token_instructions):
        tokens = {
            "prompt": language_token_instructions[: self.config.max_pad_length - 10],
            "action": self._tf_action_tokenize_fn(
                self.action_tokenizer_params, data["action"][-1], None
            )
            + self.config.action_vocab_offset,
        }

        (
            tokens,
            mask_ar,
            mask_loss,
            mask_ar_fuse,
            mask_loss_fuse,
        ) = self.compose_token_structure(tokens)

        return {
            "tokens": tokens,
            "mask_ar": mask_ar,
            "mask_input": tokens != self.config.pad_token,
        }

    def extract_action(self, data):
        action_start = (
            jnp.argmax(data["tokens"] == self.config.begin_of_action_token, axis=-1) + 1
        )
        action_data = data["tokens"][
            action_start : action_start + self.config.num_action_tokens
        ]
        action_data = self.bin_detokenize(action_data)
        return action_data
