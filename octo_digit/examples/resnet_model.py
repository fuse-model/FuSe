from typing import Dict, Optional

import flax.linen as nn
import jax
import jax.numpy as jnp
from octo.model.components.vit_encoders import StdConv, ViTResnet
from octo.utils.typing import Data, Sequence


class ResnetModule(nn.Module):
    image_encoder_stages: Sequence[tuple[str, tuple]] = (
        ("image_primary", (2, 2, 2, 2)),
        ("image_wrist", (2, 2, 2, 2)),
    )
    image_embedding_size: int = 512
    mlp_widths: tuple[int] = ()
    language_key: str = "language_instruction"
    action_dim: int = 7
    action_pred_horizon: int = 1

    @nn.compact
    def __call__(
        self,
        batch: Data,
    ):
        observations = batch["observation"]
        b, w = observations[self.image_encoder_stages[0][0]].shape[:2]
        embeddings = []
        for observation_key, encoder_stages in self.image_encoder_stages:
            embedding = ViTResnet(num_layers=encoder_stages)(
                observations[observation_key]
            )
            embedding = StdConv(self.image_embedding_size, (3, 3))(embedding)
            embedding = jnp.mean(embedding, axis=(-2, -3))  # GAP
            embeddings.append(embedding)

        lang = jnp.tile(
            batch["task"][self.language_key][:, None, ...], (1, w, 1)
        )  # repeat task embedding over window
        embeddings.append(lang)
        x = jnp.concatenate(embeddings, axis=-1)
        x = jnp.reshape(b, -1)
        for width in self.mlp_widths:
            x = nn.Dense(width)(x)
        x = nn.Dense(self.action_dim * self.action_pred_horizon)(x)
        x = jnp.reshape(x, (-1, self.action_pred_horizon, self.action_dim))
        return x

    def create(
        cls,
        observation_tokenizers: Dict[str, ModuleSpec],
        task_tokenizers: Dict[str, ModuleSpec],
        heads: Dict[str, ModuleSpec],
        readouts: Dict[str, int],
        transformer_kwargs: Dict,
        token_embedding_size: int,
        max_horizon: int,
        repeat_task_tokens: bool = False,
        use_correct_attention: bool = False,
    ) -> "OctoModule":
        """
        Canonical way to create an OctoModule from configuration.

        Args:
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
        task_tokenizer_defs = {
            k: ModuleSpec.instantiate(spec)() for k, spec in task_tokenizers.items()
        }

        head_defs = {k: ModuleSpec.instantiate(spec)() for k, spec in heads.items()}

        model_def = OctoTransformer(
            observation_tokenizers=observation_tokenizer_defs,
            task_tokenizers=task_tokenizer_defs,
            readouts=readouts,
            token_embedding_size=token_embedding_size,
            max_horizon=max_horizon,
            repeat_task_tokens=repeat_task_tokens,
            transformer_kwargs=transformer_kwargs,
            use_correct_attention=use_correct_attention,
        )

        return cls(
            octo_transformer=model_def,
            heads=head_defs,
        )
