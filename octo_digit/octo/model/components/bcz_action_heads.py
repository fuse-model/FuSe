from abc import ABC, abstractmethod
import logging
from typing import Dict, Optional, Tuple

import distrax
from einops import rearrange
import flax.linen as nn
import jax
from jax import Array
import jax.numpy as jnp
from jax.typing import ArrayLike

from octo.model.components.base import TokenGroup
from octo.model.components.diffusion import cosine_beta_schedule, create_diffusion_model
from octo.model.components.tokenizers import BinTokenizer
from octo.model.components.transformer import MAPHead
from octo.model.components.unet import ConditionalUnet1D, unet_squaredcos_cap_v2
from octo.utils.typing import PRNGKey
from octo.model.components.action_heads import masked_mean


class BczActionHead(ABC):

    @abstractmethod
    def loss(
        self,
        encoded_features: ArrayLike,
        actions: ArrayLike,
        timestep_pad_mask: ArrayLike,
        action_pad_mask: ArrayLike,
        *args,
        **kwargs,
    ) -> Tuple[Array, Dict[str, Array]]:
        raise NotImplementedError

    @abstractmethod
    def predict_action(
        self,
        encoded_features: ArrayLike,
        sample_shape: Tuple[int, ...] = (),
        rng: Optional[PRNGKey] = None,
        embodiment_action_dim: Optional[int] = None,
        train: bool = True
    ) -> Array:
        """Predict the action for the last timestep in the window. Returns shape
        (*sample_shape, batch_size, action_horizon, action_dim).
        """
        raise NotImplementedError


def continuous_loss(
    pred_value: ArrayLike,
    ground_truth_value: ArrayLike,
    mask: ArrayLike,
    loss_type: str = "mse",
    huber_delta: float = 1.0
) -> Array:
    """
    Args:
        pred_value: shape (batch_dims...)
        ground_truth_value: continuous values w/ shape (batch_dims...)
        mask: broadcastable to ground_truth
    """
    if loss_type == "mse":
        loss = jnp.square(pred_value - ground_truth_value)
    elif loss_type == "huber": 
        diff = jnp.abs(pred_value - ground_truth_value)
        mse_loss = jnp.square(pred_value - ground_truth_value)
        outlier_loss = 2 * huber_delta * (diff - 0.5 * huber_delta)
        loss = jnp.where(diff > huber_delta, outlier_loss, mse_loss)
    elif loss_type == "l1":
        loss = jnp.abs(pred_value - ground_truth_value)
    else:
        raise ValueError(f"Invalid loss type: {loss_type}")

    loss = masked_mean(loss, mask)

    mse = jnp.square(pred_value - ground_truth_value)
    mse = masked_mean(mse, mask)
    return loss, {
        "loss": loss,
        "mse": mse,
    }


class BczContinuousActionHead(nn.Module, BczActionHead):
    hidden_width: int = 256
    action_horizon: int = 1
    action_dim: int = 7
    max_action: float = 5.0
    loss_type: str = "mse"
    
    @nn.compact
    def __call__(
        self, encoded_features: ArrayLike, train: bool = True
    ) -> jax.Array:
        
        features = nn.Dense(self.hidden_width)(encoded_features)
        features = nn.relu(features)
        features = nn.Dense(self.hidden_width)(features)
        features = nn.relu(features)
        preds = nn.Dense(self.action_dim * self.action_horizon)(features)
        preds = rearrange(
            preds, "b w (h a) -> b w h a", h=self.action_horizon, a=self.action_dim
        )
        preds = jnp.tanh(preds / self.max_action) * self.max_action
        return preds

    def loss(
        self,
        encoded_features: ArrayLike,
        actions: ArrayLike,
        timestep_pad_mask: ArrayLike,
        action_pad_mask: ArrayLike,
        train: bool = True
    ) -> Tuple[Array, Dict[str, Array]]:
        """Computes the loss for the action regression objective.

        Args:
            encoded_features: (batch_size, embedding_size) array resulting from FiLM encoding of visual features and caption
            actions: shape (batch_size, window_size, action_horizon, action_dim)
            timestep_pad_mask: boolean array (batch, window_size) which is True if the timestep is not a padding timestep
            action_pad_mask: boolean array (same shape as actions) which is True if the action dimension is not a padding dimension

        Returns:
            loss: float
            metrics: dict
        """
        # (batch, window_size, action_horizon, action_dim)
        preds = self(encoded_features)

        # combine the timestep pad mask with the action pad mask
        mask = timestep_pad_mask[:, :, None, None] & action_pad_mask

        loss, metrics = continuous_loss(preds, actions, mask, loss_type=self.loss_type)
        # Sum over action dimension instead of averaging
        loss = loss * self.action_dim
        metrics["loss"] = metrics["loss"] * self.action_dim
        metrics["mse"] = metrics["mse"] * self.action_dim
        return loss, metrics

    def predict_action(
        self,
        encoded_features,
        *args,
        sample_shape: tuple = (),
        train: bool = True, 
        **kwargs,
    ) -> jax.Array:
        """Convenience methods for predicting actions for the final timestep in the window."""
        # only get the last timestep in the window
        mean = self(encoded_features)[:, -1]
        return jnp.broadcast_to(mean, sample_shape + mean.shape)

class MSEBczActionHead(BczContinuousActionHead):
    max_action: float = 5.0
    loss_type: str = "mse"

class HuberBczActionHead(BczContinuousActionHead): 
    max_action: float = 5.0
    loss_type: str = "huber"
