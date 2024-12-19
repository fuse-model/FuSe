from abc import ABC, abstractmethod
import logging
from typing import Dict, Optional, Sequence, Tuple, Union

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
from dataclasses import field
import optax
from octo.model.components.action_heads import masked_mean
from dataclasses import dataclass

EPSILON = 0.1

def mse_loss(
    pred_embedding: ArrayLike, 
    true_embedding: ArrayLike,
): 
    return jnp.mean(jnp.square(pred_embedding - true_embedding))

def cosine_loss(
    pred_embedding: ArrayLike, 
    true_embedding: ArrayLike,
): 
    cosine_distance = jnp.mean(optax.cosine_distance(true_embedding, pred_embedding, epsilon=0.1))
    mse = mse_loss(pred_embedding, true_embedding)
    return cosine_distance, {
        'loss': cosine_distance, 
        'mse': mse
    }    

# from CLIP
def contrastive_loss(
    pred_embedding: ArrayLike,
    true_embedding: ArrayLike, 
    temperature: Union[float, ArrayLike] = 1.0
): 
    assert pred_embedding.shape == true_embedding.shape and pred_embedding.ndim == 2, ( 
        f'Expected equal shapes of (b, emb_dim), but got {pred_embedding.shape} and {true_embedding.shape}'
    )
    batch_size, emb_dim = pred_embedding.shape
    def _normalize(vec): 
        norm = jnp.linalg.norm(vec, axis=-1)
        return vec / (norm[:, None] + EPSILON)
    def _symmetric_cross_entropy(logits, labels): 
        loss_rows = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
        loss_cols = optax.softmax_cross_entropy_with_integer_labels(logits.T, labels)
        return (loss_rows + loss_cols) / 2
    norm_pred = _normalize(pred_embedding)  
    norm_true = _normalize(true_embedding)
    logits = jnp.dot(norm_pred, norm_true.T) * temperature 
    labels = jnp.arange(batch_size)
    loss = jnp.mean(_symmetric_cross_entropy(logits, labels)) 

    mse = mse_loss(pred_embedding, true_embedding)
    return loss, {
        'loss': loss, 
        'mse': mse
    }




class LanguageReconstructionHead(ABC):
    """A head used to reconstruct language in some way (e.g. BC-style embedding prediction, caption generation, contrastive-style, etc.) from 
    the outputs of the transformer (without language input)
    """

    @abstractmethod
    def loss(
        self,
        encoded_features: ArrayLike,
        true_language_embeddings: ArrayLike, 
        timestep_pad_mask: ArrayLike,
        train: bool = True,
    ) -> Tuple[Array, Dict[str, Array]]:
        raise NotImplementedError

class BczCosineFeaturesLanguageHead(nn.Module, LanguageReconstructionHead): 
    language_dim: int
    
    @nn.compact
    def __call__(
        self, 
        encoded_features: ArrayLike, 
        train: bool = False
    ): 
        reconstructed_embedding = nn.Dense(self.language_dim)(encoded_features)
        reconstructed_embedding = jnp.mean(reconstructed_embedding, axis=-2) # average over window dimension
        return reconstructed_embedding
    
    def loss(
        self,
        encoded_features: ArrayLike,
        true_language_embeddings: ArrayLike, 
        timestep_pad_mask: ArrayLike,
        train: bool = True,
    ): 
        reconstructed_embedding = self(encoded_features, train=train)
        return cosine_loss(reconstructed_embedding, true_language_embeddings)

class BczNullLanguageHead(nn.Module, LanguageReconstructionHead): 
    @nn.compact
    def __call__(
        self, 
        encoded_features: ArrayLike, 
        train: bool = False
    ): 
        return encoded_features
    
    def loss(
        self,
        encoded_features: ArrayLike,
        true_language_embeddings: ArrayLike, 
        timestep_pad_mask: ArrayLike,
        train: bool = True,
    ): 
        return 0.0, {}