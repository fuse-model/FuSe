from abc import ABC, abstractmethod
from typing import Dict, Tuple, Union

from einops import rearrange
import flax.linen as nn
import jax
from jax import Array
import jax.numpy as jnp
from jax.typing import ArrayLike
import optax

from octo.model.components.base import TokenGroup
from octo.model.components.transformer import MAPHead
from octo.model.components.action_heads import masked_mean

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
        transformer_outputs: Dict[str, TokenGroup],
        true_language_embeddings: jax.Array, 
        timestep_pad_mask: ArrayLike,
        train: bool = True,
    ) -> Tuple[Array, Dict[str, Array]]:
        raise NotImplementedError


class ConinuousLanguageEmbeddingHead(nn.Module, LanguageReconstructionHead, ABC): 
    """Predicts continuous language embedding of shape associated with vision/sensor context from transformer outputs. 
    
    """

    readout_key: str = 'readout_language'
    latent_dim: int = 512
    use_map: bool = True
    n_lang_tokens: int = 16
    use_separate_heads: bool = False

    def setup(self):
        if self.use_map:
            self.pred_map_head = MAPHead(name='pred_map_head')
            self.true_map_head = MAPHead(name='true_map_head')
        self.pred_latent_proj = nn.Dense(self.latent_dim, name='pred_latent_proj')
        self.true_latent_proj = nn.Dense(self.latent_dim, name='true_latent_proj')
        self.temperature = self.param('temperature', jax.nn.initializers.constant(1.0), (1,))


    def __call__(
        self, transformer_outputs: Dict[str, TokenGroup], train: bool = True
    ) -> jax.Array:
        token_group = transformer_outputs[self.readout_key]
        
        features = token_group.tokens
        assert features.ndim == 4, (
            f"Expected pred features to have shape (batch_size, window_size, num_tokens, embedding_size)"
            f"but got shape {features.shape}"
        )
        if self.use_map: 
            embeddings = self.pred_map_head(features, train=train)[:, :, 0] # remove token dimension
        else: 
            embeddings = features.mean(axis=-2)
            
        embeddings = embeddings[:, -1] # remove window dimension
        
        # Now, embeddings is (batch_size, embedding_size)
        pred_embeddings_latent = self.pred_latent_proj(embeddings)

        true_language_embedding = transformer_outputs['raw_lang']
        features = true_language_embedding
        assert features.ndim == 3, (
            f"Expected true features to have shape (batch_size, num_tokens, embedding_size)"
            f"but got shape {features.shape}"
        )
        if self.use_map: 
            if self.use_separate_heads:
                embeddings = self.true_map_head(features, train=train)[:, :, 0]
            else:
                embeddings = self.pred_map_head(features, train=train)[:, 0] # remove token dimension
        else: 
            embeddings = features.mean(axis=-2)

        true_embeddings_latent = self.true_latent_proj(embeddings)

        return pred_embeddings_latent, true_embeddings_latent

    @abstractmethod
    def loss(
        self,
        transformer_outputs: Dict[str, TokenGroup],
        true_language_embeddings: jax.Array, 
        timestep_pad_mask: ArrayLike,
        train: bool = True,
    ) -> Tuple[Array, Dict[str, Array]]:
        raise NotImplementedError



class BCZLanguageHead(ConinuousLanguageEmbeddingHead): 

    def loss(
        self,
        transformer_outputs: Dict[str, TokenGroup],
        true_language_embedding: jax.Array, 
        timestep_pad_mask: ArrayLike,
        train: bool = True,
    ) -> Tuple[Array, Dict[str, Array]]:
        transformer_outputs['raw_lang'] = true_language_embedding
        pred_embeddings_latent, true_embeddings_latent = self(transformer_outputs, train=train)
        loss, metrics = cosine_loss(pred_embeddings_latent, true_embeddings_latent)
        return loss, metrics


class CLIPContrastiveHead(ConinuousLanguageEmbeddingHead): 

    def loss(
        self,
        transformer_outputs: Dict[str, TokenGroup],
        true_language_embedding: jax.Array, 
        timestep_pad_mask: ArrayLike,
        train: bool = True,
    ) -> Tuple[Array, Dict[str, Array]]:
        transformer_outputs['raw_lang'] = true_language_embedding
        pred_embeddings_latent, true_embeddings_latent = self(transformer_outputs, train=train)
        loss, metrics = contrastive_loss(pred_embeddings_latent, true_embeddings_latent, self.temperature)
        return loss, metrics
    
    
    
    
class LanguageGenerationHead(ABC):

    @abstractmethod
    def loss(
        self,
        transformer_outputs: Dict[str, TokenGroup],
        tokenized_language_target: jax.Array, 
        timestep_pad_mask: ArrayLike,
        train: bool = True,
    ) -> Tuple[Array, Dict[str, Array]]:
        raise NotImplementedError
    
    
class ContinuousGenerationHead(nn.Module, LanguageGenerationHead, ABC): 
    """Predicts continuous language embedding of shape associated with vision/sensor context from transformer outputs. 
    
    """

    readout_key: str = 'readout_language'
    vocab_size: int = 32_218
    use_map: bool = True
    n_lang_tokens: int = 16

    def setup(self):
        if self.use_map: 
            self.map_head = MAPHead(name='decode_map_head', num_readouts=self.n_lang_tokens)
        self.latent_to_vocab = nn.Dense(self.vocab_size, name='latent_to_vocab')
        


    def __call__(
        self, transformer_outputs: Dict[str, TokenGroup], train: bool = True
    ) -> jax.Array:
        token_group = transformer_outputs[self.readout_key]
        
        features = token_group.tokens
        assert features.ndim == 4, (
            f"Expected pred features to have shape (batch_size, window_size, num_tokens, embedding_size)"
            f"but got shape {features.shape}"
        )
        if self.use_map: 
            embeddings = self.map_head(features, train=train)
        else: 
            embeddings = features
            
        embeddings = embeddings[:, -1] # remove window dimension -> (b, n_lang, emb_size)
        
        pred_lang = self.latent_to_vocab(embeddings)  # (b, n_lang, vocab_size) probability distributions
        return pred_lang
    
    def reconstruct_lang(self, transformer_outs: Dict[str, TokenGroup], train: bool = True): 
        logits = self(transformer_outs, train=train)
        vocab_ids = jnp.argmax(logits, axis=-1)
        # decode_ids = rearrange(vocab_ids, '( b nlang) -> b nlang', nlang=self.n_lang_tokens)
        return vocab_ids

    def loss(
        self,
        transformer_outputs: Dict[str, TokenGroup],
        tokenized_language_target: jax.Array, 
        timestep_pad_mask: ArrayLike,
        train: bool = True,
    ) -> Tuple[Array, Dict[str, Array]]:
        pred_lang_logits = self(transformer_outputs, train=train)
        logits = rearrange(pred_lang_logits, 'b nlang vocab -> (b nlang) vocab')
        targets = rearrange(tokenized_language_target, 'b nlang -> (b nlang)')
        sftmax_loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
        loss_val = jnp.mean(sftmax_loss)
        return loss_val, {'loss': loss_val }
    
    
    
class SingleHeadContinuousGenerationHead(nn.Module, LanguageGenerationHead, ABC): 
    """Predicts continuous language embedding of shape associated with vision/sensor context from transformer outputs. 
    
    """

    readout_key: str = 'readout_language'
    vocab_size: int = 32_218
    use_map: bool = True
    n_lang_tokens: int = 16
    token_embedding_shape: int = 384

    def setup(self):
        if self.use_map: 
            self.map_head = MAPHead(name='decode_map_head', num_readouts=self.n_lang_tokens)
        self.latent_to_vocab = nn.Dense(self.vocab_size, name='latent_to_vocab')
        self.modality_class_tokens = nn.Embed(
            num_embeddings=9,
            features=self.token_embedding_shape,
        )

    def __call__(
        self, transformer_outputs: Dict[str, TokenGroup], modality_idx: int, train: bool = True
    ) -> jax.Array:
        token_group = transformer_outputs[self.readout_key]
        
        features = token_group.tokens
        assert features.ndim == 4, (
            f"Expected pred features to have shape (batch_size, window_size, num_tokens, embedding_size)"
            f"but got shape {features.shape}"
        )
        b, w, n_tok, emb_size = features.shape
        class_token = self.modality_class_tokens(modality_idx)
        class_token = jnp.broadcast_to(class_token, (b, w, 1, emb_size))
        
        all_features = jnp.concatenate((features, class_token), axis=-2)
        
        if self.use_map: 
            embeddings = self.map_head(all_features, train=train)
        else: 
            embeddings = all_features
            raise NotImplementedError
            
        embeddings = embeddings[:, -1] # remove window dimension -> (b, n_lang, emb_size)
        
        pred_lang = self.latent_to_vocab(embeddings)  # (b, n_lang, vocab_size) probability distributions
        return pred_lang
    
    def reconstruct_lang(self, transformer_outs: Dict[str, TokenGroup], modality_idx: int, train: bool = True): 
        logits = self(transformer_outs, modality_idx=modality_idx, train=train)
        vocab_ids = jnp.argmax(logits, axis=-1)
        return vocab_ids

    def loss(
        self,
        transformer_outputs: Dict[str, TokenGroup],
        tokenized_language_target: jax.Array, 
        modality_idx: int,
        timestep_pad_mask: ArrayLike,
        mask: ArrayLike, 
        train: bool = True,
    ) -> Tuple[Array, Dict[str, Array]]:
        pred_lang_logits = self(transformer_outputs, modality_idx=modality_idx, train=train)
        logits = rearrange(pred_lang_logits, 'b nlang vocab -> (b nlang) vocab')
        targets = rearrange(tokenized_language_target, 'b nlang -> (b nlang)')
        sftmax_loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
        loss = rearrange(sftmax_loss, '(b nlang) -> b nlang', nlang=self.n_lang_tokens)
        if mask is not None:
            loss_val = masked_mean(loss, mask=mask)
        else:
            loss_val = jnp.mean(loss)
        return loss_val, {'loss': loss_val }


        
        
        