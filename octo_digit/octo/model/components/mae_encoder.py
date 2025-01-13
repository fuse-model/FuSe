from typing import Callable

import flax.linen as nn
from einops import rearrange
import jax
import jax.numpy as jnp


class EarlyCNN(nn.Module):
    encoder_dim: int
    key: str = 'image'
    
    @nn.compact
    def __call__(self, x):
        x = nn.relu(nn.Conv(features=self.encoder_dim // 8, kernel_size=(6, 6), strides=4, padding=1)(x))
        x = nn.relu(nn.Conv(features=self.encoder_dim // 4, kernel_size=(6, 6), strides=4, padding=1)(x))
        kernel_size = 4 if self.key == 'image' else 3
        stride = 2 if self.key == 'image' else 1
        x = nn.relu(nn.Conv(features=self.encoder_dim // 2, kernel_size=(kernel_size, kernel_size), strides=stride, padding=1)(x))
        x = nn.Conv(features=self.encoder_dim, kernel_size=(1, 1))(x)
        x = rearrange(x, 'b h w c -> b (h w) c')
        return x


class FeedForward(nn.Module):
    dim: int
    hidden_dim: int
    dropout: float = 0.0
    act: Callable = nn.gelu
    
    @nn.compact
    def __call__(self, x, train: bool = True):
        x = nn.LayerNorm(epsilon=1e-5)(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = self.act(x)
        x = nn.Dropout(self.dropout, deterministic=not train)(x)
        x = nn.Dense(self.dim)(x)
        x = nn.Dropout(self.dropout, deterministic = not train)(x)
        return x


class Attention(nn.Module):
    dim: int
    heads: int = 8
    dim_head: int = 64
    dropout: float = 0.0
    
    @nn.compact
    def __call__(self, x, train: bool = True):
        inner_dim = self.dim_head * self.heads
        project_out = not (self.heads == 1 and self.dim_head == self.dim)
        scale = self.dim_head ** -0.5
        x = nn.LayerNorm()(x)
        qkv = nn.Dense(features=3 * inner_dim, use_bias=False, name='to_qkv')(x)
        qkv = jnp.split(qkv, 3, axis=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
        
        
        dots = jnp.matmul(q, jnp.swapaxes(k, -1, -2)) * scale
        attn = nn.softmax(dots)
        attn = nn.Dropout(self.dropout, deterministic=not train)(attn)
        
        out = jnp.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        if project_out:
            out = nn.Dense(self.dim, name='to_out')(out)
            out = nn.Dropout(self.dropout, deterministic = not train)(out)
        return out


class TransformerLayer(nn.Module):
    dim: int
    heads: int
    dim_head: int
    mlp_dim: int
    dropout: float = 0.0
    
    @nn.compact
    def __call__(self, x, train: bool = True):
        x = Attention(
            dim=self.dim,
            heads=self.heads,
            dim_head=self.dim_head,
            dropout=self.dropout,
            name=f'attentionblock'
        )(x, train=train) + x
        x = FeedForward(
            dim=self.dim,
            hidden_dim=self.mlp_dim,
            dropout=self.dropout,
            name=f'feedforward'
        )(x, train=train) + x
        return x


class Transformer(nn.Module):
    dim: int
    depth: int
    heads: int
    dim_head: int
    mlp_dim: int
    dropout: float = 0.0
    
    @nn.compact
    def __call__(self, x, train: bool = True):
        for _ in range(self.depth):
            x = TransformerLayer(
                dim=self.dim,
                heads=self.heads,
                dim_head=self.dim_head,
                mlp_dim=self.mlp_dim,
                dropout=self.dropout,
            )(x, train=train)
        x = nn.LayerNorm(epsilon=1e-5)(x)
        return x


class MAE(nn.Module):
    tactile_size: int
    tactile_patch_size: int
    dim: int
    depth: int
    heads: int
    mlp_dim: int
    dim_head: int = 64
    dropout: float = 0.0
    emb_dropout: float = 0.0

    @nn.compact
    def __call__(self, tactile, train: bool = True):      
        num_tactile_patches = (self.tactile_size // self.tactile_patch_size) ** 2
        tactile_tokens = EarlyCNN(encoder_dim=self.dim, key='tactile')(tactile)
        tactile_tokens += self.param('tactile_pos_embedding', jax.nn.initializers.xavier_normal(), (1, num_tactile_patches, self.dim))
        
        encoded_tokens = Transformer(
            dim=self.dim,
            depth=self.depth,
            heads=self.heads,
            dim_head=self.dim_head,
            mlp_dim=self.mlp_dim,
            dropout=self.dropout,
            name='transformer',
        )(tactile_tokens, train=train)
        
        return encoded_tokens


class JaxMAE(MAE):
    tactile_size: int = 128
    tactile_patch_size: int = 16
    dim: int = 1024
    depth: int = 6
    heads: int = 8
    mlp_dim: int = 2048       