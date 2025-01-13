from enum import Enum
from functools import partial
from typing import Any, Callable, Iterable, Optional, Tuple, Union

from einops import rearrange
from flax import linen as nn
from flax.traverse_util import flatten_dict, unflatten_dict
import jax
import jax.numpy as jnp
from octo.model.components.vit_encoders import normalize_images


def to_n_tuple(num_or_tuple: Union[float, Tuple], n: int):
    if isinstance(num_or_tuple, Iterable):
        assert len(num_or_tuple) == n
        return num_or_tuple
    return tuple([num_or_tuple for _ in range(2)])


to_2tuple = partial(to_n_tuple, n=2)


class JaxIdentity(nn.Module):
    @nn.compact
    def __call__(self, x):
        return x


class JaxAttention(nn.Module):
    dim: int = 384
    num_heads: int = 6
    qkv_bias: bool = True
    qk_norm: bool = False
    attn_drop: float = 0.0
    proj_drop: float = 0.0
    norm_layer: nn.Module = nn.LayerNorm

    def setup(self):
        assert self.dim % self.num_heads == 0, "dim should be divisible by num_heads"
        self.head_dim = self.dim // self.num_heads

        self.qkv = nn.Dense(features=self.dim * 3, use_bias=self.qkv_bias)
        self.proj = nn.Dense(self.dim)

    def __call__(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = rearrange(qkv, "B N D nh h -> D B N nh h")
        q, k, v = qkv
        x = nn.dot_product_attention(q, k, v)
        x = rearrange(x, "B N nh h -> B N (nh h)")
        x = self.proj(x)
        return x


class JaxPatchEmbed(nn.Module):
    img_size: int = 224
    patch_size: int = 16
    in_chans: int = 3
    embed_dim: int = 768
    norm_layer: Optional[Callable] = None
    flatten: bool = True
    output_fmt: Optional[str] = None
    bias: bool = True
    strict_image_size: bool = True
    dynamic_img_pad: bool = False

    def setup(self):
        patch_size = to_2tuple(self.patch_size)
        if self.img_size is not None:
            img_size = to_2tuple(self.img_size)
            self.grid_size = tuple([s // p for s, p in zip(img_size, patch_size)])
            self.num_patches = self.grid_size[0] * self.grid_size[1]
        else:
            raise NotImplementedError

        if self.output_fmt is not None:
            raise NotImplementedError

        self.proj = nn.Conv(
            features=self.embed_dim,
            kernel_size=patch_size,
            strides=patch_size,
            use_bias=self.bias,
            padding=0,
        )
        self.norm = self.norm_layer() if self.norm_layer else JaxIdentity()

    def __call__(self, x):
        x = self.proj(x)
        if self.flatten:
            x = rearrange(x, "n h w c -> n (h w) c")
        else:
            raise NotImplementedError
        x = self.norm(x)
        return x


class JaxMlpLayer(nn.Module):
    in_features: int
    hidden_features: int = None
    out_features: int = None
    act_layer: Callable = nn.gelu
    norm_layer: Callable = None
    bias: bool = True
    drop: float = 0.0
    use_conv: bool = False

    def setup(self):
        out_features = self.out_features or self.in_features
        hidden_features = self.hidden_features or self.in_features
        bias = to_2tuple(self.bias)
        drop_probs = to_2tuple(self.drop)
        if self.use_conv or self.drop > 0:
            raise NotImplementedError
        linear_layer = partial(nn.Conv, kernel_size=1) if self.use_conv else nn.Dense

        self.fc1 = linear_layer(hidden_features, use_bias=bias[0])
        self.act = self.act_layer
        self.drop1 = (
            nn.Dropout(rate=drop_probs[0]) if drop_probs[0] > 0 else JaxIdentity()
        )
        self.norm = (
            self.norm_layer(hidden_features)
            if self.norm_layer is not None
            else JaxIdentity()
        )
        self.fc2 = linear_layer(out_features, use_bias=bias[1])
        self.drop2 = (
            nn.Dropout(rate=drop_probs[1]) if drop_probs[1] > 0 else JaxIdentity()
        )

    def __call__(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class JaxLayerScale(nn.Module):
    dim: int = 384
    init_values: float = 1e-5

    def setup(self):
        self.gamma = self.param(
            "gamma", nn.initializers.constant(value=self.init_values), (self.dim,)
        )

    def __call__(self, x):
        return x * self.gamma


class JaxBlock(nn.Module):
    dim: int
    num_heads: int
    mlp_ratio: float = 4.0
    qkv_bias: bool = False
    qk_norm: bool = False
    proj_drop: float = 0.0
    attn_drop: float = 0.0
    init_values: Any = None
    drop_path: float = 0.0
    act_layer: Callable = nn.gelu
    norm_layer: Callable = nn.LayerNorm
    mlp_layer: Callable = JaxMlpLayer
    block_num: int = 0

    def setup(self):
        self.norm1 = self.norm_layer()
        self.attn = JaxAttention(
            dim=self.dim,
            num_heads=self.num_heads,
            qkv_bias=self.qkv_bias,
            qk_norm=self.qk_norm,
            attn_drop=self.attn_drop,
            proj_drop=self.proj_drop,
            norm_layer=self.norm_layer,
        )
        if self.init_values:
            raise ValueError
        self.ls1 = (
            JaxLayerScale(self.dim, init_values=self.init_values)
            if self.init_values
            else JaxIdentity()
        )
        if self.init_values:
            raise NotImplementedError
        if self.drop_path > 0:
            raise NotImplementedError
        else:
            self.drop_path1 = JaxIdentity()
        self.norm2 = self.norm_layer()
        self.mlp = self.mlp_layer(
            in_features=self.dim,
            hidden_features=int(self.dim * self.mlp_ratio),
            act_layer=self.act_layer,
            drop=self.proj_drop,
        )
        self.ls2 = JaxLayerScale(self.dim) if self.init_values else JaxIdentity()
        self.drop_path2 = JaxIdentity()

    def __call__(self, x):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


class t3ViT(nn.Module):
    img_size: Union[int, Tuple[int, int]] = 224
    patch_size: Union[int, Tuple[int, int]] = 16
    in_chans: int = 3
    num_classes: int = 1000
    global_pool: str = "token"
    embed_dim: int = 768
    depth: int = 3
    num_heads: int = 12
    mlp_ratio: float = 4.0
    qkv_bias: bool = True
    qk_norm: bool = False
    init_values: Optional[float] = None
    class_token: bool = True
    no_embed_class: bool = False
    pre_norm: bool = False
    fc_norm: Optional[bool] = None
    dynamic_img_size: bool = False
    dynamic_img_pad: bool = False
    drop_rate: float = 0.0
    pos_drop_rate: float = 0.0
    patch_drop_rate: float = 0.0
    proj_drop_rate: float = 0.0
    attn_drop_rate: float = 0.0
    drop_path_rate: float = 0.0
    weight_init: str = ""
    embed_layer: Callable = JaxPatchEmbed
    norm_layer: Optional[Callable] = None
    act_layer: Optional[Callable] = None
    block_fn: Callable = JaxBlock
    mlp_layer: Callable = JaxMlpLayer
    # octo_embedding_dim: int = 512
    normalization_type: str = "digit_bgs"

    def setup(self):
        assert self.global_pool in ("avg", "token")
        assert self.class_token or self.global_pool != "token"
        use_fc_norm = (
            self.global_pool == "avg" if self.fc_norm is None else self.fc_norm
        )
        norm_layer = self.norm_layer or partial(nn.LayerNorm, epsilon=1e-6)
        act_layer = self.act_layer or nn.gelu

        self.num_features = self.embed_dim
        self.num_prefix_tokens = 1 if self.class_token else 0

        assert not self.dynamic_img_size
        self.grad_checkpointing = False

        self.patch_embed = self.embed_layer(
            img_size=self.img_size,
            patch_size=self.patch_size,
            in_chans=self.in_chans,
            embed_dim=self.embed_dim,
            bias=not self.pre_norm,
            dynamic_img_pad=self.dynamic_img_pad,
        )
        num_patches = self.patch_embed.num_patches

        if self.class_token:
            self.cls_token = self.param(
                "cls_token", nn.initializers.lecun_normal(), (1, 1, self.embed_dim)
            )
        else:
            raise NotImplementedError
        embed_len = (
            num_patches if self.no_embed_class else num_patches + self.num_prefix_tokens
        )
        if self.no_embed_class:
            raise NotImplementedError
        self.pos_embed = self.param(
            "pos_embed", nn.initializers.lecun_normal(), (1, embed_len, self.embed_dim)
        )
        self.pos_drop = (
            nn.Dropout(rate=self.pos_drop_rate)
            if self.pos_drop_rate > 0
            else JaxIdentity()
        )
        if self.pos_drop_rate > 0:
            raise NotImplementedError
        if self.patch_drop_rate > 0:
            raise NotImplementedError
        else:
            self.patch_drop = JaxIdentity()

        self.norm_pre = norm_layer() if self.pre_norm else JaxIdentity

        if self.drop_path_rate > 0:
            raise NotImplementedError
        self.blocks = nn.Sequential(
            [
                self.block_fn(
                    name=f"block{i}",
                    dim=self.embed_dim,
                    num_heads=self.num_heads,
                    mlp_ratio=self.mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                )
                for i in range(self.depth)
            ]
        )

        # self.head = nn.Dense(self.octo_embedding_dim) if self.octo_embedding_dim > 0 else JaxIdentity()
        self.head = JaxIdentity()

    def _pos_embed(self, x):
        if self.dynamic_img_size:
            raise NotImplementedError
        else:
            pos_embed = self.pos_embed
        if self.no_embed_class:
            raise NotImplementedError
        else:
            if self.cls_token is not None:
                b = x.shape[0]
                expanded_cls_token = jnp.broadcast_to(
                    self.cls_token, (b, *self.cls_token.shape[1:])
                )
                x = jnp.concatenate((expanded_cls_token, x), axis=1)
            x = x + pos_embed
        return self.pos_drop(x)

    def __call__(self, x, train: bool = False, pre_logits: bool = False):
        # normalize
        x = normalize_images(x, img_norm_type=self.normalization_type)

        # patchify
        x = self.patch_embed(x)
        b = x.shape[0]
        expanded_cls_token = jnp.broadcast_to(
            self.cls_token, (b, *self.cls_token.shape[1:])
        )
        x = jnp.concatenate((expanded_cls_token, x), axis=1)
        x = x + self.pos_embed

        # run transformer
        x = self.blocks(x)

        # pool - not in the original t3 architecture, but necessary to reduce token num
        x = jnp.mean(x[:, self.num_prefix_tokens :], axis=1)

        # also not in the original t3 architecture, but need to product to octo latent space
        x = self.head(x)
        return x


class t3ViTtiny(t3ViT):
    embed_dim: int = 192
    num_heads = 3


class t3ViTsmall(t3ViT):
    embed_dim: int = 384
    num_heads: int = 6


class t3ViTmedium(t3ViT):
    embed_dim: int = 768
    num_heads: int = 6


class t3ViTlarge(t3ViT):
    embed_dim: int = 1024
    num_heads: int = 8
