from functools import partial
from typing import Any, Callable

from einops import rearrange
from flax import linen as nn
import jax.numpy as jnp


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
    bias: bool = True
    dynamic_img_pad: bool = False
    flatten: bool = True
    strict_image_size: bool = True

    def setup(self):
        if isinstance(self.patch_size, int):
            patch_size = (self.patch_size, self.patch_size)
        else:
            patch_size = self.patch_size
        self.grid_size = tuple([s // p for s, p in zip(self.img_size, patch_size)])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.proj = nn.Conv(
            features=self.embed_dim,
            kernel_size=patch_size,
            strides=patch_size,
            use_bias=self.bias,
            padding=0,
        )

    def __call__(self, x):
        x = self.proj(x)
        x = rearrange(x, "n h w c -> n (h w) c")
        return x


def to_2tuple(num):
    return tuple([num for _ in range(2)])


class JaxMlpLayer(nn.Module):
    in_features: int
    hidden_features: int = None
    out_features: int = None
    act_layer: nn.Module = nn.gelu
    norm_layer: nn.Module = None
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
        linear_layer = nn.Dense

        self.fc1 = linear_layer(hidden_features, use_bias=bias[0])
        self.act = self.act_layer
        self.fc2 = linear_layer(out_features, use_bias=bias[1])

    def __call__(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
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
    act_layer: nn.Module = nn.gelu
    norm_layer: nn.Module = nn.LayerNorm
    mlp_layer: nn.Module = JaxMlpLayer
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
        self.ls1 = JaxLayerScale(self.dim) if self.init_values else JaxIdentity()

        self.norm2 = self.norm_layer()
        self.mlp = self.mlp_layer(
            in_features=self.dim,
            hidden_features=int(self.dim * self.mlp_ratio),
            act_layer=self.act_layer,
            drop=self.proj_drop,
        )
        self.ls2 = JaxLayerScale(self.dim) if self.init_values else JaxIdentity()

    def __call__(self, x):
        x = x + self.ls1(self.attn(self.norm1(x)))
        x = x + self.ls2(self.mlp(self.norm2(x)))
        return x


class tvlViT(nn.Module):
    img_size: tuple[int] = (224, 224)
    patch_size: int = 16
    in_chans: int = 3
    num_classes: int = 768
    global_pool: str = "avg"
    depth: int = 12
    class_token: bool = True
    no_embed_class: bool = False
    pre_norm: bool = False
    fc_norm: nn.Module = None
    dynamic_img_size: bool = False
    dynamic_img_pad: bool = False
    drop_rate: float = 0.0
    pos_drop_rate: float = 0.0
    patch_drop_rate: float = 0.0
    drop_path_rate: float = 0.0
    weight_init: str = ""
    embed_layer: nn.Module = JaxPatchEmbed
    act_layer: Callable = None
    attn_drop_rate: float = 0.0
    block_fn: nn.Module = JaxBlock
    embed_dim: int = 384
    init_values: float = None
    mlp_layer: nn.Module = JaxMlpLayer
    mlp_ratio: float = 4.0
    norm_layer: Callable = None
    num_heads: int = 6
    proj_drop_rate: float = 0.0
    qk_norm: bool = False
    qkv_bias: bool = True
    grad_checkpointing: bool = False
    normalization_type: str = "digit_bgs"
    # octo_embedding_dim: int = 512  # overrides num_classes if present

    def setup(self):
        assert self.global_pool == "avg"
        use_fc_norm = (
            self.global_pool == "avg" if self.fc_norm is None else self.fc_norm
        )
        norm_layer = self.norm_layer or partial(nn.LayerNorm, epsilon=1e-6)
        act_layer = self.act_layer or nn.gelu

        self.num_features = self.embed_dim
        self.num_prefix_tokens = 1 if self.class_token else 0

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
            raise ValueError
        embed_len = (
            num_patches if self.no_embed_class else num_patches + self.num_prefix_tokens
        )
        self.pos_embed = self.param(
            "pos_embed", nn.initializers.lecun_normal(), (1, embed_len, self.embed_dim)
        )

        self.blocks = nn.Sequential(
            [
                self.block_fn(
                    name=f"block{i}",
                    dim=self.embed_dim,
                    num_heads=self.num_heads,
                    mlp_ratio=self.mlp_ratio,
                    qkv_bias=self.qkv_bias,
                    qk_norm=self.qk_norm,
                    init_values=self.init_values,
                    proj_drop=self.proj_drop_rate,
                    attn_drop=self.attn_drop_rate,
                    drop_path=0.0,
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                    mlp_layer=self.mlp_layer,
                    block_num=i,
                )
                for i in range(self.depth)
            ]
        )
        self.norm = norm_layer() if not use_fc_norm else JaxIdentity()

        # Classifier Head
        self.fc_norm_layer = (
            norm_layer(name="fc_norm") if use_fc_norm else JaxIdentity()
        )
        # num_classes = self.octo_embedding_dim if self.octo_embedding_dim > 0 else self.num_classes
        self.head = (
            nn.Dense(self.num_classes) if self.num_classes > 0 else JaxIdentity()
        )

    def _pos_embed(self, x):
        b = x.shape[0]
        expanded_cls_token = jnp.broadcast_to(
            self.cls_token, (b, *self.cls_token.shape[1:])
        )
        x = jnp.concatenate((expanded_cls_token, x), axis=1)
        x = x + self.pos_embed
        return x

    def __call__(self, x, train: bool = False):
        # patchify
        x = self.patch_embed(x)
        x = self._pos_embed(x)

        # run transformer
        x = self.blocks(x)
        x = self.norm(x)

        # global pool
        x = jnp.mean(x[:, self.num_prefix_tokens :], axis=1)

        # head
        x = self.fc_norm_layer(x)
        x = self.head(x)
        return x
