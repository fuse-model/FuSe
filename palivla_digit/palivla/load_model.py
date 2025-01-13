import fnmatch
from functools import partial
from typing import Dict, List, Optional

from big_vision.models.proj.paligemma import paligemma
from big_vision.trainers.proj.paligemma import predict_fns
from flax import linen as nn
import jax
import jax.numpy as jnp
import ml_collections
from ml_collections import ConfigDict, FrozenConfigDict
import optax
from palivla.spec import ModuleSpec
from palivla.tokenizer import Tokenizer
from palivla.utils import key_string
from scalax.sharding import MeshShardingHelper, ShardingRule

model_config = {
    "llm": {"vocab_size": 257_152},
    "img": {
        "variant": "So400m/14",
        "pool_type": "none",
        "scan": True,
    },
}


def get_model_spec(**kwargs):
    return ModuleSpec.create(
        paligemma.Model,
        model_config | kwargs,
    )


def load_pretrained_params(
    checkpoint_path: str,
    dtype: jnp.dtype = jnp.float32,
):
    params = paligemma.load(None, checkpoint_path, FrozenConfigDict(model_config))
    params = jax.tree.map(lambda x: x.astype(dtype), params)

    return params


def get_decode_fn(model: nn.Module, tokenizer: Tokenizer):
    decode_fn = predict_fns.get_all(model)["decode"]
    decode = partial(
        decode_fn,
        devices=jax.devices(),
        eos_token=tokenizer.config.eos_token,
    )
    return decode


def load_model_params_decode(config: ConfigDict, tokenizer: Tokenizer):
    from big_vision.models.proj.paligemma import paligemma
    from big_vision.trainers.proj.paligemma import predict_fns

    # Define model
    model = paligemma.Model(**FrozenConfigDict(model_config))

    # Load params - this can take up to 1 minute in T4 colabs.
    params = paligemma.load(None, config.model_path, FrozenConfigDict(model_config))

    # Change params to fp32
    params = jax.tree.map(lambda x: x.astype(jnp.float32), params)

    decode_fn = predict_fns.get_all(model)["decode"]
    decode = partial(
        decode_fn,
        devices=jax.devices(),
        eos_token=tokenizer.config.eos_token,
    )

    return model, params, decode


def component_label_fn(nested_params_dict):
    label_rules = [
        # Assign the input/output embeddings and the image connector to "embed"
        ("*/embedder", "embed"),
        ("img/head", "embed"),
        # Assign the llm and the image encoder to "llm" and "img" respectively
        ("llm/*", "llm"),
        ("img/*", "img"),
        # Assign the rest to "embed"
        # This includes any other modality-specific encoders
        ("*", "embed"),
    ]

    def label_fn(path, _):
        path_str = key_string(path)
        for pattern, label in label_rules:
            if fnmatch.fnmatch(path_str, pattern):
                return label
        return path_str

    return jax.tree_util.tree_map_with_path(label_fn, nested_params_dict)


def adamw_cosine_warmup(
    *,
    learning_rate,
    warmup_steps,
    total_steps,
    global_norm,
    weight_decay,
    b1=0.9,
    b2=0.999,
):
    @optax.inject_hyperparams
    def _make_optimizer(lr):
        return optax.chain(
            optax.clip_by_global_norm(global_norm),
            optax.adamw(lr, weight_decay=weight_decay, b1=b1, b2=b2),
        )

    return _make_optimizer(
        optax.warmup_cosine_decay_schedule(
            0, learning_rate, warmup_steps, total_steps - warmup_steps
        )
    )


def make_optimizer(**config):
    @optax.inject_hyperparams
    def _make_optimizer(llm_learning_rate, img_learning_rate, embed_learning_rate):
        def _make_opt(lr, weight_decay, grad_norm_clip, b1, b2):
            if config["optimizer"] == "adamw":
                return optax.chain(
                    optax.clip_by_global_norm(grad_norm_clip),
                    optax.adamw(lr, weight_decay=weight_decay, b1=b1, b2=b2),
                )
            elif config["optimizer"] == "sgd":
                return optax.chain(
                    optax.clip_by_global_norm(grad_norm_clip),
                    optax.sgd(lr),
                )
            else:
                raise ValueError(f"Unknown optimizer: {config['optimizer']}")

        img_optimizer = _make_opt(
            img_learning_rate,
            config["img_optimizer_kwargs"]["weight_decay"],
            config["img_optimizer_kwargs"]["grad_norm_clip"],
            config["img_optimizer_kwargs"]["b1"],
            config["img_optimizer_kwargs"]["b2"],
        )
        embed_optimizer = _make_opt(
            embed_learning_rate,
            config["embed_optimizer_kwargs"]["weight_decay"],
            config["embed_optimizer_kwargs"]["grad_norm_clip"],
            config["embed_optimizer_kwargs"]["b1"],
            config["embed_optimizer_kwargs"]["b2"],
        )
        llm_optimizer = _make_opt(
            llm_learning_rate,
            config["llm_optimizer_kwargs"]["weight_decay"],
            config["llm_optimizer_kwargs"]["grad_norm_clip"],
            config["llm_optimizer_kwargs"]["b1"],
            config["llm_optimizer_kwargs"]["b2"],
        )

        return optax.multi_transform(
            {
                "llm": llm_optimizer,
                "img": img_optimizer,
                "embed": embed_optimizer,
            },
            component_label_fn,
        )

    def _make_learning_rate(optimizer_kwargs):
        return optax.warmup_cosine_decay_schedule(
            optimizer_kwargs["init_learning_rate"],
            optimizer_kwargs["learning_rate"],
            optimizer_kwargs["warmup_steps"],
            config["num_steps"] - optimizer_kwargs["warmup_steps"],
        )

    return _make_optimizer(
        _make_learning_rate(config["llm_optimizer_kwargs"]),
        _make_learning_rate(config["img_optimizer_kwargs"]),
        _make_learning_rate(config["embed_optimizer_kwargs"]),
    )


def components_by_label(values):
    labels = component_label_fn(values)
    groups = {}
    for label, value in zip(jax.tree.leaves(labels), jax.tree.leaves(values)):
        groups[label] = value
    return groups
