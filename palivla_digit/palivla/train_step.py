from functools import partial
from typing import Dict

import chex
from dataset import create_fuse_modal_mask, enforce_valid_language_instruction
from einops import repeat
from flax.training.train_state import TrainState
import jax
import jax.numpy as jnp
import optax
from palivla.load_model import components_by_label
from palivla.tokenizer import ActionTokenizer, Tokenizer
from palivla.types import TrainingBatch


def smooth_nll_loss(logits, labels, sigma, base_action_token, action_vocab_size):
    # Normal PDF with mean `label` and std `sigma`
    bin_cutoffs = jnp.arange(action_vocab_size - 1) + 0.5
    cdf = jax.scipy.stats.norm.cdf(
        bin_cutoffs, labels[..., None] - base_action_token, sigma
    )
    cdf = jnp.concatenate(
        [jnp.zeros_like(cdf[..., :1]), cdf, jnp.ones_like(cdf[..., :1])], axis=-1
    )
    label_probs = jnp.diff(cdf, axis=-1)

    logits = jax.nn.log_softmax(logits, axis=-1)
    entropy = -jnp.sum(jax.scipy.special.xlogy(label_probs, label_probs), axis=-1)
    return (
        -jnp.sum(
            label_probs
            * logits[..., base_action_token : base_action_token + action_vocab_size],
            axis=-1,
        )
        - entropy
    )


def get_action_tokens(
    pred_logits, tokens, num_action_tokens: int, begin_of_action_token: int
):
    _get_action_tokens = jax.vmap(
        lambda x, i: jax.lax.dynamic_slice(x, (i,), (num_action_tokens,))
    )

    action_token_starts = jnp.argmax(tokens == begin_of_action_token, axis=-1) + 1
    pred_tokens = jnp.argmax(pred_logits, axis=-1)
    pred_action_tokens = _get_action_tokens(pred_tokens, action_token_starts)
    pred_action_logits = jax.vmap(_get_action_tokens, in_axes=(-1, None), out_axes=-1)(
        pred_logits, action_token_starts
    )
    gt_action_tokens = _get_action_tokens(tokens, action_token_starts)

    return {
        "pred_action_tokens": pred_action_tokens,
        "pred_action_logits": pred_action_logits,
        "gt_action_tokens": gt_action_tokens,
    }


def compute_action_metrics(
    detokenize_fn,
    *,
    pred_action_tokens,
    pred_action_logits,
    gt_action_tokens,
    gt_actions,
    action_dim: int,
    tokenizer_config: Tokenizer.TokenizerConfig,
    log_segment_prefix=None,
):
    decoded_actions = detokenize_fn(pred_action_tokens)
    decoded_actions_gt = detokenize_fn(gt_action_tokens)

    batch_size = gt_actions.shape[0]
    chex.assert_shape(
        (pred_action_tokens, gt_action_tokens),
        (batch_size, tokenizer_config.num_action_tokens),
    )
    if pred_action_logits is not None:
        chex.assert_shape(
            (pred_action_logits), (batch_size, tokenizer_config.num_action_tokens, None)
        )
    chex.assert_shape(
        (gt_actions, decoded_actions_gt, decoded_actions),
        (batch_size, None, action_dim),
    )

    def stats_for_metric(value, name, shape):
        chex.assert_shape(value, (None, shape))
        return {f"{log_segment_prefix}{name}": jnp.mean(value),} | {
            f"_details/{log_segment_prefix}{name}_{i}": jnp.mean(value[:, i])
            for i in range(shape)
        }

    error = decoded_actions - decoded_actions_gt
    tokenization_error = gt_actions - decoded_actions_gt

    if tokenizer_config.min_action_value is not None:
        tokenization_error = (
            jnp.clip(
                gt_actions,
                tokenizer_config.min_action_value,
                tokenizer_config.max_action_value,
            )
            - decoded_actions_gt
        )

    return {
        **stats_for_metric(jnp.mean(jnp.abs(error), axis=1), "l1", action_dim),
        **stats_for_metric(jnp.mean(jnp.square(error), axis=1), "l2", action_dim),
        **stats_for_metric(
            jnp.mean(jnp.abs(tokenization_error), axis=1), "tok_l1", action_dim
        ),
        **stats_for_metric(
            jnp.mean(jnp.square(tokenization_error), axis=1), "tok_l2", action_dim
        ),
        **stats_for_metric(
            pred_action_tokens == gt_action_tokens,
            "acc",
            tokenizer_config.num_action_tokens,
        ),
    } | (
        {
            **stats_for_metric(
                optax.softmax_cross_entropy_with_integer_labels(
                    pred_action_logits, gt_action_tokens
                ),
                "action_loss",
                tokenizer_config.num_action_tokens,
            ),
        }
        if pred_action_logits is not None
        else {}
    )


def compute_fuse_metrics(pred_logits, labels, tokenizer_config, log_segment_prefix):
    batch_size, seq_len, num_logits = pred_logits.shape

    # extract text tokens
    end_of_text_token = repeat(
        jnp.argmax(labels == tokenizer_config.begin_of_action_token, axis=-1),
        "b-> b s",
        s=seq_len,
    )
    is_prompt_mask = (
        repeat(jnp.arange(seq_len), "i -> b i", b=batch_size) < end_of_text_token
    )

    pred_tokens = jnp.argmax(pred_logits, axis=-1)
    acc = (pred_tokens == labels) * is_prompt_mask
    total_acc = jnp.sum(acc, axis=-1) / jnp.sum(is_prompt_mask, axis=-1)
    total_acc = jnp.mean(total_acc)

    metrics = {f"{log_segment_prefix}acc": total_acc,} | {
        f"_details/{log_segment_prefix}acc_{i}": jnp.sum(acc[:, i])
        / (jnp.sum(is_prompt_mask[:, i]) + 1e-5)
        for i in range(10)
    }
    return metrics


def compute_stats(
    *,
    detokenize_fn,
    pred_logits,
    tokens,
    actions,
    mask_loss,
    tokenizer_config: Tokenizer.TokenizerConfig,
    log_segment_prefix: str = "",
    fuse_step: bool = False,
):
    output_pred_mask = mask_loss[..., 1:]
    labels = tokens[..., 1:]

    if not fuse_step:
        action_token_info = get_action_tokens(
            pred_logits,
            labels,
            tokenizer_config.num_action_tokens,
            tokenizer_config.begin_of_action_token,
        )
        metrics = compute_action_metrics(
            detokenize_fn,
            **action_token_info,
            action_dim=actions.shape[-1],
            gt_actions=actions,
            tokenizer_config=tokenizer_config,
            log_segment_prefix=log_segment_prefix,
        )
    else:
        metrics = compute_fuse_metrics(
            pred_logits=pred_logits,
            labels=labels,
            tokenizer_config=tokenizer_config,
            log_segment_prefix=log_segment_prefix,
        )

    loss = jnp.mean(
        output_pred_mask
        * optax.softmax_cross_entropy_with_integer_labels(pred_logits, labels)
    ) / jnp.mean(output_pred_mask)
    metrics[f"{'fuse_' if fuse_step else ''}loss"] = loss

    return loss, metrics


def step_fn(
    detokenize_fn,
    train: bool,
    tokenizer_config: Tokenizer.TokenizerConfig,
    fuse_step: bool,
    train_state: TrainState,
    batch: TrainingBatch,
    key: chex.PRNGKey,
):
    def loss_fn(params, batch: TrainingBatch, key: chex.PRNGKey):
        all_inputs = batch.sensors | {"text": batch.tokens[..., :-1]}
        if fuse_step:
            sensor_masks = batch.modal_mask | {
                "modality_idx": jnp.squeeze(
                    jnp.ones_like(batch.sensors["modality_idx"], dtype=jnp.bool_),
                    axis=-1,
                ),
            }
            mask_loss = enforce_valid_language_instruction(
                batch.tokens_loss_fuse,
                batch.modality_idx,
                jnp.squeeze(batch.mic_mask, axis=-1),
            )
            ar_mask = batch.tokens_ar_fuse
        else:
            sensor_masks = batch.sensors_mask | {
                "modality_idx": jnp.squeeze(
                    jnp.zeros_like(batch.sensors["modality_idx"], dtype=jnp.bool_),
                    axis=-1,
                ),
            }
            mask_loss = batch.tokens_loss
            ar_mask = batch.tokens_ar
        all_masks = sensor_masks | {
            "text": jnp.ones_like(batch.tokens[..., :-1], dtype=jnp.bool_)
        }

        logits, _ = train_state.apply_fn(
            {"params": params},
            all_inputs,
            data_masks=all_masks,
            text_ar_mask=ar_mask,
            train=train,
            rngs={"dropout": key},
        )

        return compute_stats(
            detokenize_fn=partial(detokenize_fn, obs=batch.sensors),
            pred_logits=logits,
            tokens=batch.tokens,
            actions=batch.actions,
            mask_loss=mask_loss,
            tokenizer_config=tokenizer_config,
            log_segment_prefix="train/tf_" if not fuse_step else "train/fuse_",
            fuse_step=fuse_step,
        )

    grad_fn = jax.grad(loss_fn, has_aux=True)

    key, dropout_key = jax.random.split(key)
    grads, info = grad_fn(train_state.params, batch, dropout_key)
    updates, opt_state = train_state.tx.update(
        grads, train_state.opt_state, params=train_state.params
    )
    params = optax.apply_updates(train_state.params, updates)

    train_state = train_state.replace(
        params=params, opt_state=opt_state, step=train_state.step + 1
    )

    info = info | train_state.opt_state.hyperparams

    def _norm_info(values, prefix):
        components = components_by_label(values)
        result = {f"{prefix}_{k}": optax.global_norm(v) for k, v in components.items()}
        result[prefix] = jnp.sqrt(sum(x**2 for x in result.values()))
        return result

    info = (
        info
        | _norm_info(grads, "norm/grad")
        | _norm_info(updates, "norm/update")
        | _norm_info(train_state.params, "norm/param")
    )

    return train_state, info, key
