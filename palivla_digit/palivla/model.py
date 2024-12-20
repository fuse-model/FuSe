from functools import partial
from typing import Dict, Sequence, Tuple

from big_vision.models.proj.paligemma.gemma_bv import Model as GemmaModel
from big_vision.models.proj.paligemma.paligemma import make_attn_mask
from big_vision.models.vit import Model as ViTModel
import chex
import einops
from einops import repeat
import flax
from flax.core import freeze, FrozenDict
import flax.linen as nn
import jax
import jax.numpy as jnp
from palivla.spec import ModuleSpec
from palivla.types import Data, Info
from scalax.sharding import MeshShardingHelper, PartitionSpec, ShardingRule


def make_gather_indices(
    total_num_tokens: int, embeds_sizes: jax.Array, reordering: jax.Array
):
    """
    Creates gather indices for reordering token groups.

    Args:
        embeds_sizes: Array of shape (batch_size, num_token_groups) containing the number of tokens in each group.
        reordering: Array of shape (batch_size, num_token_groups) containing the new order of token groups.

    Returns:
        gather_indices: Array of shape (batch_size, total_num_tokens) containing the indices to gather tokens in the new order.

    Example:
        total_num_tokens = 8
        embeds_sizes = [3, 4, 1]  # text: 3, image: 4, proprio: 1
        reordering = [[2, 1, 0]]    # new order: proprio, image, text
        Result: [7, 3, 4, 5, 6, 0, 1, 2]

    Method:
     - First, find the start and end of each token group in the original order and in the new order.
     - For each token group find the delta between the start of the token group in the new order and the start of the token group in the original order.
     - For each index in the output, which token group it belongs to in the new order, and what is the offset delta of this group.
     - The output indices are then the original indices of the tokens in the token group, plus the delta of the token group.
    """
    inverse_reordering = jnp.argsort(reordering, stable=True)

    # Find the embed sizes, reordered for the new token group order
    embeds_sizes_reordered = embeds_sizes[reordering]

    # Calculate cumulative sizes for original and reordered embeddings
    cumulative_sizes = jnp.cumsum(embeds_sizes)
    cumulative_sizes_reordered = jnp.cumsum(embeds_sizes_reordered)

    # Pad with zeros at the beginning for easier indexing
    cumulative_sizes = jnp.pad(cumulative_sizes, ((1, 0)))
    cumulative_sizes_reordered = jnp.pad(cumulative_sizes_reordered, ((1, 0)))

    offsets_original = cumulative_sizes[:-1]
    offsets_reordered = cumulative_sizes_reordered[:-1][inverse_reordering]

    offset_delta = offsets_original - offsets_reordered

    # Find which group each token belongs to in the reordered sequence
    membership_in_reordered_groups = (
        jnp.sum(
            jnp.arange(total_num_tokens)[:, None]
            >= cumulative_sizes_reordered[None, :],
            axis=-1,
        )
        - 1
    )
    membership_in_original_groups = reordering[membership_in_reordered_groups]

    return jnp.arange(total_num_tokens) + offset_delta[membership_in_original_groups]


def pack_embeddings(embeds: jax.Array, masks: jax.Array):
    idcs = jnp.argsort(~masks, stable=True)
    return embeds[idcs]


def collect_embeddings(
    embeds: Dict[str, jax.Array],
    embed_masks: Dict[str, jax.Array],
    language_embeds: jax.Array,
    language_masks: jax.Array,
    language_ar_mask: jax.Array,
    *,
    rng: jax.Array = None,
    target_key_order: Sequence[str] | None = None,
):
    """
    Collects embeddings for a single sample (with no batch axis), randomly reordered and packed according to masks.

    Will always include the language embeddings last.

    Params:
        embeds: Dictionary of embeddings, keyed by modality.
        embed_masks: Dictionary of masks, keyed by modality.
        language_embeds: Language embeddings.
        language_masks: Language masks.
        language_ar_mask: Language AR mask.
        rng: Random number generator (optional, unused if target_key_order is provided).
        target_key_order: Optional list of modalities to collect embeddings for. If None, will randomize the order (optional). Should not include "text", which will be added last.
    """
    # Make sure we don't have any batch axes.
    _, embed_dim = language_embeds.shape
    chex.assert_shape([language_embeds, *embeds.values()], (None, embed_dim))
    chex.assert_shape([language_masks, *embed_masks.values()], (None,))

    # Randomly reorder the embeddings.
    key_order = list(sorted(embeds.keys())) + ["text"]
    if target_key_order is not None:
        sequence_order = jnp.array([key_order.index(k) for k in target_key_order])
    else:
        assert rng is not None, "Must provide rng if target_key_order is not provided."
        sequence_order = jax.random.permutation(rng, len(embeds))

    # Add the language embeddings to the _end_ of the embeds and masks.
    embeds = embeds | {"text": language_embeds}
    embed_masks = embed_masks | {"text": language_masks}
    ar_masks = {
        k: jnp.zeros(v.shape[0], dtype=jnp.bool_) for k, v in embeds.items()
    } | {"text": language_ar_mask}
    sequence_order = jnp.concatenate([sequence_order, jnp.array([len(embeds) - 1])])

    # Number of tokens in each embed.
    embed_sizes = [embeds[k].shape[0] for k in key_order]
    total_num_tokens = sum(embed_sizes)

    # Randomly reorder the embeddings.
    # The last token group is the language tokens, which we want to attend to with full self-attention.
    gather_indices = make_gather_indices(
        total_num_tokens, jnp.array(embed_sizes), jnp.array(sequence_order)
    )

    concat_masks = jnp.concatenate([embed_masks[k] for k in key_order], axis=0)
    group_ids = {k: jnp.full(embeds[k].shape[0], i) for i, k in enumerate(key_order)}

    def _gather_values(data, indices):
        concat_data = jnp.concatenate([data[k] for k in key_order], axis=0)
        return pack_embeddings(concat_data[indices], concat_masks[indices])

    # Pack the embeddings according to the masks, and compute the start indices as the first unmasked index in the packed sequence.
    packed_embeds = _gather_values(embeds, gather_indices)
    packed_masks = _gather_values(embed_masks, gather_indices)
    packed_ar = _gather_values(ar_masks, gather_indices)
    packed_group_membership = jnp.where(
        packed_masks, _gather_values(group_ids, gather_indices), -1
    )

    # Find the first and last (+1) true element
    def _find_first_last(x):
        return jnp.argmax(x, axis=0), x.shape[0] - jnp.argmax(x[::-1], axis=0)

    groups = {
        k: _find_first_last(packed_group_membership == i)
        for i, k in enumerate(key_order)
    }

    return packed_embeds, packed_masks, packed_ar, groups


def get_default_config():
    return {
        "llm_spec": {
            "__ctor": "big_vision.models.proj.paligemma.gemma_bv.Model",
            "config": {"vocab_size": 257_152},
        },
        "img_spec": {
            "__ctor": "big_vision.models.vit.Model",
            "config": {"variant": "So400m/14", "pool_type": "none", "scan": True},
        },
        "encoder_specs": {},
        "modality_mappings": {"image_primary": "img", "text": "llm"},
        "prompt_autoregressive": False,
        "target_key_order": ("image_primary",),
    }


class PaliVLAModel(nn.Module):
    # Specifications for the basic modules
    llm_spec: ModuleSpec
    img_spec: ModuleSpec

    # Encoders by name
    # Should exclude the "image" modality.
    encoder_specs: FrozenDict[str, ModuleSpec]

    # Mapping from modality to embedding
    modality_mappings: FrozenDict[str, str]

    prompt_autoregressive: bool
    target_key_order: Sequence[str]

    def setup(self):
        self.llm: GemmaModel = self.llm_spec.instantiate(name="llm")
        self.image: ViTModel = self.img_spec.instantiate(
            name="img", num_classes=self.llm.embdim
        )

        def _encode_image(data):
            # Shim to process the stack dimension as a batch dimension
            orig_shape = data.shape
            if len(orig_shape) == 5:
                data = einops.rearrange(data, "... h w c -> (...) h w c")
            result, out = self.image(data)
            if len(orig_shape) == 5:
                result = einops.rearrange(
                    result, "(b k) t d -> b (k t) d", b=orig_shape[0]
                )
            return result, out

        self.encoders = {"llm": self.llm.embed_tokens, "img": _encode_image,} | {
            name: spec.instantiate(name=name)
            for name, spec in self.encoder_specs.items()
        }

        self.modality_start_tokens = {
            modality: self.param(
                name=f"start_{modality}",
                init_fn=nn.linear.default_embed_init,
                shape=(
                    1,
                    self.llm.embdim,
                ),
            )
            for modality in self.modality_mappings.keys()
        }

    def embed_modalities(
        self, data: Data, masks: Dict[str, jax.Array]
    ) -> Tuple[Data, Data, Info]:
        """
        Get embeddings for all modalities.
        """

        embeds = {}
        embed_masks = {}
        info = {}

        for modality, encoder_name in self.modality_mappings.items():
            # Embed the data
            encoder = self.encoders[encoder_name]

            result = encoder(data[modality])

            # Some encoders return a tuple of (embeddings, info), others just return the embeddings.
            if isinstance(result, tuple):
                embeds[modality], m_info = result
            else:
                embeds[modality], m_info = result, {}

            m_info["embed"] = embeds[modality]

            # If only one embedding was produced, expand it to match the sequence length
            if embeds[modality].ndim == 2:
                embeds[modality] = jnp.expand_dims(embeds[modality], axis=1)

            # Add the modality start tokens
            start_token = einops.repeat(
                self.modality_start_tokens[modality],
                "1 e -> b 1 e",
                b=data[modality].shape[0],
            )
            embeds[modality] = jnp.concatenate([start_token, embeds[modality]], axis=1)

            # Create a mask to match the shape of the embeddings
            # Add one to the length to account for the modality start tokens
            if masks[modality].ndim < embeds[modality].ndim - 1:
                embed_masks[modality] = repeat(
                    masks[modality],
                    "... -> ... t",
                    t=embeds[modality].shape[1],
                )
            else:
                mask = masks[modality]
                mask = jnp.concatenate([mask[:, :1], mask], axis=1)
                chex.assert_shape(mask, embeds[modality].shape[:-1])
                embed_masks[modality] = mask

            info.update({f"{modality}/{k}": v for k, v in m_info.items()})

        missing_modalities = set(self.modality_mappings.keys()) - set(data.keys())

        if missing_modalities:
            print(f"Warning: missing modalities {missing_modalities}")

        for modality in missing_modalities:
            embeds[modality] = jnp.zeros((0, 0), dtype=jnp.float16)
            embed_masks[modality] = jnp.zeros((0,), dtype=jnp.bool_)

        batch_size = jax.tree.leaves(data)[0].shape[0]
        embed_dim = self.llm.embdim
        chex.assert_shape(list(embeds.values()), (batch_size, None, embed_dim))

        return embeds, embed_masks, info

    def make_model_inputs(
        self,
        data: Data,
        masks: Data | None,
        text_ar_mask: jax.Array,
        target_key_order: Sequence[str] | None = None,
        rng: jax.Array | None = None,
    ):
        data_embeds, data_masks, info = self.embed_modalities(data, masks)

        # Text is handled separately so it always goes last
        embeds_without_text = {k: v for k, v in data_embeds.items() if k != "text"}
        embed_masks_without_text = {k: v for k, v in data_masks.items() if k != "text"}

        if rng is not None:
            batch_size = jax.tree.leaves(data)[0].shape[0]
            rng = jax.random.split(rng, batch_size)

        packed_embeds, packed_masks, packed_ar, groups = jax.vmap(
            partial(collect_embeddings, target_key_order=target_key_order)
        )(
            embeds_without_text,
            embed_masks_without_text,
            data_embeds["text"],
            data_masks["text"],
            text_ar_mask,
            rng=rng,
        )

        return packed_embeds, packed_masks, packed_ar, groups, info

    def __call__(
        self,
        data: Data,
        text_ar_mask: jax.Array,
        *,
        train: bool = False,
        data_masks: Dict[str, jax.Array] | None = None,
        target_key_order: Sequence[str] | None = None,
    ):
        if data_masks is None:
            data_masks = jax.tree.map(lambda _: None, data)
        data_masks = jax.tree.map(
            lambda x, m: jnp.ones(x.shape[0], dtype=jnp.bool_) if m is None else m,
            data,
            data_masks,
        )

        packed_embeds, packed_masks, packed_ar, groups, info = self.make_model_inputs(
            data,
            data_masks,
            text_ar_mask,
            target_key_order=target_key_order,
            rng=self.make_rng("dropout") if target_key_order is None else None,
        )

        attn_mask = make_attn_mask(packed_masks, packed_ar)

        _, llm_info = self.llm(packed_embeds, mask=attn_mask, train=train)

        info = info | {
            f"{modality}_info": m_info for modality, m_info in llm_info.items()
        }

        # Get only the logits for the text tokens, which should be the last `n` tokens
        all_pre_logits = llm_info["pre_logits"]
        text_start, text_end = groups["text"]

        _, text_length_padded = data["text"].shape

        @jax.vmap
        def extract_text_pre_logits(pre_logits, text_start, text_end, mask):
            # Add one to the start index to account for the modality start tokens
            text_pre_logits = jax.lax.dynamic_slice(
                pre_logits,
                (text_start + 1, 0),
                (text_length_padded, pre_logits.shape[1]),
            )
            text_logit_masks = (
                jnp.arange(text_length_padded) < (text_end - text_start - 1)
            ) & mask
            return text_pre_logits, text_logit_masks

        text_pre_logits, text_logit_masks = extract_text_pre_logits(
            all_pre_logits, text_start, text_end, data_masks["text"]
        )
        text_logits = self.llm.compute_logits(text_pre_logits, train=train)

        info["text_logits"] = text_logits
        info["text_tokens"] = jnp.argmax(text_logits, axis=-1)
        info["text_logit_masks"] = text_logit_masks

        return text_logits, info

    def embed_text(self, tokens, train=False):
        out = {}
        ztxt = out["llm/ztxt"] = self.llm.embed_tokens(tokens, train=train)
        return ztxt, out

    def prefill_cache(
        self,
        x: jax.Array,
        input_mask: jax.Array,
        mask_ar: jax.Array,
        *,
        cache_size,
    ):
        """Initializes decoding cache with `x` [B, N, E] as prompt."""
        if hasattr(self.llm, "prefill_cache"):
            attn_mask = make_attn_mask(input_mask, mask_ar)
            return self.llm.prefill_cache(
                x, input_mask, attn_mask, cache_size=cache_size
            )
        else:
            return self._fallback_prefill_cache(x, input_mask, mask_ar, cache_size)

    def extend_cache(self, x):
        """Advances decoding cache with `x` [B, 1, E]."""
        if hasattr(self.llm, "prefill_cache"):
            return self.llm.extend_cache(x)
        else:
            return self._fallback_extend_cache(x)


def load_from_pretrained(
    path,
    model_cfg,
    batch_shape: Dict[str, jax.ShapeDtypeStruct],
    *,
    mesh: MeshShardingHelper | None = None,
    sharding_rule: ShardingRule | PartitionSpec | None = None,
    seed: int = 0,
    param_dtype: jnp.dtype = jnp.float32,
):
    """
    Loads from a pretrained PaliGemma model.
    """
    from big_vision.models.proj.paligemma.paligemma import load as load_paligemma
    from ml_collections import FrozenConfigDict

    model_cfg = FrozenDict(model_cfg)

    base_model_cfg = FrozenConfigDict(
        {
            "llm": {"vocab_size": 257_152},
            "img": {
                "variant": "So400m/14",
                "pool_type": "none",
                "scan": True,
            },
        }
    )
    base_params = load_paligemma(
        None,
        path,
        base_model_cfg,
    )

    model_spec = ModuleSpec.create(PaliVLAModel, model_cfg)
    palivla_model: PaliVLAModel = model_spec.instantiate()

    def _init_params_fn(param_replacements):
        params = palivla_model.lazy_init(
            jax.random.PRNGKey(seed),
            data=batch_shape,
            text_ar_mask=jnp.ones(batch_shape["text"].shape, dtype=jnp.bool_),
        )["params"]

        for k, v in param_replacements.items():
            chex.assert_trees_all_equal_shapes(params[k], v)
            jax.debug.print(f"Replacing param {k}")
            params[k] = v

        flat_params = flax.traverse_util.flatten_dict(params)
        for encoder_name, spec in palivla_model.encoder_specs.items():
            if encoder_name in {"llm", "img"} or spec.load_fn is None:
                continue
            jax.debug.print(f"Replacing params for {encoder_name}")
            loaded_params = spec.load_fn(**spec.load_kwargs)
            for k, param in flat_params.items():
                if k[0] == encoder_name:
                    loaded_key = k[1:]
                    if loaded_key not in loaded_params:
                        jax.debug.print(
                            f"Param {loaded_key} not present in loaded params"
                        )
                        continue
                    loaded_param = jnp.array(loaded_params[loaded_key])
                    assert (
                        param.dtype == loaded_param.dtype
                    ), f"Loaded param had dtype {loaded_param.dtype}, expected {param.dtype}"
                    if param.shape != loaded_param.shape:
                        jax.debug.print(
                            f"Received parameter of shape {loaded_param.shape} when trying to load param for {k}, expected {param.shape}. Skipping."
                        )
                        continue
                    flat_params[k] = loaded_param

        params = flax.traverse_util.unflatten_dict(flat_params)
        params = jax.tree.map(lambda x: x.astype(param_dtype), params)

        return params

    if mesh is not None:
        init_params_fn = mesh.sjit(
            _init_params_fn,
            in_shardings=None,
            out_shardings=sharding_rule,
        )
    else:
        init_params_fn = jax.jit(_init_params_fn)

    init_params = init_params_fn(base_params)
    return model_spec, init_params


#####
