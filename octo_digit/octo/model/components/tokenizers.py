import importlib
import logging
import re
from typing import Any, Dict, Optional, Sequence, Union

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
from jax.scipy.stats import norm
import numpy as np
from octo.model.components.base import TokenGroup
from octo.model.components.transformer import MAPHead
from octo.utils.spec import ModuleSpec

EPS = 1e-6


def generate_proper_pad_mask(
    tokens: jax.Array,
    pad_mask_dict: Optional[Dict[str, jax.Array]],
    keys: Sequence[str],
) -> jax.Array:
    if pad_mask_dict is None:
        logging.warning("No pad_mask_dict found. Nothing will be masked.")
        return jnp.ones(tokens.shape[:-1])
    if not all([key in pad_mask_dict for key in keys]):
        logging.warning(
            f"pad_mask_dict missing keys {set(keys) - set(pad_mask_dict.keys())}."
            "Nothing will be masked."
        )
        return jnp.ones(tokens.shape[:-1])

    pad_mask = jnp.stack([pad_mask_dict[key] for key in keys], axis=-1)
    pad_mask = jnp.any(pad_mask, axis=-1)
    pad_mask = jnp.broadcast_to(pad_mask[..., None], tokens.shape[:-1])
    return pad_mask


class TokenLearner(nn.Module):
    """
    Learns to map fixed-length sequence of tokens into specified number of tokens.

    Args:
        num_tokens (int): Number of output tokens.
        bottleneck_dim (int): Size of the hidden layers of the mapping MLP.
        dropout_rate (float): Rate of dropout applied in the mapping MLP. Defaults to no dropout.
    """

    num_tokens: int

    @nn.compact
    def __call__(self, inputs, train: bool = True):
        pos_embed = self.param(
            "pos_embed",
            nn.initializers.normal(stddev=0.02),
            (inputs.shape[-2], inputs.shape[-1]),
        )
        x = inputs + jnp.broadcast_to(pos_embed, inputs.shape)
        x = nn.LayerNorm()(x)
        return MAPHead(num_readouts=self.num_tokens)(x, train=train)


def regex_match(regex_keys, x):
    return any([re.match(r_key, x) for r_key in regex_keys])


def regex_filter(regex_keys, xs):
    return list(filter(lambda x: regex_match(regex_keys, x), xs))


class ImageTokenizer(nn.Module):
    """Image tokenizer that encodes image stack into tokens with optional FiLM conditioning.

    Args:
        encoder (ModuleSpec): Encoder class.
        use_token_learner (bool): Whether to use token learner. Defaults to False.
        num_tokens (int): Number of output tokens, only enforced when use_token_learner is True.
        obs_stack_keys (Sequence[str]): Which spatial observation inputs get stacked for encoder input. Supports regex.
        task_stack_keys (Sequence[str]): Which spatial task inputs get stacked for encoder input. Supports regex.
        task_film_keys (Sequence[str]): Which non-spatial task keys get passed into FiLM conditioning. Supports regex.
    """

    encoder: ModuleSpec
    use_token_learner: bool = False
    num_tokens: int = 8
    conditioning_type: str = "none"
    obs_stack_keys: Sequence[str] = ("image_.*", "depth_.*")
    task_stack_keys: Sequence[str] = tuple()
    task_film_keys: Sequence[str] = tuple()
    proper_pad_mask: bool = True
    exclude_backgrounds: bool = True
    add_channel_dim: bool = False
    repeat_channel_dim: bool = False

    @nn.compact
    def __call__(
        self,
        observations,
        tasks=None,
        train: bool = True,
    ):
        def extract_inputs(keys, inputs, check_spatial=False):
            extracted_outputs = []
            for key in keys:
                obs_input = inputs[key]
                if self.add_channel_dim:
                    obs_input = obs_input[..., None]
                elif self.repeat_channel_dim:
                    obs_input = jnp.broadcast_to(
                        obs_input[..., None], obs_input.shape + (3,)
                    )
                if check_spatial:
                    assert len(obs_input.shape) >= 4
                extracted_outputs.append(obs_input)
            return jnp.concatenate(extracted_outputs, axis=-1)

        obs_stack_keys = regex_filter(self.obs_stack_keys, sorted(observations.keys()))
        if self.exclude_backgrounds:
            obs_stack_keys = [
                key for key in obs_stack_keys if not key.endswith("background")
            ]
        if len(obs_stack_keys) == 0:
            logging.info(
                f"No image inputs matching {self.obs_stack_keys} were found."
                "Skipping tokenizer entirely."
            )
            assert self.proper_pad_mask, "Cannot skip unless using proper_pad_mask."
            return None

        # stack all spatial observation and task inputs
        enc_inputs = extract_inputs(obs_stack_keys, observations, check_spatial=True)
        if self.task_stack_keys:
            needed_task_keys = regex_filter(self.task_stack_keys, observations.keys())
            # if any task inputs are missing, replace with zero padding (TODO: be more flexible)
            for k in needed_task_keys:
                if k not in tasks:
                    logging.info(
                        f"No task inputs matching {k} were found. Replacing with zero padding."
                    )
                    tasks = flax.core.copy(
                        tasks, {k: jnp.zeros_like(observations[k][:, 0])}
                    )
            task_stack_keys = regex_filter(self.task_stack_keys, sorted(tasks.keys()))
            if len(task_stack_keys) == 0:
                raise ValueError(
                    f"No task inputs matching {self.task_stack_keys} were found."
                )
            task_inputs = extract_inputs(task_stack_keys, tasks, check_spatial=True)
            task_inputs = task_inputs[:, None].repeat(enc_inputs.shape[1], axis=1)
            enc_inputs = jnp.concatenate([enc_inputs, task_inputs], axis=-1)
        b, t, h, w, c = enc_inputs.shape
        enc_inputs = jnp.reshape(enc_inputs, (b * t, h, w, c))

        # extract non-spatial FiLM inputs
        encoder_input_kwargs = {}
        if self.task_film_keys:
            film_inputs = extract_inputs(self.task_film_keys, tasks)
            film_inputs = film_inputs[:, None].repeat(t, axis=1)
            encoder_input_kwargs.update(
                {"cond_var": jnp.reshape(film_inputs, (b * t, -1))}
            )

        # run visual encoder
        encoder_def = ModuleSpec.instantiate(self.encoder)()
        image_tokens = encoder_def(
            enc_inputs, **encoder_input_kwargs
        )  # (b * t, n_tok, dim) for encoders from vit_encoders, (b*t, dim) for encoders from tvl_vit
        if image_tokens.ndim == 2:
            image_tokens = image_tokens[:, None, :]

        image_tokens = jnp.reshape(image_tokens, (b, t, -1, image_tokens.shape[-1]))

        if self.use_token_learner:
            image_tokens = TokenLearner(num_tokens=self.num_tokens)(
                image_tokens, train=train
            )

        if self.proper_pad_mask:
            pad_mask = generate_proper_pad_mask(
                image_tokens,
                observations.get("pad_mask_dict", None),
                obs_stack_keys,
            )
        else:
            pad_mask = jnp.ones(image_tokens.shape[:-1])
        return TokenGroup(image_tokens, pad_mask)


class ImageTokenizerConcatTokens(nn.Module):
    """Image tokenizer that encodes image stack into tokens with optional FiLM conditioning.

    Args:
        encoder (ModuleSpec): Encoder class.
        use_token_learner (bool): Whether to use token learner. Defaults to False.
        num_tokens (int): Number of output tokens, only enforced when use_token_learner is True.
        obs_stack_keys (Sequence[str]): Which spatial observation inputs get stacked for encoder input. Supports regex.
        task_stack_keys (Sequence[str]): Which spatial task inputs get stacked for encoder input. Supports regex.
        task_film_keys (Sequence[str]): Which non-spatial task keys get passed into FiLM conditioning. Supports regex.
    """

    encoder: ModuleSpec
    use_token_learner: bool = False
    num_tokens: int = 8
    conditioning_type: str = "none"
    obs_stack_keys: Sequence[str] = ("image_.*", "depth_.*")
    task_stack_keys: Sequence[str] = tuple()
    task_film_keys: Sequence[str] = tuple()
    proper_pad_mask: bool = True
    exclude_backgrounds: bool = True
    add_channel_dim: bool = False

    @nn.compact
    def __call__(
        self,
        observations,
        tasks=None,
        train: bool = True,
    ):
        def extract_inputs(keys, inputs, check_spatial=False):
            extracted_outputs = []
            for key in keys:
                obs_input = inputs[key]
                if self.add_channel_dim:
                    obs_input = obs_input[..., None]
                if check_spatial:
                    assert len(obs_input.shape) >= 4
                extracted_outputs.append(obs_input)
            return extracted_outputs

        obs_stack_keys = regex_filter(self.obs_stack_keys, sorted(observations.keys()))
        if self.exclude_backgrounds:
            obs_stack_keys = [
                key for key in obs_stack_keys if not key.endswith("background")
            ]
        if len(obs_stack_keys) == 0:
            logging.info(
                f"No image inputs matching {self.obs_stack_keys} were found."
                "Skipping tokenizer entirely."
            )
            assert self.proper_pad_mask, "Cannot skip unless using proper_pad_mask."
            return None

        # stack all spatial observation and task inputs
        enc_inputs = extract_inputs(obs_stack_keys, observations, check_spatial=True)
        if self.task_stack_keys:
            raise NotImplementedError

        token_outputs = []
        pad_masks = []
        encoder_def = ModuleSpec.instantiate(self.encoder)()
        for enc_input in enc_inputs:
            b, t, h, w, c = enc_input.shape
            enc_input = jnp.reshape(enc_input, (b * t, h, w, c))

            # extract non-spatial FiLM inputs
            encoder_input_kwargs = {}
            if self.task_film_keys:
                raise ValueError

            # run visual encoder
            image_tokens = encoder_def(
                enc_input, **encoder_input_kwargs
            )  # (b * t, n_tok, dim) for encoders from vit_encoders, (b*t, dim) for encoders from tvl_vit
            if image_tokens.ndim == 2:
                image_tokens = image_tokens[:, None, :]

            image_tokens = jnp.reshape(image_tokens, (b, t, -1, image_tokens.shape[-1]))

            if self.use_token_learner:
                image_tokens = TokenLearner(num_tokens=self.num_tokens)(
                    image_tokens, train=train
                )
            token_outputs.append(image_tokens)
            if self.proper_pad_mask:
                pad_mask = generate_proper_pad_mask(
                    image_tokens,
                    observations.get("pad_mask_dict", None),
                    obs_stack_keys,
                )
            else:
                pad_mask = jnp.ones(image_tokens.shape[:-1])
            pad_masks.append(pad_mask)
        image_tokens = jnp.concatenate(token_outputs, axis=-2)
        pad_mask = jnp.concatenate(pad_masks, axis=-1)
        return TokenGroup(image_tokens, pad_mask)


class UnsqueezingImageTokenizer(ImageTokenizer):
    """Wrapper around ImageTokenizer; unsqueezes last dimension of a 1-channel image (e.g. spectrogram or depth).

    Args:
        encoder (ModuleSpec): Encoder class.
        use_token_learner (bool): Whether to use token learner. Defaults to False.
        num_tokens (int): Number of output tokens, only enforced when use_token_learner is True.
        obs_stack_keys (Sequence[str]): Which spatial observation inputs get stacked for encoder input. Supports regex.
        task_stack_keys (Sequence[str]): Which spatial task inputs get stacked for encoder input. Supports regex.
        task_film_keys (Sequence[str]): Which non-spatial task keys get passed into FiLM conditioning. Supports regex.
    """

    encoder: ModuleSpec
    use_token_learner: bool = False
    num_tokens: int = 8
    conditioning_type: str = "none"
    obs_stack_keys: Sequence[str] = ("image_.*", "depth_.*")
    task_stack_keys: Sequence[str] = tuple()
    task_film_keys: Sequence[str] = tuple()
    proper_pad_mask: bool = True

    @nn.compact
    def __call__(
        self,
        observations,
        tasks=None,
        train: bool = True,
    ):
        obs_stack_keys = regex_filter(self.obs_stack_keys, sorted(observations.keys()))
        for key in obs_stack_keys:
            obs = observations[key]
            if len(obs.shape) == 4:  # reshape spectrogram
                observations[key] = obs[..., None]
        return super().__call__(observations, tasks=tasks, train=train)


class LanguageTokenizer(nn.Module):
    """
    Language tokenizer that embeds text input IDs into continuous language embeddings. Supports pre-trained HF models.

     Args:
         num_tokens (int): Number of output tokens (not enforced).
         encoder (str, optional): Optional HuggingFace AutoModel name for encoding input IDs.
         finetune_encoder (bool, optional): Optional finetune last layers of the language model.
    """

    encoder: str = None
    finetune_encoder: bool = False
    proper_pad_mask: bool = True
    repeat_tokens_window: Union[int, None] = None

    def setup(self):
        if self.encoder is not None:
            from transformers import AutoConfig, FlaxAutoModel, FlaxT5EncoderModel

            config = AutoConfig.from_pretrained(self.encoder)
            if "t5" in self.encoder:
                self.hf_model = FlaxT5EncoderModel(config).module
            else:
                self.hf_model = FlaxAutoModel.from_config(config).module

    def __call__(
        self,
        observations,
        tasks=None,
        train: bool = True,
    ):
        if "language_instruction" not in tasks:
            logging.warning("No language inputs found. Skipping tokenizer entirely.")
            assert self.proper_pad_mask, "Cannot skip unless using proper pad mask."
            return None

        if not isinstance(tasks["language_instruction"], (jax.Array, np.ndarray)):
            assert (
                self.encoder is not None
            ), "Received language tokens but no encoder specified."
            tokens = self.hf_model(**tasks["language_instruction"]).last_hidden_state
        else:
            # add a # tokens dimension to language
            if tasks["language_instruction"].ndim == 2:
                tokens = tasks["language_instruction"][:, None, :]
            else:
                tokens = tasks["language_instruction"]

            if self.repeat_tokens_window:
                tokens = tokens[:, None, :, :]  # add window dimension
                tokens = jnp.tile(tokens, (1, self.repeat_tokens_window, 1, 1))

        if not self.finetune_encoder:
            tokens = jax.lax.stop_gradient(tokens)

        # TODO: incorporate padding info from language tokens here too
        if self.proper_pad_mask:
            pad_mask = generate_proper_pad_mask(
                tokens,
                tasks.get("pad_mask_dict", None),
                ("language_instruction",),
            )
        else:
            pad_mask = jnp.ones(tokens.shape[:-1])

        return TokenGroup(tokens, pad_mask)


class BinTokenizer(nn.Module):
    """
    Tokenizes continuous inputs via dimension-wise binning in given range.

    Args:
        n_bins (int): Number of discrete bins per dimension.
        bin_type (str): Type of binning. ['uniform', 'normal' = Gaussian]
        low (float): Lower bound for bin range.
        high (float): Upper bound for bin range.
    """

    n_bins: int
    bin_type: str = "uniform"
    low: float = 0
    high: float = 1

    def setup(self):
        if self.bin_type == "uniform":
            self.thresholds = jnp.linspace(self.low, self.high, self.n_bins + 1)
        elif self.bin_type == "normal":
            self.thresholds = norm.ppf(jnp.linspace(EPS, 1 - EPS, self.n_bins + 1))
        else:
            raise ValueError(
                f"Binning type {self.bin_type} not supported in BinTokenizer."
            )

    def __call__(self, inputs):
        if self.bin_type == "uniform":
            inputs = jnp.clip(inputs, self.low + EPS, self.high - EPS)
        inputs = inputs[..., None]
        token_one_hot = (inputs < self.thresholds[1:]) & (
            inputs >= self.thresholds[:-1]
        ).astype(jnp.uint8)
        output_tokens = jnp.argmax(token_one_hot, axis=-1)
        return output_tokens

    def decode(self, inputs):
        one_hot = jax.nn.one_hot(inputs, self.n_bins)
        bin_avgs = (self.thresholds[1:] + self.thresholds[:-1]) / 2
        outputs = jnp.sum(one_hot * bin_avgs, axis=-1)
        return outputs


class LowdimObsTokenizer(BinTokenizer):
    """
    Tokenizer for non-spatial observations. Optionally discretizes into bins per dimension (see BinTokenizer).

    Args:
        obs_keys (Sequence[str]): List of non-spatial keys to concatenate & tokenize. Supports regex.
        discretize (bool): If True, discretizes inputs per dimension, see BinTokenizer.
    """

    obs_keys: Sequence[str] = tuple()
    discretize: bool = False
    proper_pad_mask: bool = True

    def __call__(self, observations, *unused_args, **unused_kwargs):
        assert self.obs_keys, "Need to specify observation keys to tokenize."
        if len(regex_filter(self.obs_keys, sorted(observations.keys()))) == 0:
            logging.warning(
                f"No observation inputs matching {self.obs_keys} were found."
                "Skipping tokenizer entirely."
            )
            assert self.proper_pad_mask, "Cannot skip unless using proper pad mask."
            return None

        tokenizer_inputs = []
        for o_key in self.obs_keys:
            for key in filter(re.compile(o_key).match, sorted(observations.keys())):
                assert (
                    len(observations[key].shape) == 3
                ), f"Only supports non-spatial inputs but {key} has shape {observations[key].shape}."
                tokenizer_inputs.append(observations[key])
        tokenizer_inputs = jnp.concatenate(tokenizer_inputs, axis=-1)
        if self.discretize:
            tokenized_inputs = super().__call__(tokenizer_inputs)
            tokens = jax.nn.one_hot(tokenized_inputs, self.n_bins)
        else:
            tokens = tokenizer_inputs[..., None]
        mask = jnp.ones(tokens.shape[:-1])
        return TokenGroup(tokens, mask)


class ProjectionTokenizer(LowdimObsTokenizer):
    """
    Wrapper to apply TokenLearner dimension reduction to LowdimObs for higher-dimension, non-spatial observations

    Args:
        obs_keys (Sequence[str]): List of non-spatial keys to concatenate & tokenize. Supports regex.
        discretize (bool): If True, discretizes inputs per dimension, see BinTokenizer.
    """

    num_output_tokens: int = 7
    obs_keys: Sequence[str] = tuple()
    num_hidden_layers: int = 1
    hidden_size: int = 512
    discretize: bool = False
    proper_pad_mask: bool = True

    @nn.compact
    def __call__(self, observations, *unused_args, **unused_kwargs):
        obs_size = None
        projection_inputs = []
        projection_keys = {}
        for o_key in self.obs_keys:
            for i, key in enumerate(
                filter(re.compile(o_key).match, sorted(observations.keys()))
            ):
                curr_obs_size = observations[key].shape[-1]
                assert (
                    obs_size is None or obs_size == curr_obs_size
                ), "All observations must have same size."
                obs_size = curr_obs_size
                projection_inputs.append(observations[key])
                projection_keys[key] = i

        projection = jnp.stack(projection_inputs)
        for i in range(self.num_hidden_layers):
            projection = nn.Dense(features=self.hidden_size)(projection)
            projection = nn.relu(projection)
        projection = nn.Dense(features=self.num_output_tokens)(projection)

        pass_observations = {}
        for key in observations.keys():
            if key in projection_keys:
                pass_observations[key] = projection[projection_keys[key]]
            else:
                pass_observations[key] = observations[key]
        return super().__call__(pass_observations, *unused_args, **unused_kwargs)


class SiglipTokenizer(LowdimObsTokenizer):
    image: Optional[Any] = None  # dict(variant='B/16', pool_type='map')
    image_model: str = "vit"
    encoder_path: str = "/home/sjosh/nfs/octo_digit/siglip.npz:img"
    num_output_tokens: int = 7
    obs_stack_keys: Sequence[str] = "image_digit_left"
    obs_keys = ("siglip",)
    num_hidden_layers: int = 1
    proper_pad_mask: bool = True

    @nn.compact
    def __call__(
        self,
        observations,
        tasks=None,
        train: bool = True,
        *unused_args,
        **unused_kwargs,
    ):
        def extract_inputs(keys, inputs, check_spatial=False):
            extracted_outputs = []
            for key in keys:
                if check_spatial:
                    assert len(inputs[key].shape) >= 4
                extracted_outputs.append(inputs[key])
            return jnp.concatenate(extracted_outputs, axis=-1)

        obs_stack_keys = regex_filter(self.obs_stack_keys, sorted(observations.keys()))
        if len(obs_stack_keys) == 0:
            logging.info(
                f"No image inputs matching {self.obs_stack_keys} were found."
                "Skipping tokenizer entirely."
            )
            assert self.proper_pad_mask, "Cannot skip unless using proper_pad_mask."
            return None

        # stack all spatial observation and task inputs
        enc_inputs = extract_inputs(obs_stack_keys, observations, check_spatial=True)
        b, t, h, w, c = enc_inputs.shape
        enc_inputs = jnp.reshape(enc_inputs, (b * t, h, w, c))

        enc_inputs = enc_inputs * 2.0 / 255.0 - 1.0

        image_model = importlib.import_module(
            f"big_vision.models.{self.image_model}"
        ).Model(**{"num_classes": None, **(self.image or {})}, name="img")
        zimg, _ = image_model(enc_inputs)
        zimg = zimg.reshape((b, t, -1))
        pass_observations = {self.obs_keys[0]: zimg}
        return super().__call__(pass_observations, *unused_args, **unused_kwargs)


class IdentityObsTokenizer(nn.Module):
    """
    Tokenizer that simply collects pre-computed tokens.

    Args:
        obs_keys (Sequence[str]): List of non-spatial keys to concatenate & tokenize. Supports regex.
        discretize (bool): If True, discretizes inputs per dimension, see BinTokenizer.
    """

    obs_keys: Sequence[str] = tuple()
    discretize: bool = False
    proper_pad_mask: bool = True
    strict_match: bool = True

    def __call__(self, observations, *unused_args, **unused_kwargs):
        assert self.obs_keys, "Need to specify observation keys to tokenize."

        def matching_observation_keys(obs_keys):
            if self.strict_match:
                return [key for key in self.obs_keys if key in obs_keys]
            else:
                obs_keys = []
                for o_key in self.obs_keys:
                    obs_keys.extend(
                        list(
                            filter(re.compile(o_key).match, sorted(observations.keys()))
                        )
                    )

        if len(matching_observation_keys(observations.keys())) == 0:
            logging.warning(
                f"No observation inputs matching {self.obs_keys} were found."
                "Skipping tokenizer entirely."
            )
            assert self.proper_pad_mask, "Cannot skip unless using proper pad mask."
            return None

        tokenizer_inputs = []

        for key in matching_observation_keys(observations.keys()):
            assert (
                len(observations[key].shape) == 3
            ), f"Only supports non-spatial inputs but {key} has shape {observations[key].shape}."
            tokenizer_inputs.append(observations[key])
        tokenizer_inputs = jnp.concatenate(tokenizer_inputs, axis=-1)
        tokens = tokenizer_inputs[:, :, None, :]
        mask = jnp.ones(tokens.shape[:-1])
        return TokenGroup(tokens, mask)
