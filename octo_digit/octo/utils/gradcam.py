from optparse import Option
from typing import Dict, Tuple, Optional
from octo.utils.typing import Data, Sequence, JaxArray
from octo.utils.typing import Config, Data, Params, PRNGKey, Perturbations, Sequence, JaxArray
from octo.model.octo_model import OctoModel
import einops
import math
from flax.traverse_util import flatten_dict
from octo.model.components.action_heads import ActionHead
import jax.numpy as jnp 
import logging 
import jax 
import tensorflow as tf
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from flax.core.frozen_dict import freeze


def extract_tokens_from_all(
    extract_keys: Sequence[str], 
    model: OctoModel,
    token_collection: Data, 
    dummy_batch: Data,
    dummy_readouts: Optional[Sequence[str]] = None,
    transformer_outputs: Optional[Data] = None,
) -> Dict[str, Data]: 
    """Util method to extract tokens into tokenizer_name->tokens dict from (batch, total_num_tokens, token_embedding_dim) collection 

    Args:
        extract_keys: Sequence of strings corresponding to tokenizer group names to extract, typically 'obs_primary' or 'obs_wrist'. 
        model: OctoModel with appropriate structure, containing all relevant tokenizers 
        token_collection: Params, Perturbations, or Intermediates-like PyTree from which to extract the tokens. 
        dummy_batch: a batch of data containing the appropriate observation names. Needed only for performing a dummy run of the transformer, to determine the tokenizer order
        dummy_readouts: the readout names used when creating token_collection; similar to dummy_batch 
        transformer_outputs: the output of running the transformer on a dummy batch. If pre-computed, can save computation

    Returns: 
        extracted_tokens: a tokenizer_name -> tokens dict
    """

    dummy_observations = dummy_batch['observation']
    dummy_tasks = dummy_batch['task']
    dummy_pad_mask = dummy_observations['timestep_pad_mask']

    # Get order of tokens
    octo_transformer = model.module.octo_transformer
    if transformer_outputs is None: 
        output, _ = model.module.apply({'params': model.params}, dummy_observations, dummy_tasks, dummy_pad_mask)
    else: 
        output = transformer_outputs
    token_group_names = [] 
    for name, _ in octo_transformer.task_tokenizers.items():
        group_name = f"task_{name}"
        token_group_names.append(group_name)
    for name, _ in octo_transformer.observation_tokenizers.items():
        group_name = f"obs_{name}"
        token_group_names.append(group_name)
    if octo_transformer.repeat_task_tokens:
        for name, _ in octo_transformer.task_tokenizers.items():
            task_name = f"task_{name}"
            group_name = f"obs_{task_name}"
            token_group_names.append(group_name)
    if dummy_readouts is None:
        dummy_readouts = list(octo_transformer.readouts.keys())
    for readout_name in dummy_readouts:
        group_name = f"readout_{readout_name}"
        token_group_names.append(group_name)

    # batch window n_tokens embedding_dim -> batch (window n_tokens) embedding_dim 
    collapsed_name_and_size = [] 
    for name in token_group_names: 
        token_shape = output[name].tokens.shape 
        if len(token_shape) == 4: 
            batch, window, n_tokens, embedding_dim = token_shape 
            token_shape = (batch, window * n_tokens, embedding_dim)
        collapsed_name_and_size.append((name, token_shape))

    # Create name -> assigned tokens mapping 
    batch, window = dummy_observations[list(dummy_observations.keys())[0]].shape[:2] 
    cumulative_shapes = {}
    n_toks_so_far = 0 
    for name, shape in collapsed_name_and_size: 
        _, n_tokens, _ = shape 
        cumulative_shapes[name] = (n_toks_so_far, n_toks_so_far + n_tokens)
    
    # Extract the tokens for each key
    flat_token_collection = flatten_dict(token_collection)
    extracted_tokens = {key: {} for key in extract_keys}
    for token_key, token in flat_token_collection.items(): 
        if isinstance(token, tuple): 
            assert len(token) == 1 
            token = token[0]
        if 'attention_mask' in token_key: 
            continue 
        for name in extract_keys: 
            tok_start, tok_end = cumulative_shapes[name]
            extracted_grad = token[:, tok_start:tok_end, :]
            if name.startswith('obs'): 
                extracted_grad = einops.rearrange(
                    extracted_grad,
                    "batch (horizon n_tokens) d -> batch horizon n_tokens d",
                    horizon=window,
                )
            extracted_tokens[name][token_key] = extracted_grad
    
    return extracted_tokens


def apply_func(
    params: Data, 
    perturbations: Data,
    model: OctoModel, 
    batch: Data, 
    dropout_rng: PRNGKey, 
    rng: Optional[PRNGKey] = None,
    psuedo_loss_kwargs: Dict = None, 
): 
    """A function to compute a psuedo loss (which may not represent a true loss, just something to take the gradient wrt to activations of) for gradcam
    
    Args: 
        params: current model params 
        perturbations: perturbations for activations of interest, see transformer.py 
        batch: batch to apply the psuedo_loss
        dropout_rng: see OctoModel
        rng: see OctoModel 
        psuedo_loss_kwargs: kwargs related to the psuedo loss. Must represent either a true MSE loss, or simply action predictions 

    Returns: 
        apply_func_out: the psuedo-loss (MSE or single-dimension of prediction)
        transformer_outputs: the transformer embeddings. Useful to prevent repeat computation. 
        intermediates: the intermediate activations of the transformer (see 'sow' calls in transformer.py)
    
    """
    observations = batch['observation']
    tasks = batch['task']
    timestep_pad_mask = observations["timestep_pad_mask"]
    perturbations = freeze(perturbations)
    transformer_outputs, intermediates = model.run_transformer_with_intermediates(
        observations, tasks, timestep_pad_mask, train=False,
        params=params,
        perturbations=perturbations
    )
    bound_module = model.module.bind({"params": params, "perturbations": perturbations}, rngs={"dropout": dropout_rng})
    if psuedo_loss_kwargs is None: 
        psuedo_loss_kwargs = {'psuedo_loss_type': 'loss'}
    psuedo_loss_type = psuedo_loss_kwargs['psuedo_loss_type']
    if psuedo_loss_type == 'loss': 
        action_loss, action_metrics = bound_module.heads["action"].loss(
            transformer_outputs,  
            batch["action"],
            batch["observation"]["timestep_pad_mask"],
            batch["action_pad_mask"],
            train=True,
        )
        apply_func_out = -1 * action_loss  # want to record activations that have positive effect on DECREASING loss
    elif psuedo_loss_type == 'prediction': 
        unnormalization_statistics = model.dataset_statistics['action']
        action_head: ActionHead = bound_module.heads[
            "action"
        ]
        action = action_head.predict_action(
            transformer_outputs,
            train=True,
            rng=rng,
            embodiment_action_dim=len(unnormalization_statistics["mean"])
        )
        mask = unnormalization_statistics.get(
            "mask", jnp.ones_like(unnormalization_statistics["mean"], dtype=bool)
        )
        action = action[..., : len(mask)]
        action = jnp.where(
            mask,
            (action * unnormalization_statistics["std"])
            + unnormalization_statistics["mean"],
            action,
        )
        # action.shape = (batch, pred_horizon, action_dim)
        horizon_step = psuedo_loss_kwargs['pred_horizon_step']
        action_dim = psuedo_loss_kwargs['action_dim']
        selected_action = action[0, horizon_step, action_dim]
        apply_func_out = selected_action
    else: 
        raise ValueError

    return apply_func_out, (transformer_outputs, intermediates)

def remove_batch_and_window(
   batched_imgs: Sequence[JaxArray], 
   which_in_batch: Optional[int] = 0, 
   which_in_window: Optional[int] = -1, 
) -> JaxArray: 
    """Removes batch and window dimensions from each image in list"""
    debatched_imgs = []
    for batched_img in batched_imgs: 
        debatched_imgs.append(batched_img[which_in_batch, which_in_window])
    return debatched_imgs

def unflatten_imgs(
    flat_imgs: Sequence[JaxArray], 
): 
    """Unflattens each image in a list"""
    unflattened_imgs = [] 
    for img in flat_imgs: 

        pre, (n_tok, emb_dim) = img.shape[:-2], img.shape[-2:]
        img_size = int(math.sqrt(1.0 * n_tok))
        assert img_size ** 2 == n_tok, 'Original image was not square! '
        image_tokens = jnp.reshape(img, (*pre, img_size, img_size, emb_dim))
        unflattened_imgs.append(image_tokens)
    return unflattened_imgs

def normalize_image(img: JaxArray) -> JaxArray: 
    """Normalizes image to [0, 1]"""
    img = img - jnp.min(img)
    img_max = jnp.max(img)
    if img_max > 0: 
        img = img / img_max 
    else: 
        logging.warn('Image gradient min == image gradient max. Image gradient is likely 0.')
    return img 

def gradCAM(
    model: OctoModel, 
    params: Data, 
    perturbations: Data, 
    obs_key: str, 
    batch: Data, 
    dropout_rng: PRNGKey, 
    flat_layer_id: Tuple[str] = None, 
    window_step: int = -1,
    psuedo_loss_type: str = 'loss', 
    pred_horizon_step: int = 0, 
    action_dim: int = 0, 
    rng: Optional[PRNGKey] = None, 
): 
    """Computes the Grad-CAM weighted activation map corresponding to the tokens defined by obs_key and the layer defined by flat_layer_id 

    Args: 
        model: OctoModel with correct structure 
        params: model params needed to run the transformer 
        perturbations: model perturbations needed to record the intermediate gradients 
        obs_key: tokenizer group name of tokens to extract and form Grad-CAM 
        batch: batch containing observations to form Grad-CAM. We only need one step, 
            but compute all JITs batched before extracting the appropriate data since the parallelism is set up to support this 
        dropout_rng: see OctoModel 
        flat_layer_id: the flattened key corresponding to transformer output we would like to inspect. 
            Probably ('octo_transformer', 'BlockTransformer_0', 'Transformer_0', 'layer_10_out')
        window_step: defines which image in the window we should create the Grad-CAM for
        psuedo_loss_type: see apply_func 
        pred_horizon_step: see apply_func 
        action_dim: see apply_func 
        rng: see OctoModel
    
    Returns: 
        original_image: the image that the Grad-CAM was taken wrt 
        resized_gradcam: the Grad-CAM weighted activation map, resized to the same resolution as original image 

    
    """
    if flat_layer_id is None: 
        flat_layer_id = ('octo_transformer', 'BlockTransformer_0', 'Transformer_0', 'layer_10_out')
    psuedo_loss_kwargs = { 
        'psuedo_loss_type': psuedo_loss_type, 
        'pred_horizon_step': pred_horizon_step, 
        'action_dim': action_dim, 
    }

    (_, info), loss_grad =  jax.value_and_grad(apply_func, argnums=1, has_aux=True)(
        params, perturbations, model, batch, dropout_rng, rng, psuedo_loss_kwargs
    )
    transformer_outputs, intermediates = info

    extracted_intermediate_activations = extract_tokens_from_all(
        [obs_key],
        model,
        intermediates['intermediates'],
        batch,
        transformer_outputs=transformer_outputs
    )[obs_key]
    extracted_token_collection = extract_tokens_from_all(
        [obs_key], 
        model, 
        loss_grad, 
        batch,
        transformer_outputs=transformer_outputs
    )[obs_key]

    activation_img = extracted_intermediate_activations[flat_layer_id]
    gradient_img = extracted_token_collection[flat_layer_id] 
    activation_img, gradient_img = remove_batch_and_window( 
        batched_imgs=(activation_img, gradient_img), 
        which_in_window=window_step
    )
    activation_img, gradient_img = unflatten_imgs(
        flat_imgs=(activation_img, gradient_img)
    )
    assert activation_img.shape == gradient_img.shape

    GAP_gradient = jnp.mean(gradient_img, axis=(0, 1))
    gradcam = jnp.sum(activation_img * GAP_gradient, axis=-1)
    gradcam = jnp.clip(gradcam, a_min=0)

    original_image = batch['observation'][obs_key.replace('obs', 'image')][0, window_step]
    resize_size = original_image.shape[:2]
    resized_gradcam = tf.image.resize(gradcam[..., None], resize_size, method="lanczos3", antialias=True)[..., 0]
    return original_image, resized_gradcam

def get_overlaid_attention_map(img, attn_map, blur=True):
    """Computes Grad-CAM heatmaps, then overlays that on the original image"""
    if img.dtype == jnp.uint8: 
        img = img * 1.0 / 255.0 
    if blur:
        attn_map = gaussian_filter(attn_map, 0.02*max(img.shape[:2]))
        attn_map = normalize_image(attn_map)
    cmap = plt.get_cmap('jet')
    attn_map_cmap = cmap(attn_map)[..., :3]
    overlaid_attn_map = 1*(1-attn_map**0.7).reshape(attn_map.shape + (1,))*img + \
            (attn_map**0.7).reshape(attn_map.shape+(1,)) * attn_map_cmap
    return overlaid_attn_map

def visualize_img_and_gradcam(img, attn_map, blur=True):
    """Plots original image and overlaid map"""
    _, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img)
    axes[1].imshow(get_overlaid_attention_map(img, attn_map, blur))
    for ax in axes:
        ax.axis("off")
    plt.show()
