import jax.numpy as jnp
import jax

modality_combination_order = ['simple', '', 'visual', 'tactile', 'audio', 'visual,tactile', 'visual,audio', 'tactile,audio', 'visual,tactile,audio']
_modality_combinations = [('visual',), ('tactile',), ('audio',), ('visual', 'tactile'), ('visual', 'audio'), ('tactile', 'audio'), ('visual', 'tactile', 'audio')]
modality_combinations = [','.join(combination) for combination in _modality_combinations]
fuse_loss_modal_indices = {i: combination for i, combination in enumerate(modality_combination_order) if combination != ''}
contrastive_indices = fuse_loss_modal_indices
generative_indices = {k: v for k, v in contrastive_indices.items() if v != 'simple'}
name_to_index_generative = {v: k for k, v in generative_indices.items()}

modality_to_observation_keys = {
    'visual': ['image_primary', 'image_wrist'], 
    'tactile': ['image_digit_right', 'image_digit_right_background', 'image_digit_left', 'image_digit_left_background'],
    'audio': ['mic', 'mel_spectro']
}
modality_specific_keys = []
for v in modality_to_observation_keys.values(): 
    modality_specific_keys.extend(v)
modality_specific_keys = set(modality_specific_keys)
nonspecific_keys= ['task_completed', 'timestep', 'modality_idx']

modality_to_observation_keys['simple'] = list(modality_specific_keys)


def create_fuse_modal_masks(example_obs):
    modal_masks = {}
    pad_mask_dict = example_obs['pad_mask_dict']
    for i, combination in fuse_loss_modal_indices.items():
        combination_mask = {}
        for modality in combination.split(','):
            for obs_key in modality_to_observation_keys[modality]:
                if obs_key in pad_mask_dict:
                    combination_mask[obs_key] = jnp.ones_like(pad_mask_dict[obs_key])
        for obs_key in nonspecific_keys:
            if obs_key in pad_mask_dict:
                combination_mask[obs_key] = jnp.ones_like(pad_mask_dict[obs_key])
        for obs_key in pad_mask_dict:
            if obs_key not in combination_mask:
                combination_mask[obs_key] = jnp.zeros_like(pad_mask_dict[obs_key])
        modal_masks[i] = combination_mask
        assert modal_masks[i].keys() == pad_mask_dict.keys()
    return modal_masks


def create_batch(batch, observation_masks, fuse_modal_masks, modality_combination_index: int):
    if observation_masks is None:
        batch['observation']['pad_mask_dict'] = fuse_modal_masks[modality_combination_index]
    else:
        batch['observation']['pad_mask_dict'] = jax.tree_map(
            lambda true_mask, fuse_mask: jnp.logical_and(true_mask, fuse_mask),
            observation_masks,
            fuse_modal_masks[modality_combination_index],
        )
    return batch