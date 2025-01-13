from functools import partial
import json
import pickle

import jax
import jax.numpy as jnp
import tensorflow as tf
from octo.model.octo_module import OctoModule
from octo.model.octo_model import OctoModel
from octo.utils.typing import Data
from octo.utils.logging_utils import append_identity_to_metrics
from octo.utils.fuse_constants import (
    contrastive_indices,
    generative_indices,
    create_batch
)


def fuse_loss_fn_contrastive(bound_module: OctoModule, batch: Data, train: bool = True): 
    total_loss = 0.0
    info = {}
    batch_language_instruction = batch['task'].pop('language_instruction')

    transformer_embeddings_without_language = bound_module.octo_transformer(
        batch["observation"],
        batch["task"],
        batch["observation"]["timestep_pad_mask"],
        train=train,
    )

    modal_commands = batch['task']['modal_commands']
    for i, modality_combination in contrastive_indices.items(): 
        batch['task']['language_instruction'] = jax.tree_map(lambda x: x[:, i], modal_commands)
        true_language_embeddings = bound_module.octo_transformer.embed_language(batch['task'], train=train)
        lang_loss, lang_metrics = bound_module.heads["clip"].loss(
            transformer_embeddings_without_language,
            true_language_embeddings,
            batch["observation"]["timestep_pad_mask"],
            train=train,
        )
        lang_metrics = append_identity_to_metrics(lang_metrics, identity_suffix=f'contrastive_{modality_combination}')      
        total_loss += lang_loss
        info.update(lang_metrics)
    
    batch['task']['language_instruction'] = batch_language_instruction 
    total_loss = total_loss / len(contrastive_indices)
    return total_loss, info


def _fuse_generative(model: OctoModel, params: Data, rng: Data, batch: Data, train: bool = True, mask_invalid_language: bool = True): 
    total_loss = 0.0
    batch_language_instruction = batch['task'].pop('language_instruction')
    batch['task']['language_instruction'] = batch['task']['null']
    observation_masks = batch['observation']['pad_mask_dict']

    @partial(
        jax.jit,
        static_argnames=['train'],
    )
    def modal_step(batch: Data, target_ids: Data, mask_loss: Data, modality_idx: int, train: bool = True): 
        bound_module = model.module.bind({"params": params}, rngs={"dropout": rng})
        modality_transformer_embedding = bound_module.octo_transformer(
            batch['observation'],
            batch["task"],
            batch["observation"]["timestep_pad_mask"],
            train=train,
        )
        if train:
            gen_loss, gen_metrics = bound_module.heads[f"gen"].loss(
                modality_transformer_embedding,
                target_ids,
                modality_idx, 
                batch["observation"]["timestep_pad_mask"],
                mask=mask_loss,
                train=train,
            )
        else:
            decode_ids = bound_module.heads[f"gen"].reconstruct_lang(
                modality_transformer_embedding,
                modality_idx=modality_idx,
                train=train,
            )
            gen_metrics = {'gen': [decode_ids, target_ids]}
            gen_loss = 0
        return gen_loss, gen_metrics 
    
    def wrapped_modal_step(batch: Data, observation_masks: Data, fuse_modal_masks: Data, modality_idx: int, train: bool = True):
        masked_batch = create_batch(batch, observation_masks, fuse_modal_masks, modality_idx)
        target_ids = masked_batch['task']['modal_commands']['input_ids'][:, modality_idx]
        mask_loss = jax.tree_map(lambda x: x[:, modality_idx][..., None], masked_batch['task']['language_validity'])
        if not mask_invalid_language:
            mask_loss = jnp.ones_like(mask_loss)
        return modal_step(batch=masked_batch, target_ids=target_ids, mask_loss=mask_loss, modality_idx=jnp.array(modality_idx), train=train)
        
    all_info = {}
    total_loss = 0.0
    for modality_idx, combination in generative_indices.items(): 
        modality_loss, modality_info = wrapped_modal_step(batch=batch, observation_masks=observation_masks, fuse_modal_masks=model.fuse_modal_masks, modality_idx=modality_idx, train=train)
        all_info.update(append_identity_to_metrics(modality_info, identity_suffix=f'gen_{combination}'))
        total_loss += modality_loss
    total_loss /= len(generative_indices)
    batch['task']['language_instruction'] = batch_language_instruction 
    batch['observation']['pad_mask_dict'] = observation_masks
    return total_loss, all_info

fuse_loss_fn_generative = partial(_fuse_generative, train=True)
fuse_decode_ids = partial(_fuse_generative, train=False)


class FuseRephraser:
    def create_static_hash_table(self, dictionary, key_dtype, value_dtype, default_value):
        """Takes a python dictionary with string keys and values and creates a tf static hash table"""
        keys = list(dictionary.keys())
        values = list(dictionary.values())
        initializer = tf.lookup.KeyValueTensorInitializer(
            keys, values, key_dtype=key_dtype, value_dtype=value_dtype
        )
        hash_table = tf.lookup.StaticHashTable(initializer, default_value=default_value)
        return hash_table

    @classmethod
    def create(cls, file_path: str, initial_keys: tuple = (), key_dtype=tf.string, value_dtype=tf.string, default_value=""):
        def pickle_load(pickle_file_path: str):
            with tf.io.gfile.GFile(pickle_file_path, "rb") as file:
                return pickle.load(file)
        def json_load(json_file_path: str):
            with tf.io.gfile.GFile(json_file_path, "r") as file:
                return json.load(file)
        def load(file_path: str):
            if file_path.endswith(".json"):
                return json_load(file_path)
            elif file_path.endswith(".pkl"):
                return pickle_load(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_path}")
        dictionary = load(file_path)
        for key in initial_keys:
            dictionary = dictionary[key]
        return cls(dictionary, key_dtype, value_dtype, default_value)

    def __init__(self, dictionary: dict, key_dtype, value_dtype, default_value):
        self.dictionary = dictionary
        self.hash_table = self.create_static_hash_table(dictionary, key_dtype, value_dtype, default_value)