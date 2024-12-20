from eval.recursive_dict_print import recursive_dict_print
import jax
import numpy as np
from octo.data.utils.text_processing import HFTokenizer
from octo.model.octo_model import OctoModel
from octo.model.octo_module import OctoModule

gen_modes = [("visual",), ("tactile",), ("visual", "tactile")]
csv_modes = [",".join(modality_tuple) for modality_tuple in gen_modes]
gen_mode_lang_names = ["all_lang_2", "all_lang_3", "all_lang_5"]
modality_obs_keys = {
    "visual": ["image_primary", "image_wrist"],
    "tactile": ["image_digit_right", "image_digit_left"],
}

modality_specific_keys = []
for v in modality_obs_keys.values():
    modality_specific_keys.extend(v)
modality_specific_keys = set(modality_specific_keys)

includes = ["pad_mask_dict", "task_completed", "timestep", "timestep_pad_mask"]

WINDOW_SIZE = 2
pad_mask = np.array([[True for _ in range(WINDOW_SIZE)]])[0]


def create_batch(obs, gen_mode):
    modality_obs = {}
    for modality_key in gen_mode:
        for obs_key in modality_obs_keys[modality_key]:
            modality_obs[obs_key] = obs[obs_key]
    modality_obs["timestep_pad_mask"] = pad_mask
    return modality_obs


def get_language_decoded(model: OctoModel, obs, rng, train=False):
    text_processor = model.text_processor
    # batch_size = len(obs['image_primary'])
    batch_size = 1
    # task = {
    #     'language_instruction': text_processor.encode(['' for s in range(batch_size)])
    # }
    task = model.create_tasks(texts=["" for s in range(batch_size)])
    info = {}
    bound_module = model.module.bind({"params": model.params}, rngs={"dropout": rng})
    print("\n\n\n", gen_modes, "\n\n\n")
    module: OctoModule = model.module
    if "gen" in module.heads:
        for gen_mode, csv_mode, gen_mode_lang_name in zip(
            gen_modes, csv_modes, gen_mode_lang_names
        ):
            modality_obs = create_batch(obs, gen_mode)
            modality_obs = jax.tree_map(lambda x: x[None], modality_obs)
            recursive_dict_print(modality_obs)
            recursive_dict_print(task)
            modality_transformer_embedding = bound_module.octo_transformer(
                modality_obs,
                task,
                modality_obs["timestep_pad_mask"],
                train=train,
            )
            # target_lang = batch['task'][gen_mode_lang_name]['input_ids']
            decode_ids = bound_module.heads[f"gen"].reconstruct_lang(
                modality_transformer_embedding,
                mode=csv_mode,
                train=train,
            )
            info[f"gen_{csv_mode}"] = text_processor.decode(decode_ids)
    else:
        for gen_mode, csv_mode, gen_mode_lang_name in zip(
            gen_modes, csv_modes, gen_mode_lang_names
        ):
            modality_obs = create_batch(obs, gen_mode)
            modality_obs = jax.tree_map(lambda x: x[None], modality_obs)

            # recursive_dict_print(modality_obs)
            # recursive_dict_print(task)
            modality_transformer_embedding = bound_module.octo_transformer(
                modality_obs,
                task,
                modality_obs["timestep_pad_mask"],
                train=train,
            )
            # target_lang = batch['task'][gen_mode_lang_name]['input_ids']
            decode_ids = bound_module.heads[f"gen_{csv_mode}"].reconstruct_lang(
                modality_transformer_embedding,
                train=train,
            )
            info[f"gen_{csv_mode}"] = text_processor.decode(decode_ids)
    return info
