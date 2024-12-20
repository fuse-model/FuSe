import numpy as np

gen_modes = [("visual",), ("tactile",), ("visual", "tactile")]
# gen_modes = [('audio',)]
csv_modes = [",".join(modality_tuple) for modality_tuple in gen_modes]
gen_mode_lang_names = [
    "all_lang_4",
]
modality_obs_keys = {
    "visual": ["image_primary", "image_wrist"],
    "tactile": ["image_digit_right", "image_digit_left"],
    "audio": [
        "mel_spectro",
    ],
}

modality_specific_keys = []
for v in modality_obs_keys.values():
    modality_specific_keys.extend(v)
modality_specific_keys = set(modality_specific_keys)

includes = ["pad_mask_dict", "task_completed", "timestep", "timestep_pad_mask"]

WINDOW_SIZE = 2
pad_mask = np.array([[True for _ in range(WINDOW_SIZE)]])[0]
