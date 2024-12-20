from dataclasses import dataclass, field
from itertools import combinations
from typing import Union

from octo.data.utils.text_processing import HFTokenizer
from octo.utils.spec import ModuleSpec

# from ckpt_utils import get_all_runs

VERBOSE = False
PRINT_OBS_INFO = False
MIXED_COMMANDS = True
WRIST_SIZE = (128, 128)
# WRIST_SIZE = (256, 256)

# DIG_SIZE = (256, 256)
# DIG_SIZE = (256, 256)
# DIG_SIZE = (128, 128)
DIG_SIZE = (224, 224)
WIDOWX_SERVICE_SIZE_MAP = {
    "image_0": (256, 256),
    "image_1": WRIST_SIZE,  #  (128, 128),
    "digit": DIG_SIZE,  # (128, 128),
    "background": DIG_SIZE,
    "mel_spectro": (128, 128),
}

# RESIZE_DIG_SIZE = (224, 224)
# RESIZE_DIG_SIZE = (256, 256)
# RESIZE_DIG_SIZE = (128, 128)
RESIZE_DIG_SIZE = DIG_SIZE
RESIZE_MAP = {
    "image_0": (256, 256),
    "image_1": WRIST_SIZE,
    "digit_l": RESIZE_DIG_SIZE,  # (128, 128),
    "digit_r": RESIZE_DIG_SIZE,  # (128, 128),
    "background_l": RESIZE_DIG_SIZE,  # (128, 128),
    "background_r": RESIZE_DIG_SIZE,
}

OBS_KEY_MAP = {
    "image": {
        "primary": "image_0",
        "wrist": "image_1",
        "digit_left": "digit_l",
        "digit_right": "digit_r",
        "digit_left_background": "background_l",
        "digit_right_background": "background_r",
    },
}
CALCULATED_FIELDS = [
    "spectro",
    # 'digit_embeddings'
    # 'mae_viz'
    # 'mae_tac_uniform',
]
PREFIX_COMMANDS = {"visual": "looks", "tactile": "feels", "audio": "sounds"}
REMOVE_KEYS = {"", "audio"}
MULTIMODAL_COMBOS = ["visual", "tactile", "visual,tactile,audio"]


@dataclass
class MultimodalCommand:
    annotation_dict: dict[str, str]
    simple_command: str = None
    prefix: str = "Grab the object that"
    def_key_list: list[str] = field(default_factory=lambda: MULTIMODAL_COMBOS.copy())

    def __post_init__(self):
        sorted_keys = list(self.annotation_dict.keys())
        all_combos = tuple()
        for i in range(len(sorted_keys) + 1):
            all_combos += tuple(combinations(sorted_keys, i))
        self.string_combos = [",".join(tup) for tup in all_combos]

        def construct_annotation(string_key):
            annotation = [self.prefix]
            is_first = True
            for modality in string_key.split(","):
                connector = "" if is_first else " and"
                annotation.append(
                    f"{connector} {PREFIX_COMMANDS[modality]} {self.annotation_dict[modality]}"
                )
                is_first = False
            annotation.append(".")
            return "".join(annotation)

        self.key_list = self.def_key_list if self.def_key_list else self.string_combos
        self.all_annotations = [
            construct_annotation(key) for key in self.key_list if key not in REMOVE_KEYS
        ]
        if self.simple_command is not None:
            self.all_annotations.append(self.simple_command)
            self.key_list.append("simple")
        self.index = 0

    @property
    def curr_key(self):
        return self.key_list[self.index]

    def curr_annotation(self):
        return self.all_annotations[self.index]

    def next_annotation(self):
        self.index = (self.index + 1) % len(self.all_annotations)
        return self.curr_annotation()

    def reset(self):
        self.index = 0

    def is_last(self):
        return self.index == len(self.all_annotations) - 1

    def peek_next(self):
        return self.all_annotations[(self.index + 1) % len(self.all_annotations)]


@dataclass
class AnnotationList:
    annotations: list[str]
    keys: list[str] = None
    index: int = 0

    def __post_init__(self):
        if self.keys is None:
            self.keys = [str(i) for i in range(len(self.annotations))]
        if len(self.annotations) != len(self.keys):
            raise ValueError(
                f"annotations and keys must have save lengths; received {self.annotations} and {self.keys}\nwith lengths {len(self.annotations)} and {len(self.keys)}"
            )

    @property
    def curr_key(self):
        return self.keys[self.index]

    def curr_annotation(self):
        return self.annotations[self.index]

    def next_annotation(self):
        self.index = (self.index + 1) % len(self.annotations)
        return self.curr_annotation()

    def reset(self):
        self.index = 0

    def is_last(self):
        return self.index == len(self.annotations) - 1

    def peek_next(self):
        return self.annotations[(self.index + 1) % len(self.annotations)]


GroupedAnnotations = Union[AnnotationList, MultimodalCommand]
CommandType = Union[GroupedAnnotations, str]

# @dataclass
# class MultimodalCommand:
#     annotation_dict: dict[str, str]
#     prefix: str = 'Grab the object that'
#     def __post_init__(self):
#         sorted_keys = list(self.annotation_dict.keys())
#         all_combos = tuple()
#         for i in range(len(sorted_keys)+1):
#             all_combos += tuple(combinations(sorted_keys, i))
#         string_combos = [','.join(tup) for tup in all_combos]
#         def construct_annotation(string_key):
#             annotation = [self.prefix]
#             is_first = True
#             for modality in string_key.split(','):
#                 connector = '' if is_first else ' and'
#                 annotation.append(f"{connector} {PREFIX_COMMANDS[modality]} {self.annotation_dict[modality]}")
#                 is_first = False
#             annotation.append('.')
#             return ''.join(annotation)
#         key_list = MULTIMODAL_COMBOS if MULTIMODAL_COMBOS else string_combos
#         self.all_annotations = [construct_annotation(key) for key in key_list if key not in REMOVE_KEYS]
#         self.index = 0
#         self.reset_state = True

#     def curr_annotation(self):
#         if self.reset_state:
#             print('WARNING: multimodal command in reset state, reading {self.index}. Should call next annot first')
#         return self.all_annotations[self.index ]
#     def next_annotation(self):

#         self.index = (self.index + 1 ) % len(self.all_annotations)
#         finished_sweep = not self.reset_state and self.index == 0
#         self.reset_state = False
#         if finished_sweep:
#             print('WARNING:   finished sweep, but model is unchanged. Did you want to change models?')
#         return self.curr_annotation()
#     def reset(self):
#         self.index = len(self.all_annotations) - 1


TRAIN_SIMPLE_CMD = "Grab the silver keys."
TRAIN_MULTI_CMD = MultimodalCommand(
    annotation_dict={
        "visual": "silver, large",
        "tactile": "metallic, sharp",
        "audio": "jangly",
    },
    simple_command=TRAIN_SIMPLE_CMD if MIXED_COMMANDS else "",
)

TEST_SIMPLE_CMD = "Grab the yellow block."
TEST_MULTI_CMD = MultimodalCommand(
    annotation_dict={
        "visual": "yellow, rectangle",
        "tactile": "wooden, matte",
        "audio": "solid",
    },
    simple_command=TEST_SIMPLE_CMD if MIXED_COMMANDS else "",
)

TAC_SIMPLE_CMD = "Grab the yellow block."
TAC_MULTI_CMD = MultimodalCommand(
    annotation_dict={
        "visual": "yellow, rectangle",
        "tactile": "wooden, matte",
        "audio": "solid",
    },
    simple_command=None,
    def_key_list=["tactile"],
)

TAC2_MULTI_CMD = MultimodalCommand(
    annotation_dict={
        "visual": "white, animal",
        "tactile": "fuzzy, furry",
        "audio": "quiet",
    },
    simple_command="Grab the white rabbit",
    def_key_list=["tactile"],
)

TAC3_MULTI_CMD = MultimodalCommand(
    annotation_dict={
        "visual": "white, animal",
        "tactile": "soft, squishy",
        "audio": "quiet",
    },
    simple_command=None,  #'Grab the white rabbit',
    def_key_list=["tactile"],
)

yellow_ball_multi = MultimodalCommand(
    annotation_dict={"visual": "yellow", "tactile": "foam, pocked", "audio": "quiet"},
    simple_command=None,  #'Grab the white rabbit',
    def_key_list=["visual,tactile"],
)
yball_viztac = yellow_ball_multi.curr_annotation()

yellow_block_multi = MultimodalCommand(
    annotation_dict={"visual": "yellow", "tactile": "wooden, matte", "audio": "solid"},
    simple_command=None,
    def_key_list=["visual,tactile"],
)
yblock_viztac = yellow_block_multi.curr_annotation()

DISAMBIG_CMD = AnnotationList(
    annotations=[yball_viztac, yblock_viztac], keys=["ball", "block"]
)

blue_ball_multi = MultimodalCommand(
    annotation_dict={"visual": "blue", "tactile": "foam, pocked", "audio": "quiet"},
    simple_command=None,  #'Grab the white rabbit',
    def_key_list=["visual,tactile"],
)
bball_viztac = blue_ball_multi.curr_annotation()

blue_block_multi = MultimodalCommand(
    annotation_dict={"visual": "yellow", "tactile": "wooden, matte", "audio": "solid"},
    simple_command=None,
    def_key_list=["visual,tactile"],
)
bblock_viztac = blue_block_multi.curr_annotation()

blue_duck_viztac = MultimodalCommand(
    annotation_dict={
        "visual": "blue",
        "tactile": "rubbery, irregular",
        "audio": "squeaky",
    },
    simple_command=None,
    def_key_list=["visual,tactile"],
).curr_annotation()
blue_duck_simple = "Grab the blue duck."

green_duck_viztac = MultimodalCommand(
    annotation_dict={
        "visual": "green",
        "tactile": "rubbery, irregular",
        "audio": "squeaky",
    },
    simple_command=None,
    def_key_list=["visual,tactile"],
).curr_annotation()
green_duck_simple = "Grab the green duck."

blue_animal_viztac = MultimodalCommand(
    annotation_dict={"visual": "blue", "tactile": "soft, squishy", "audio": "quiet"},
    simple_command=None,
    def_key_list=["visual,tactile"],
).curr_annotation()
blue_animal_simple = "Grab the blue animal."

green_animal_viztac = MultimodalCommand(
    annotation_dict={"visual": "green", "tactile": "soft, squishy", "audio": "quiet"},
    simple_command=None,
    def_key_list=["visual,tactile"],
).curr_annotation()
green_animal_simple = "Grab the green animal."


DISAMBIG2_CMD = AnnotationList(
    annotations=[
        blue_duck_viztac,
        blue_duck_simple,
        green_duck_viztac,
        green_duck_simple,
        blue_animal_viztac,
        blue_animal_simple,
        green_animal_viztac,
        green_animal_simple,
    ],
    keys=[
        "blue_duck_viztac",
        "blue_duck_simple",
        "green_duck_viztac",
        "green_duck_simple",
        "blue_animal_viztac",
        "blue_animal_simple",
        "green_animal_viztac",
        "green_animal_simple",
    ],
)

blue_duck_vizsound = MultimodalCommand(
    annotation_dict={"visual": "blue", "tactile": "N/A", "audio": "squeaky"},
    simple_command=None,
    def_key_list=["visual,audio"],
).curr_annotation()
green_duck_vizsound = MultimodalCommand(
    annotation_dict={"visual": "green", "tactile": "N/A", "audio": "squeaky"},
    simple_command=None,
    def_key_list=["visual,audio"],
).curr_annotation()

DISAMBIG3_CMD = AnnotationList(
    annotations=[
        blue_duck_simple,
        blue_duck_vizsound,
        green_duck_simple,
        green_duck_vizsound,
    ],
    keys=[
        "blue_duck_simple",
        "blue_duck_vizsound",
        "green_duck_simple",
        "green_duck_vizsound",
    ],
)


SOUND1_CMD = AnnotationList(
    annotations=[
        "Grab the object that sounds jangly.",
        "Grab the bronze keys.",
        "Grab the object that sounds squeaky.",
        "Grab the blue duck.",
    ],
    keys=["key_sound", "key_simple", "duck_sound", "duck_simple"],
)


yellow_ball_simple = "Grab the yellow ball."
yellow_ball_tac = "Grab the object that feels squishy."
red_block_simple = "Grab the red block."
red_block_tac = "Grab the object that feels firm."

TAC_TEST = AnnotationList(
    annotations=[yellow_ball_simple, yellow_ball_tac, red_block_simple, red_block_tac],
    keys=["yellow_ball_simple", "yellow_ball_tac", "red_block_simple", "red_block_tac"],
)


runs = {
    # "cont_hf_cams_digits_41_20240727_135125": {
    #     'Window size': 2,
    #     'Language type': 'multimodal'
    # },
    # "cams_digits_41_20240727_134318": {
    #     'Window size': 2,
    #     'Language type': 'multimodal'
    # },
    # "cont_cams_digits_41_20240727_133652": {
    #     'Window size': 2,
    #     'Language type': 'multimodal'
    # },
    # "cosine_cams_digits_41_20240727_133650": {
    #     'Window size': 2,
    #     'Language type': 'multimodal'
    # },
    # "simple_cont_hf_cams_digits_41_20240728_013448": {
    #     'Window size': 2,
    #     'Language type': 'simple'
    # },
    # "simple_cont_cams_digits_41_20240728_013446": {
    #     'Window size': 2,
    #     'Language type': 'simple'
    # },
    # "simple_cosine_cams_digits_41_20240728_013445": {
    #     'Window size': 2,
    #     'Language type': 'simple'
    # },
    # "simple_cams_digits_41_20240728_013443": {
    #     'Window size': 2,
    #     'Language type': 'simple'
    # },
    # 'simple_20k_20240805_180735': {
    #     'Window size': 2,
    #     'Language type': 'simple'
    # },
    # 'simple_cont_20k_20240805_161651': {
    #     'Window size': 2,
    #     'Language type': 'simple'
    # },
    # "multi_cont_20k_20240806_072717": {
    #     'Window size': 2,
    #     'Language type': 'multimodal'
    # },
    # "multi_base_20k_20240806_224758": {
    #     'Window size': 2,
    #     'Language type': 'multimodal'
    # },
    # "scratch_cont_full_20240806_065105": {
    #     'Window size': 2,
    #     'Language type': 'simple'
    # },
    # "scratch_cont_full_20240806_065105": {
    #     'Window size': 2,
    #     'Language type': 'simple'
    # },
    # "scratch_cont_frozen_20240806_195316": {
    #     'Window size': 2,
    #     'Language type': 'simple'
    # },
    # "bcz_20k_20240805_223201": {
    #     'Window size': 2,
    #     'Language type': 'simple'
    # },
    # "bcz_20k_20240805_223201": {
    #     'Window size': 2,
    #     'Language type': 'simple'
    # },
    # "bcz_20k_20240805_223159": {
    #     'Window size': 2,
    #     'Language type': 'simple'
    # },
    # "multi_cont_20k_20240806_072717": {
    #     'Window size': 2,
    #     'Language type': 't2',
    # },
    # "multi_base_20k_20240806_224758": {
    #     'Window size': 2,
    #     'Language type': 'color_test', # 'dis2'
    # },
    # "audio_cont_20k_20240807_220439": {
    #     'Window size': 2,
    #     'Language type': 'dis3',
    # },
    # 'mic_audio_cont_20k_d128_20240808_234005': {
    #     'Window size': 2,
    #     'Language type': 'dis3',
    # }
    "tac_base_20240824_173914": {
        "Window size": 2,
        "Language type": "test_tac",
    },
    "vit_20240824_173949": {
        "Window size": 2,
        "Language type": "test_tac",
    },
    "resnet_imagenet_20240824_174202": {
        "Window size": 2,
        "Language type": "test_tac",
    },
    "tvl_finetuned_20240824_070132": {
        "Window size": 2,
        "Language type": "test_tac",
    },
    "tvl_frozen_20240824_123443": {
        "Window size": 2,
        "Language type": "test_tac",
    },
    "t3_frozen_20240824_205857": {
        "Window size": 2,
        "Language type": "test_tac",
    },
    "pod_b128": {
        "Window size": 2,
        "Language type": "pod",
    },
    "pod_b256": {
        "Window size": 2,
        "Language type": "pod",
    },
    "pod_b512": {
        "Window size": 2,
        "Language type": "pod",
    },
    "pod_b1024": {
        "Window size": 2,
        "Language type": "pod",
    },
    "pod_b4096": {
        "Window size": 2,
        "Language type": "pod",
    },
    "lang_base_20240824_070202": {
        "Window size": 2,
        "Language type": "rephrase",
    },
    "rephrase_20240824_101742": {
        "Window size": 2,
        "Language type": "rephrase",
    },
    "rephrase_t5_20240824_142449": {
        "Window size": 2,
        "Language type": "rephrase",
    },
    "rephrase_clip_20240824_215439": {
        "Window size": 2,
        "Language type": "rephrase",
    },
    "rephrase_clip_t5_20240824_174543": {
        "Window size": 2,
        "Language type": "rephrase",
    },
    "multi_cont_20k_20240806_072717": {
        "Window size": 2,
        "Language type": "pod",
    },
    "good_pod_512": {
        "Window size": 2,
        "Language type": "pod",
    },
    "good_pod_1024": {
        "Window size": 2,
        "Language type": "gen",
    },
    "good_pod_2048": {
        "Window size": 2,
        "Language type": "pod",
    },
    "experiment_20240827_231310": {
        "Window size": 2,
        "Language type": "pod",
    },
    "josh_pod_gen_lang_b1024_20240830_012615": {
        "Window size": 2,
        "Language type": "gen",
    },
    "josh_pod_gen_lang_single_headb1024_20240830_060804": {
        "Window size": 2,
        "Language type": "gen",
    },
    "window3": {
        "Window size": 3,
        "Language type": "gen",
    },
    "combined_loss": {
        "Window size": 2,
        "Language type": "gen",
    },
    "rephrase_full_finetune": {
        "Window size": 2,
        "Language type": "gen",
    },
    "rephrase_frozen": {
        "Window size": 2,
        "Language type": "gen",
    },
    "rephrase_finetune_small": {
        "Window size": 2,
        "Language type": "gen",
    },
    "mae_asym_tac": {
        "Window size": 2,
        "Language type": "gen",
    },
    "mae_uniform_tac": {
        "Window size": 2,
        "Language type": "gen",
    },
    "mae_viz_asym": {
        "Window size": 2,
        "Language type": "gen",
    },
    "mae_viz_uniform": {
        "Window size": 2,
        "Language type": "gen",
    },
    "full_mae_tac_uniform_single_head": {
        "Window size": 2,
        "Language type": "final",
    },
    "full_contrastive": {
        "Window size": 2,
        "Language type": "final",
    },
    "full_base_scratch": {
        "Window size": 2,
        "Language type": "final",
    },
    "full_base": {
        "Window size": 2,
        "Language type": "final",
    },
    "full_generative": {
        "Window size": 2,
        "Language type": "final",
    },
    "full_combined": {
        "Window size": 2,
        "Language type": "final",
    },
    "new_tac_uniform": {
        "Window size": 2,
        "Language type": "final",
    },
    "frozen_combined": {
        "Window size": 2,
        "Language type": "final",
    },
    "combined_channels": {
        "Window size": 2,
        "Language type": "final",
    },
    "t3_finetune_20240826_000930": {
        "Window size": 2,
        "Language type": "final",
    },
    "josh_pod_final_combined_model_rephrase_cond_b1024_20240908_045441": {
        "Window size": 2,
        "Language type": "final",
    },
    "josh_pod_final_combined_model_rephrase_full_b1024_20240908_220627": {
        "Window size": 2,
        "Language type": "final",
    },
    "josh_pod_final_combined_model_tvl_single_b1024_20240909_184554": {
        "Window size": 2,
        "Language type": "final",
    },
    "josh_pod_final_combined_model_tvl_single_rephrase_full_b1024_20240910_020337": {
        "Window size": 2,
        "Language type": "final",
    },
    "josh_pod_final_combined_model_tvl_single_rephrase_full_combinations_b1024_20240910_220927": {
        "Window size": 2,
        "Language type": "final",
    },
    "josh_pod_final_finetune_no_sensor_b1024_20240911_034103": {
        "Window size": 2,
        "Language type": "final",
    },
    "josh_pod_final_scratch_no_sensor_b1024_20240911_050827": {
        "Window size": 2,
        "Language type": "final",
    },
    "josh_pod_final_bcz_b1024_20240911_185628": {
        "Window size": 2,
        "Language type": "final",
    },
    "josh_pod_ours_no_sound_b1024_20240912_214503": {
        "Window size": 2,
        "Language type": "final",
    },
}
new = [
    "josh_pod_ours_no_generative_b1024_20240912_173217",
    "josh_pod_ours_no_loss_b1024_20240911_230212",
    "josh_pod_ours_no_contrastive_b1024_20240912_034704",
    "josh_pod_ours_simple_labels_b1024_20240913_023300",
]
for n in new:
    runs[n] = {
        "Window size": 2,
        "Language type": "final",
    }


NAME_TO_WINDOW_SIZE = {name: int(v["Window size"]) for name, v in runs.items()}


final_bag_objects = [
    "yellow block",
    "green cloth",
    "brown animal",
    "green sponge",
    "blue duck",
    "blue block",
    "yellow sponge",
    "green animal",
    "multicolored rope",
    "silver keys",
]
final_bag_annotations = [f"Grab the {obj}." for obj in final_bag_objects]
train_or_test = ["test" if i < 5 else "train" for i in range(10)]
final_bag_labels = [
    f'{i}_{train_v_test}_{"_".join(desc.split(" "))}'
    for i, (train_v_test, desc) in enumerate(zip(train_or_test, final_bag_objects))
]
FINAL_BAG_CMDS = [
    AnnotationList(
        annotations=final_bag_annotations,
        keys=final_bag_labels,
    )
]
FINAL_BAG_PAIRS = {
    label: AnnotationList(annotations=[annotation], keys=[label])
    for annotation, label in zip(final_bag_annotations, final_bag_labels)
}


annotation_dicts = {
    "yellow_block": {"visual": "yellow", "tactile": "wooden"},
    "green_cloth": {"visual": "green", "tactile": "fibrous"},
    "brown_animal": {"visual": "brown", "tactile": "soft"},
    "green_sponge": {"visual": "green", "tactile": "spongey"},
    "blue_duck": {"visual": "blue", "tactile": "rubbery"},
    "blue_block": {"visual": "blue", "tactile": "wooden"},
    "yellow_sponge": {"visual": "yellow", "tactile": "spongey"},
    "green_animal": {"visual": "green", "tactile": "soft"},
    "multicolored_rope": {"visual": "multicolored", "tactile": "corded"},
    "silver_keys": {"visual": "silver", "tactile": "sharp"},
}
assert len(annotation_dicts) == 10, len(annotation_dicts)


def ann_dict_to_cmds(ann_dict):
    out = {}
    prefix = "Grab the object that"
    out["visual"] = f'{prefix} looks {ann_dict["visual"]}.'
    out["tactile"] = f'{prefix} feels {ann_dict["tactile"]}.'
    out[
        "visual,tactile"
    ] = f'{prefix} looks {ann_dict["visual"]} and feels {ann_dict["tactile"]}.'
    return out


final_bag_multi = {}
for k, v in annotation_dicts.items():
    out = ann_dict_to_cmds(v)
    final_bag_multi[k] = AnnotationList(
        annotations=list(out.values()),
        keys=list(out.keys()),
    )


final_flat_objects = [
    "yellow ball",
    "red duck",
    "blue paper",
    "white rope",
    "green pepper",
    "blue bottle",
    "brown shell",
    "blue cloth",
    "red paper",
    "green ball",
]
final_flat_annotations = [f"Grab the {obj}." for obj in final_flat_objects]
final_flat_labels = [
    f"{i}_{train_v_test}_{desc}"
    for i, (train_v_test, desc) in enumerate(zip(train_or_test, final_flat_objects))
]
FINAL_FLAT_PAIRS = {
    label: AnnotationList(annotations=[annotation], keys=[label])
    for annotation, label in zip(final_flat_annotations, final_flat_labels)
}

annotation_dicts = {
    "yellow_ball": {"visual": "yellow", "tactile": "squishy"},
    "red_duck": {"visual": "red", "tactile": "rubbery"},
    "blue_paper": {"visual": "blue", "tactile": "crinkly"},
    "white_rope": {"visual": "white", "tactile": "corded"},
    "green_pepper": {"visual": "green", "tactile": "slick"},
    "blue_bottle": {"visual": "blue", "tactile": "slick"},
    "brown_shell": {"visual": "brown", "tactile": "spiky"},
    "blue_cloth": {"visual": "blue", "tactile": "fibrous"},
    "red_paper": {"visual": "red", "tactile": "crinkly"},
    "green_ball": {"visual": "green", "tactile": "squishy"},
}
final_flat_multi = {}
for k, v in annotation_dicts.items():
    out = ann_dict_to_cmds(v)
    final_flat_multi[k] = AnnotationList(
        annotations=list(out.values()),
        keys=list(out.keys()),
    )


push_prefix = "Push the"
button_midfix = "button that plays"
annotation_keys = [
    "",
    "visual",
    "tactile",
    "audio",
    "visual,tactile",
    "visual,audio",
    "tactile,audio",
    "visual,tactile,audio",
    "simple",
]

sounds = {
    "red": "whistling",
    "orange": "rap",
    "yellow": "barking",
    "pink": "piano",
    "green": "metal",
    "purple": "squeaking",
}

sound_annotations = {
    color: f"{push_prefix} {button_midfix} {sound}." for color, sound in sounds.items()
}
color_annotations = {color: f"{push_prefix} {color} button." for color in sounds}

color_sound_annotations = {
    color: f"{push_prefix} {color} {button_midfix} {sound}."
    for color, sound in sounds.items()
}

button_train_colors = ["orange", "purple", "pink"]
modalities = {
    "audio": sound_annotations,
    "visual": color_annotations,
    "visual,audio": color_sound_annotations,
}
button_train_cmds = {
    f"button_{i}": AnnotationList(
        annotations=[modality[color] for modality in modalities.values()],
        keys=list(modalities.keys()),
    )
    for i, color in enumerate(button_train_colors)
}

button_test_cmds = {
    "button_3": AnnotationList(
        annotations=["Push the blue button."],
        keys=[
            "visual",
        ],
    ),
    "button_4": AnnotationList(
        annotations=["Push the pink button."],
        keys=[
            "visual",
        ],
    ),
}


info_to_command = {
    ("train", "simple"): TRAIN_SIMPLE_CMD,
    ("train", "multimodal"): TRAIN_MULTI_CMD,
    ("test", "simple"): TEST_SIMPLE_CMD,
    ("test", "multimodal"): TEST_MULTI_CMD,
    ("tac", "tac"): TAC_MULTI_CMD,  # yellow block, no blue block
    ("tac2", "tac2"): TAC2_MULTI_CMD,  # white rabbit
    ("tac3", "tac3"): TAC3_MULTI_CMD,
    # ('tac_rabbit', 'tac_rabbit'): TACRABBIT_MULTI_CMD,
    # ('tac_duck', 'tac_duck'): TACDUCK_MULTI_CMD,
    ("dis1", "dis1"): DISAMBIG_CMD,  # yellow ball vs yellow stuffed
    ("dis2", "dis2"): DISAMBIG2_CMD,
    ("red_test", "red_test"): "Grab the red duck.",
    ("color_test", "color_test"): AnnotationList(
        annotations=["Grab the green duck.", "Grab the green animal."],
        keys=["green_duck", "green_animal"],
    ),
    ("sound1", "sound1"): SOUND1_CMD,
    ("dis3", "dis3"): DISAMBIG3_CMD,
    ("test_tac", "test_tac"): TAC_TEST,
    ("t2", "t2"): AnnotationList(
        annotations=["Grab the yellow block."], keys=["simple"]
    ),
    # ('pod', 'pod'): AnnotationList(
    #     annotations=['Grab the object that looks red.', 'Grab the object that feels firm.',
    #                  'Grab the yellow ball', 'Grab the object that looks yellow.', 'Grab the object that feels squishy.'],
    #     keys=['red_block_viz', 'red_block_tac', 'yellow_ball_simple', 'yellow_block_viz', 'yellow_block_tac']
    # ),
    ("rephrase", "rephrase"): AnnotationList(
        annotations=[
            "Grab the object that looks red.",
            "Grab the object that feels firm.",
            "Grab the yellow ball",
            "Grab the object that looks yellow.",
            "Grab the object that feels squishy.",
            "Grab the object that looks yellow and feels squishy.",
        ],
        keys=[
            "red_block_viz",
            "red_block_tac",
            "yellow_ball_simple",
            "yellow_block_viz",
            "yellow_block_tac",
            "yellow_block_tacviz",
        ],
    ),
    # ('pod', 'pod'):  AnnotationList(
    #     annotations=['Grab the red block.', 'Grab the object that looks red.', 'Grab the object that feels firm.',
    #                  'Grab the yellow ball', 'Grab the object that looks yellow.', 'Grab the object that feels squishy.',
    #                  'Grab the object that looks yellow and feels squishy.'],
    #     keys=['red_block_simple', 'red_block_viz', 'red_block_tac', 'yellow_ball_simple', 'yellow_block_viz', 'yellow_block_tac', 'yellow_block_tacviz']
    # ),
    ("pod", "pod"): AnnotationList(
        annotations=[
            "Grab the object that looks red.",
            "Grab the object that feels firm.",
            "Grab the yellow ball",
            "Grab the object that feels squishy.",
            "Grab the object that looks yellow and feels squishy.",
        ],
        keys=[
            "red_block_viz",
            "red_block_tac",
            "yellow_ball_simple",
            "yellow_block_tac",
            "yellow_block_tacviz",
        ],
    ),
    # ('gen', 'gen'): AnnotationList(
    #     annotations=['Grab the object that looks red.', 'Grab the object that feels firm.',
    #                  'Grab the yellow ball', 'Grab the object that feels squishy.',
    #                  'Grab the object that looks yellow and feels squishy.'],
    #     keys=['red_block_viz', 'red_block_tac', 'yellow_ball_simple', 'yellow_block_tac', 'yellow_block_tacviz']
    # ),
    ("gen", "gen"): AnnotationList(
        annotations=[
            "Grab the object that looks green and feels corded.",
            "Grab the object that feels corded.",
        ],
        keys=["green_rope_viztac", "green_rope_tac"],
    ),
    # ('gen', 'gen'): AnnotationList(
    #     annotations=['Grab the object that feels rubbery.', 'Grab the object that looks blue and feels rubbery.'],
    #     keys=['blue_duck_tac', 'blue_duck_viztac']
    # ),
    ("button_train_1", ""): AnnotationList(
        annotations=["Push the pink button.", "Push the button that plays piano."],
        keys=["pink_button_viz", "pink_button_audio"],
    ),
    ("button_train_2", ""): AnnotationList(
        annotations=["Push the green button.", "Push the button that plays metal."],
        keys=["green_button_viz", "green_button_audio"],
    ),
    ("compositional_train_1", ""): AnnotationList(
        annotations=["Grab object that is same color as button that plays rap."],
        keys=["orange_button_comp"],
    ),
    ("compositional_test_1", ""): AnnotationList(
        annotations=["Grab object that is same color as button that plays barking."],
        keys=["yellow_comp"],
    ),
    ("compositional_test_2", ""): AnnotationList(
        annotations=["Grab object that is same color as button that plays squeaking."],
        keys=["purple_comp"],
    ),
    ("flat_test_1", ""): AnnotationList(
        annotations=[
            "Grab the object that feels soft.",
            "Grab the object that looks blue and feels soft.",
        ],
        keys=["blue_tac", "blue_viztac"],
    ),
    ("bag_test_1", ""): AnnotationList(
        annotations=[
            "Grab the object that feels soft.",
            "Grab the object that looks yellow and feels soft.",
        ],
        keys=["yellow_tac", "yellow_viztac"],
    ),
    ("bag_test_2", ""): AnnotationList(
        annotations=[
            "Grab the yellow paper.",
            "Grab the object that looks yellow and feels crinkly.",
        ],
        keys=["yellow_paper_tac", "yellow_paper_viztac"],
    ),
    ("bag_test_3", ""): AnnotationList(
        annotations=["Grab the red block."], keys=["red_block_simple"]
    ),
    ("bag_test_4", ""): AnnotationList(
        annotations=["Grab the blue duck."], keys=["blue_duck_simple"]
    ),
    # ('final_bag', ''): FINAL_BAG_CMDS,
    # ('final_flat', ''): FINAL_FLAT_CMDS,
    # ('sound_test', ''): AnnotationList(
    #     annotations=['Push the yellow button.', 'Push the orange button.', 'Push the pink button.', 'Push the green button.', 'Push the purple button.', 'Push the red button.'],
    #     keys=['yellow', 'orange', 'pink', 'green', 'purple', 'red']
    # )
    ("sound_test", ""): AnnotationList(
        annotations=["Push the blue button.", "Push the pink button."],
        keys=["blue", "magenta"],
    ),
    ("platter_cool", ""): AnnotationList(
        annotations=[
            "Grab the object that looks yellow and feels squishy",
            "Grab the object that looks yellow and feels crinkly",
            "Grab the object that looks blue and feels crinkly",
        ],
        keys=["yellow_ball", "yellow_paper", "blue_paper"],
    ),
    ("bag_cool", ""): AnnotationList(
        annotations=["Grab the object that feels corded."],
        keys=["rope"],
    ),
    ("bag_gen_lang", ""): AnnotationList(
        annotations=["Grab the navy blue rope."],
        keys=["rope"],
    ),
}

# # FOR SIMPLE
for k, v in FINAL_BAG_PAIRS.items():
    info_to_command[(f'bag_{k.split("_")[0]}', "")] = v

for k, v in FINAL_FLAT_PAIRS.items():
    info_to_command[(f'flat_{k.split("_")[0]}', "")] = v
# # END SIMPLE

# FOR MULTI
# for i, (k, v) in enumerate(final_bag_multi.items()):
#     info_to_command[(f'bag_{i}', '')] = v
# for i, (k, v) in enumerate(final_flat_multi.items()):
#     info_to_command[(f'flat_{i}', '')] = v

# END MULTI
for k, v in button_train_cmds.items():
    info_to_command[(k, "")] = v

for k, v in button_test_cmds.items():
    info_to_command[(k, "")] = v

# for k, v in FINAL_BAG_PAIRS.items():
#     print(k, v)
# exit(0)

name_to_command_type = {name: v["Language type"] for name, v in runs.items()}
# NAME_TO_COMMAND = lambda mode, name: info_to_command[(mode, name_to_command_type[name])]
NAME_TO_COMMAND = lambda mode, name: info_to_command[
    (mode, "")
]  # just based on mode flag

PATHS = [
    "all_multi_resnet_32_20240627_075737",
    # "cams_resnet_32_20240627_122302"
]

RESPONSE_TO_RESULT = {
    "s": "success",
    "f": "failure",
    "m": "find",
    "k": "correctcolorwrongtac",
    "t": "wrongcolorcorrecttac",
    "n": "exit",
}
# RESPONSE_TO_RESULT = {
#     '1s': 'success',
#     '1f': 'find',
#     '2s': 'success-correct_color_wrong_tac',
#     '2f': 'find-correct_color_wrong_tac',
#     '3s': 'success-wrong_color_correct_tac',
#     '3f': 'find-wrong_color_correct_tac',
#     '4': 'fail'
# }
# RESPONSE_TO_RESULT = {
#     '1s': 'success',
#     '1f': 'find',
#     '2s': 'success-correct_color_wrong_sound',
#     '2f': 'find-correct_color_wrong_sound',
#     '3s': 'success-wrong_color_correct_sound',
#     '3f': 'find-wrong_color_correct_sound',
#     '4': 'fail'
# }
LANGUAGE_LENGTH = 24
TEXT_PROCESSOR = ModuleSpec.create(
    HFTokenizer,
    tokenizer_name="t5-base",
    encode_with_model=False,
    tokenizer_kwargs={
        "max_length": LANGUAGE_LENGTH,
        "padding": "max_length",
        "truncation": True,
        "return_tensors": "np",
    },
)
# TEXT_PROCESSOR = None
DECODE_LANG = False
DECODE_EXCLUDE = {"josh_pod_ours_no_sound_b1024_20240912_214503"}
TAKE_INPUT = False
SAVE_DIGITS = True
