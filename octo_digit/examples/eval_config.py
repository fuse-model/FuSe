from dataclasses import dataclass
from itertools import combinations
from ckpt_utils import get_all_runs


RESIZE_MAP = {
        'image_0': (256, 256), 
        'image_1': (128, 128),
        'digit_l': (128, 128),
        'digit_r': (128, 128),
        'background_l': (128, 128), 
        'background_r': (128, 128),
}

OBS_KEY_MAP = { 
    "image": { 
        "primary": "image_0", 
        "wrist": "image_1", 
        "digit_left": "digit_l", 
        "digit_right": "digit_r", 
        'digit_left_background': 'background_l', 
        'digit_right_background': 'background_r'
    }, 
}
CALCULATED_FIELDS = [
    "spectro", 
    'digit_embeddings'
]

PREFIX_COMMANDS = {
    'visual': 'looks', 
    'tactile': 'feels', 
    'audio': 'sounds'
}
REMOVE_KEYS = {'', 'audio'}
MULTIMODAL_COMBOS = ['visual', 'tactile', 'visual,tactile,audio']

@dataclass
class MultimodalCommand: 
    annotation_dict: dict[str, str]
    prefix: str = 'Grab the object that'
    def __post_init__(self): 
        sorted_keys = list(self.annotation_dict.keys())
        all_combos = tuple() 
        for i in range(len(sorted_keys)+1): 
            all_combos += tuple(combinations(sorted_keys, i))
        self.string_combos = [','.join(tup) for tup in all_combos]
        def construct_annotation(string_key): 
            annotation = [self.prefix] 
            is_first = True 
            for modality in string_key.split(','): 
                connector = '' if is_first else ' and'
                annotation.append(f"{connector} {PREFIX_COMMANDS[modality]} {self.annotation_dict[modality]}")
                is_first = False
            annotation.append('.')
            return ''.join(annotation)
        key_list = MULTIMODAL_COMBOS if MULTIMODAL_COMBOS else self.string_combos
        self.all_annotations = [construct_annotation(key) for key in key_list if key not in REMOVE_KEYS]
        self.index = 0 
        
    def curr_annotation(self): 
        return self.all_annotations[self.index ]
    def next_annotation(self): 
        self.index = (self.index + 1 ) % len(self.all_annotations)
        return self.curr_annotation()
    def reset(self): 
        self.index = 0
    def is_last(self): 
        return self.index == len(self.all_annotations) - 1 



TRAIN_SIMPLE_CMD = 'Grab and raise the the silver keys.'
TRAIN_MULTI_CMD = MultimodalCommand( 
    annotation_dict={
        'visual': 'silver, large', 
        'tactile': 'metallic, sharp', 
        'audio': 'jangly',
    }
)

TEST_SIMPLE_CMD = 'Grab and raise the red block.'
TEST_MULTI_CMD = MultimodalCommand( 
    annotation_dict={
        'visual': 'red, rectangle', 
        'tactile': 'wooden, matte',
        'audio': 'solid'
    }
)

runs = get_all_runs(save_backup=False)
NAME_TO_WINDOW_SIZE = {name: int(v['Window size']) for name, v in runs.items()}

info_to_command = {
    ('train', 'language_instruction'): TRAIN_SIMPLE_CMD, 
    ('train', 'multimodal_annotations'): TRAIN_MULTI_CMD, 
    ('test', 'language_instruction'): TEST_SIMPLE_CMD, 
    ('test', 'multimodal_annotations'): TEST_MULTI_CMD 
}
name_to_command_type = {name: v['Language key'] for name, v in runs.items()}
NAME_TO_COMMAND = lambda mode, name: info_to_command[(mode, name_to_command_type[name])]

PATHS = [
    "all_multi_resnet_32_20240627_075737",
    # "cams_resnet_32_20240627_122302"    
]

RESPONSE_TO_RESULT = {
    's': 'success', 
    'f': 'failure', 
    'm': 'find', 
    'n': 'exit'
}

