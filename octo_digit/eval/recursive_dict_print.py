import numpy as np 
MAX_KEY_LEN = 20
INDENT_SIZE = MAX_KEY_LEN + 4
INDENT = ''.join([' ' for _ in range(INDENT_SIZE)])

def recursive_dict_print(dictionary: dict, prefix=""): 
    for key, val in dictionary.items(): 
        key = key[:MAX_KEY_LEN]
        if isinstance(val, dict): 
            print(f'{prefix}{key}')
            new_prefix = prefix + INDENT
            recursive_dict_print(val, new_prefix)
        else:
            indent = ''.join([' ' for _ in range(INDENT_SIZE - len(key))])
            try: 
                print(f'{prefix}{key}:{indent}{val.shape} {val.dtype}')
            except AttributeError: 
                print(f'{prefix}{key}:{indent}     {type(val)}')