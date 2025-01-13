MAX_KEY_LEN = 15
INDENT_SIZE = MAX_KEY_LEN + 4
INDENT = ''.join([' ' for _ in range(INDENT_SIZE)])
HEADING_SEPARATOR = "############################################"

def print_separator(log_func=print): 
    log_func(HEADING_SEPARATOR)


def pretty_print_dict(dictionary, prefix="", log_func=print, pad_with_newlines=True): 
    lines_to_output = []
    def _pretty_print(dictionary, prefix=""):
        for key, val in dictionary.items(): 
            key = key[:MAX_KEY_LEN]
            if isinstance(val, dict): 
                lines_to_output.append(f'{prefix}{key}')
                _pretty_print(val, prefix + INDENT)
            else: 
                indent = ' ' * (INDENT_SIZE - len(key))
                lines_to_output.append(f'{prefix}{key}:{indent}{val}')
    _pretty_print(dictionary, prefix)
    if pad_with_newlines: 
        lines_to_output = [''] + lines_to_output + ['']
    log_func('\n'.join(lines_to_output))


def append_identity_to_metrics(metrics: dict, identity_suffix: str) -> dict: 
    processed_metrics = {}
    for key, val in metrics.items(): 
        processed_metrics[f'{key}_{identity_suffix}'] = val
    return processed_metrics