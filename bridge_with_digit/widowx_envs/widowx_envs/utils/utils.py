import time
from contextlib import contextmanager
import os
import sys
from funcsigs import signature, Parameter
import numpy as np
import json
import inspect
import yaml

class AttrDict(dict):
    __setattr__ = dict.__setitem__

    def __getattr__(self, attr):
        # Take care that getattr() raises AttributeError, not KeyError.
        # Required e.g. for hasattr(), deepcopy and OrderedDict.
        try:
            return self.__getitem__(attr)
        except KeyError:
            raise AttributeError("Attribute %r not found" % attr)

    def __getstate__(self): return self
    def __setstate__(self, d): self = d


def np_unstack(array, axis):
    arr = np.split(array, array.shape[axis], axis)
    arr = [a.squeeze() for a in arr]
    return arr


def get_policy_args(policy, obs, t, i_tr, step_data=None):
    """
    Generalized way to get data from agent and pass to policy
    :param policy: Subclass of Policy
    :param obs: obs_dict from agent
    :param t: current timestep
    :param i_tr: current traj number
    :param step_data: dict of step specific data passed in by agent
    :return: dictionary mapping keys that the policy requests (by way of argument in policy.act) to their value
    """

    policy_args = {}
    policy_signature = signature(policy.act)  # Gets arguments required by policy
    for arg in policy_signature.parameters:  # Fills out arguments according to their keyword
        if arg == 'args':
            return {}
        value = policy_signature.parameters[arg].default
        if arg in obs:
            value = obs[arg]
        elif step_data is not None and arg in step_data:
            value = step_data[arg]

        # everthing that is not cached in post_process_obs is assigned here:
        elif arg == 't':
            value = t
        elif arg == 'i_tr':
            value = i_tr
        elif arg == 'obs':           # policy can ask for all arguments from environment
            value = obs
        elif arg == 'step_data':
            value = step_data
        elif arg == 'goal_pos':
            value = step_data['goal_pos']

        if value is Parameter.empty:
            # required parameters MUST be set by agent
            raise ValueError("Required Policy Param {} not set in agent".format(arg))
        policy_args[arg] = value
    # import pdb; pdb.set_trace()
    return policy_args

class Configurable(object):
    def _override_defaults(self, policyparams, identical_default_ok=False):
        if policyparams is None:
            return
        for name, value in policyparams.items():
            # if name not in self._hp.keys():
            #     raise ValueError('key {} not in hparam'.format(name))
            # print('overriding param {} to value {}'.format(name, value))
            # if value == getattr(self._hp, name) and not identical_default_ok:
            #     raise ValueError("attribute {} is identical to default value {} !!".format(name, self._hp[name]))
            self._hp[name] = value

    def _default_hparams(self):
        return AttrDict()


@contextmanager
def timing(text):
    start = time.time()
    yield
    elapsed = time.time() - start
    print("{} {}".format(text, elapsed))


class timed:
    """ A function decorator that prints the elapsed time """

    def __init__(self, text):
        """ Decorator parameters """
        self.text = text

    def __call__(self, func):
        """ Wrapping """

        def wrapper(*args, **kwargs):
            with timing(self.text):
                result = func(*args, **kwargs)
            return result

        return wrapper


def map_dict(fn, d):
    """takes a dictionary and applies the function to every element"""
    return type(d)(map(lambda kv: (kv[0], fn(kv[1])), d.items()))


def make_recursive(fn, *argv, **kwargs):
    """ Takes a fn and returns a function that can apply fn on tensor structure
     which can be a single tensor, tuple or a list. """

    def recursive_map(tensors):
        if tensors is None:
            return tensors
        elif isinstance(tensors, list) or isinstance(tensors, tuple):
            return type(tensors)(map(recursive_map, tensors))
        elif isinstance(tensors, dict):
            return type(tensors)(map_dict(recursive_map, tensors))
        else:
            try:
                return fn(tensors, *argv, **kwargs)
            except Exception as e:
                print("The following error was raised when recursively applying a function:")
                print(e)
                raise ValueError("Type {} not supported for recursive map".format(type(tensors)))

    return recursive_map


def map_recursive(fn, tensors):
    return make_recursive(fn)(tensors)


def save_config(confs, exp_conf_path):
    print('saving config to ', exp_conf_path)
    def func(x):
        if inspect.isclass(x):
            return x.__name__
        elif hasattr(x, 'name'):
            return x.name
        else:
            return x
    confs = map_recursive(func, confs)

    if not os.path.exists(exp_conf_path):
        os.makedirs(exp_conf_path)
    with open(exp_conf_path + '/config.json', 'w') as f:
        json.dump(confs, f, indent=4)


if sys.version_info[0] == 2:
    input_fn = raw_input
else:
    input_fn = input

def ask_confirm(text):
    print(text)
    valid = False
    response = False
    while not valid:
        str = input_fn()
        valid = True
        if str == 'y':
            response = True
        elif str == 'n':
            response = False
        else:
            valid = False
    return response

def read_yaml_file(path):
    content = yaml.load(open(os.path.expanduser(path), 'r'), Loader=yaml.CLoader)
    if content is None:
        return {}
    else:
        return content

