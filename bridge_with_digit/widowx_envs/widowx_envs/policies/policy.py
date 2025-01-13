""" This file defines the base class for the policy. """
import abc, six
import pickle as pkl
import numpy as np
import pdb

from widowx_envs.utils.utils import AttrDict, Configurable

class Policy(Configurable):
    """Abstract class for policy."""
    def _default_hparams(self):
        dict = AttrDict(
            ngpu=1,
            gpu_id=0,
        )
        default_dict = super()._default_hparams()
        default_dict.update(dict)
        return default_dict

    def act(self, *args):
        """
        Args:
            Request necessary arguments in definition
            (see Agent code)
        Returns:
            A dict of outputs D
               -One key in D, 'actions' should have the action for this time-step
        """
        raise NotImplementedError("Must be implemented in subclass.")

    def reset(self):
        pass

    def set_log_dir(self, dir):
        self.traj_log_dir = dir


class DummyPolicy(Policy):
    def __init__(self, ag_params, policyparams):
        """ Computes actions from states/observations. """
        pass

    def act(self, *args):
        return {'actions': None}

    def reset(self):
        return None


class ReplayActions(Policy):
    def __init__(self, ag_params, policyparams):
        """ Computes actions from states/observations. """
        self._hp = self._default_hparams()
        self._override_defaults(policyparams)
        self.policy_out = pkl.load(open(self._hp.load_file + '/policy_out.pkl', 'rb'))
        self.env = ag_params.env

    def _default_hparams(self):
        dict = AttrDict(
            load_file="",
            type=None,
        )
        default_dict = super(Policy, self)._default_hparams()
        default_dict.update(dict)
        return default_dict

    def act(self, t):
        return self.policy_out[t]

    def reset(self):
        return None


class NullPolicy(Policy):
    """
    Returns 0 for all timesteps
    """
    def __init__(self,  ag_params, policyparams):
        self._adim = ag_params['adim']
        self._hp = self._default_hparams()
        self._override_defaults(policyparams)

    # def _default_hparams(self):
    #     default_dict = {
    #         'wait_for_user': False
    #     }
    #     parent_params = super(NullPolicy, self)._default_hparams()
    #     for k in default_dict.keys():
    #         parent_params.add_hparam(k, default_dict[k])
    #     return parent_params

    def act(self):
        return {'actions': np.zeros(self._adim)}