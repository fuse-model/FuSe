import os
import os.path
import sys
import numpy as np
from widowx_envs.utils.utils import timed, AttrDict, Configurable
from widowx_envs.utils.raw_saver import RawSaver

sys.path.append('/'.join(str.split(__file__, '/')[:-2]))


class TrajectoryCollector(Configurable):

    def __init__(self, config, gpu_id=0, ngpu=1):
        self._hp = self._default_hparams()
        self._override_defaults(config)
        if self._hp.log_dir is not "":
            self._hp.agent['log_dir'] = self._hp.log_dir
        self.agent = self._hp.agent['type'](self._hp.agent)
        self.agent._hp.env_handle = self.agent.env
        self.agent._hp.gpu_id = gpu_id
        self.agent._hp.ngpu = ngpu

        if isinstance(self._hp.policy, list): # in case multiple policies are used
            self.policies = []
            for policy_param in self._hp.policy:
                self.policies.append(policy_param['type'](self.agent._hp, policy_param))
        else:
            self.policies = [self._hp.policy['type'](self.agent._hp, self._hp.policy)]

        self.trajectory_list = []
        self.im_score_list = []
        try:
            os.remove(self._hp.agent['image_dir'])
        except:
            pass

        self.savers = {}

        self.efforts = []

        if 'raw' in self._hp.save_format:
            self.savers['raw'] = RawSaver(self._hp.data_save_dir)


    def _default_hparams(self):
        default_dict = AttrDict({
            'save_format': ['hdf5', 'raw'],
            'save_data': True,
            'agent': {},
            'policy': {},
            'start_index': -1,
            'end_index': -1,
            'ntraj': -1,
            'gpu_id': -1,
            'current_dir': '',
            'traj_per_file': 1,
            'data_save_dir': '',
            'log_dir': '',
            'split_train_val_test': True,
            'write_traj_summaries': False,
            'collection_metadata': None,
            'make_diagnostics': False
        })
        return default_dict

    def run(self):
        for i in range(self._hp.start_index, self._hp.end_index+1):
            for policy in self.policies:
                self.take_sample(i, policy)

    @timed('traj sample time: ')
    def take_sample(self, index, policy):
        """
        :param index:  run a single trajectory with index
        :return:
        """
        agent_data, obs_dict, policy_out = self.agent.sample(policy, index)
        if self._hp.save_data:
            self.save_data(index, agent_data, obs_dict, policy_out)
        if self._hp.make_diagnostics:
            self.make_diagnostics(obs_dict)

    @timed('savingtime: ')
    def save_data(self, itr, agent_data, obs_dict, policy_outputs):
        for name, saver in self.savers.items(): # if directly saving data
            saver.save_traj(itr, agent_data, obs_dict, policy_outputs)

    def make_diagnostics(self, obs):
        effort = obs['joint_effort']
        max_effort = np.max(np.abs(effort), 0)
        print('max effort', max_effort)
        self.efforts.append(max_effort)
        max_over_all_runs = np.max(np.stack(self.efforts, axis=0), axis=0)
        print('max over all runs', max_over_all_runs)

        xpos = obs['state'][:, 0]
        des_xpos = obs['desired_state'][:, 0]

        images = obs['images'].astype(np.float32)/255
        image_delta = np.abs(images[1:][:, 0] - images[:-1][:, 0])
        tlen = image_delta.shape[0]
        image_delta = np.mean(image_delta.reshape(tlen, -1), axis=1)

        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('agg')
        plt.figure()
        ax = plt.gca()
        ax.plot(xpos, label='xpos')
        plt.plot(des_xpos, label='despos')
        xvalues = range(1, 1+ image_delta.shape[0])
        plt.plot(xvalues, image_delta, label='image_delta')
        ax.legend()
        plt.grid()
        print('saving figure', self._hp.data_save_dir + '/diagnostics.png')
        plt.savefig(self._hp.data_save_dir + '/diagnostics.png')
        # import pdb; pdb.set_trace()

