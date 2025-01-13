import pdb
import copy
from random import randrange
from PIL import Image
import numpy as np
import time
import os
from widowx_envs.utils.image_utils import npy_to_gif, npy_to_mp4, resize_store
import cv2
from widowx_envs.utils.utils import Configurable, AttrDict, get_policy_args


from widowx_envs.utils.exceptions import Environment_Exception, Bad_Traj_Exception, Image_Exception


class BlockingLoop(Configurable):
    """
    All communication between the algorithms and MuJoCo is done through
    this class.
    """

    def __init__(self, hyperparams):
        self._hp = self._default_hparams()
        self._override_defaults(hyperparams)

        self.T = self._hp.T
        self._goal = None
        self._goal_seq = None
        self._goal_image = None
        self._demo_images = None
        self._reset_state = None
        self._is_robot = 'robot_name' in hyperparams['env'][1]
        self._setup_world(0)

    def _default_hparams(self):
        default_dict = AttrDict({
            'T':None,
            'adim':None,
            'sdim':None,
            'ncam':1,
            'image_height': 480, 
            'image_width': 640,
            'ndigit': 2,
            'digit_height': 240,
            'digit_width': 320,
            'rejection_sample':False,   # repeatedly attemp to collect a trajectory if error occurs
            'type':None,
            'env':None,
            'nchannels':3,
            'data_save_dir':'',     # path where collected training data will be stored
            'log_dir':'',           # path where logs and viusals will be stored
            'make_final_gif':True,   # whether to make final gif
            'make_final_gif_freq':1,   # final gif, frequency
            'make_final_gif_pointoverlay':False,
            'make_partial_gif': False,
            'video_format': 'gif',
            'recreate_env': (True, 1),  # whether to generate xml, and how often
            'start_goal_confs': None,
            'use_save_thread':False,
            'num_load_steps':2,  # number of steps to load for benchmark to extract goal
            'num_load_cams': 1,   # number of cameras to load for benchmark to extract goal,
            'ask_traj_ok': False,
            'record':None,
            'robot_name': None,
            'imax': 100,
            'load_goal_image':False
        })
        return default_dict

    def _setup_world(self, itr):
        """
        Helper method for handling setup of the MuJoCo world.
        Args:
            filename: Path to XML file containing the world information.
        """
        env_type, env_params, other_params = self._hp.env[0], self._hp.env[1], self._hp.env[2:]
        self.env = env_type(env_params, *other_params)

        self._hp.adim = self.adim = self.env.adim
        self._hp.sdim = self.sdim = self.env.sdim
        self._hp.ncam = self.ncam = self.env.ncam
        self.num_objects = self.env.num_objects

    def reset_env(self, itraj):
        initial_env_obs = self.env.reset(itraj=itraj)
        return initial_env_obs

    def sample(self, policy, i_traj):
        """
        Runs a trial and constructs a new sample containing information
        about the trial.
        """
        self.i_traj = i_traj
        if self._hp.recreate_env[0]:
            if i_traj % self._hp.recreate_env[1] == 0 and i_traj > 0:
                self._setup_world(i_traj)

        traj_ok, obs_dict, policy_outs, agent_data = False, None, None, None
        i_trial = 0
        while not traj_ok and i_trial < self._hp.imax:
            i_trial += 1
            try:
                agent_data, obs_dict, policy_outs = self.rollout(policy, i_trial, i_traj)
                traj_ok = agent_data['traj_ok']
                if self._hp.make_partial_gif:
                    self.save_gif(i_traj)

            except (Image_Exception, Environment_Exception):
                traj_ok = False

        if not traj_ok:
            raise Bad_Traj_Exception

        print('needed {} trials'.format(i_trial))

        if self._hp.make_final_gif or self._hp.make_final_gif_pointoverlay:
            if i_traj % self._hp.make_final_gif_freq == 0:
                self.save_gif(i_traj)

            # if self._goal_image is not None:
            #     for n in range(self.ncam):
            #         im = Image.fromarray((self._goal_image[0, n]*255).astype(np.uint8))
            #         im.save(self.traj_log_dir + '/goal_image_cam{}.jpg'.format(n))

        return agent_data, obs_dict, policy_outs

    def _post_process_obs(self, env_obs, agent_data, initial_obs=False):
        """
        Handles conversion from the environment observations, to agent observation
        space. Observations are accumulated over time, and images are resized to match
        the given image_heightximage_width dimensions.

        Original images from cam index 0 are added to buffer for saving gifs (if needed)

        Data accumlated over time is cached into an observation dict and returned. Data specific to each
        time-step is returned in agent_data

        :param env_obs: observations dictionary returned from the environment
        :param initial_obs: Whether or not this is the first observation in rollout
        :return: obs: dictionary of observations up until (and including) current timestep
        """
        agent_img_height = self._hp.image_height
        agent_img_width = self._hp.image_width

        if initial_obs:
            T = self._hp.T + 1
            self._agent_cache = {}
            for k in env_obs:
                if k == 'images':
                    if 'obj_image_locations' in env_obs:
                        self.traj_points = []
                    self._agent_cache['images'] = np.zeros((T, self._hp.ncam, agent_img_height, agent_img_width, self._hp.nchannels), dtype=np.uint8)
                elif k == 'depth_images':
                    self._agent_cache['depth_images'] = np.zeros((T, env_obs['depth_images'].shape[0], agent_img_height, agent_img_width),
                                                                 dtype=env_obs['depth_images'].dtype)
                elif k == 'digit_images': 
                    self._agent_cache['digit_images'] = np.zeros((T, self._hp.ndigit, self._hp.digit_height, self._hp.digit_width, 3), dtype=np.uint8)
                elif isinstance(env_obs[k], np.ndarray):
                    obs_shape = [T] + list(env_obs[k].shape)
                    self._agent_cache[k] = np.zeros(tuple(obs_shape), dtype=env_obs[k].dtype)
                else:
                    self._agent_cache[k] = []
            self._cache_cntr = 0

        t = self._cache_cntr
        self._cache_cntr += 1

        point_target_width = agent_img_width

        obs = {}
        # import ipdb; ipdb.set_trace()
        if not isinstance(env_obs, dict):
            env_obs = env_obs[0]
        for k in env_obs:
            if k == 'images':
                self.gif_images_traj.append(env_obs['images'])  # only take first camera
                resize_store(t, self._agent_cache['images'], env_obs['images'])
            elif k == 'depth_images':
                resize_store(t, self._agent_cache['depth_images'], env_obs['depth_images'])
            elif k == 'digit_images': 
                resize_store(t, self._agent_cache['digit_images'], env_obs['digit_images'])
            elif k == 'obj_image_locations':
                self.traj_points.append(copy.deepcopy(env_obs['obj_image_locations'][0]))  # only take first camera
                env_obs['obj_image_locations'] = np.round((env_obs['obj_image_locations'] *
                                                           point_target_width / env_obs['images'].shape[2])).astype(
                    np.int64)
                self._agent_cache['obj_image_locations'][t] = env_obs['obj_image_locations']
            elif isinstance(env_obs[k], np.ndarray):
                self._agent_cache[k][t] = env_obs[k]
            else:
                self._agent_cache[k].append(env_obs[k])

            # print('storing at', t)
            obs[k] = self._agent_cache[k][:self._cache_cntr]

        if 'obj_image_locations' in env_obs:
            agent_data['desig_pix'] = env_obs['obj_image_locations']
        if self._goal_image is not None:
            agent_data['goal_image'] = self._goal_image
        if hasattr(self.env, '_goal_arm_pose'):
            agent_data['goal_arm_pose'] = self.env._goal_arm_pose
        if hasattr(self.env, '_goal_obj_pose'):
            agent_data['goal_obj_pose'] = self.env._goal_obj_pose
        if hasattr(self, '_loaded_traj_info') and self._loaded_traj_info is not None:
            agent_data['loaded_traj_info'] = self._loaded_traj_info.copy()
        if self._demo_images is not None:
            agent_data['demo_images'] = self._demo_images
        if self._reset_state is not None:
            agent_data['reset_state'] = self._reset_state
            obs['reset_state'] = self._reset_state

        return obs

    def _required_rollout_metadata(self, agent_data, t, traj_ok):
        """
        Adds meta_data into the agent dictionary that is MANDATORY for later parts of pipeline
        :param agent_data: Agent data dictionary
        :param traj_ok: Whether or not rollout succeeded
        :return: None
        """
        agent_data['term_t'] = t - 1
        if hasattr(self.env, 'eval'):
            agent_data['stats'] = self.env.eval()

    def rollout(self, policy, i_trial, i_traj):
        """
        Rolls out policy for T timesteps
        :param policy: Class extending abstract policy class. Must have act method (see arg passing details)
        :param i_trial: Rollout attempt index (increment each time trajectory fails rollout)
        :return: - agent_data: Dictionary of extra statistics/data collected by agent during rollout
                 - obs: dictionary of environment's observations. Each key maps to that values time-history
                 - policy_ouputs: list of policy's outputs at each timestep.
                 Note: tfrecord saving assumes all keys in agent_data/obs/policy_outputs point to np arrays or primitive int/float
        """
        self._init()

        agent_data, policy_outputs = {}, []

        # Take the sample.
        done = False
        initial_env_obs = self.reset_env(i_traj)
        obs = self._post_process_obs(initial_env_obs, agent_data, True)
        policy.reset()

        self.traj_log_dir = self._hp.log_dir + '/verbose/traj{}'.format(i_traj)
        if not os.path.exists(self.traj_log_dir) and self._hp.log_dir != "":
            os.makedirs(self.traj_log_dir)
        policy.set_log_dir(self.traj_log_dir)

        action_list = []
        self.env.start()
        while not done:
            """
            Every time step send observations to policy, acts in environment, and records observations

            Policy arguments are created by
                - populating a kwarg dict using get_policy_arg
                - calling policy.act with given dictionary

            Policy returns an object (pi_t) where pi_t['actions'] is an action that can be fed to environment
            Environment steps given action and returns an observation
            """
            pi_t = policy.act(**get_policy_args(policy, obs, self._cache_cntr, i_traj, agent_data))
            policy_outputs.append(pi_t)
            if 'done' in pi_t:
                done = pi_t['done']
            try:
                action_list.append(pi_t['actions'])
                obs = self._post_process_obs(self.env.step(pi_t['actions'], blocking=True), agent_data)
                # obs = self._post_process_obs(self.env.step(copy.deepcopy(pi_t['actions']), stage=stage), agent_data, stage=pi_t['policy_index'])
            except Environment_Exception as e:
                print(e)
                return {'traj_ok': False}, None, None

            # print('tstep', t)
            if (self._hp.T - 1) == self._cache_cntr or obs['env_done'][-1]:   # environements can include the tag 'env_done' in the observations to signal that time is over
                done = True
        self.env.finish()

        traj_ok = self.env.valid_rollout()
        if self._hp.rejection_sample:
            assert self.env.has_goal(), 'Rejection sampling enabled but env has no goal'
            traj_ok = self.env.goal_reached()
            print('goal_reached', traj_ok)

        if self._hp.ask_traj_ok:
            traj_ok = self.get_input()

        agent_data['traj_ok'] = traj_ok

        self._required_rollout_metadata(agent_data, self._cache_cntr, traj_ok)

        return agent_data, obs, policy_outputs


    def get_input(self):
        valid = False
        traj_okay = False
        while not valid:
            str = input("Was the trajectory okay? y/n")
            valid = True
            if str == 'y':
                traj_okay = True
            elif str == 'n':
                traj_okay = False
            else:
                print('key invalid!')
                valid = False
        return traj_okay

    def save_gif(self, i_traj, overlay=False):
        if self.traj_points is not None and overlay:
            colors = [tuple([np.random.randint(0, 256) for _ in range(3)]) for __ in range(self.num_objects)]
            for pnts, img in zip(self.traj_points, self.gif_images_traj):
                for i in range(self.num_objects):
                    center = tuple([int(np.round(pnts[i, j])) for j in (1, 0)])
                    cv2.circle(img, center, 4, colors[i], -1)

        for n in range(self.ncam):
            cam_images = [im[n] for im in self.gif_images_traj]
            if self._hp.video_format == 'gif':
                video_func = npy_to_gif
            elif  self._hp.video_format == 'mp4':
                video_func = npy_to_mp4
            else:
                raise NotImplementedError
            video_func(cam_images, self.traj_log_dir + '/video_cam{}'.format(n)) # todo make extra folders for each run?

    def _init(self):
        """
        Set the world to a given model
        """

        def filtering_printer_room(data_frame):
            return data_frame[(data_frame['environment'] == 'bww_printer_room')]
        def filtering_cam1(data_frame):
            return data_frame[(data_frame['camera_index'] == 1)]
        def filtering_cam3(data_frame):
            return data_frame[(data_frame['camera_index'] == 3)]

        if self._hp.load_goal_image:
            if isinstance(self._hp.load_goal_image, str):
                hp = AttrDict(
                    image_size_beforecrop=[self._hp.image_height, self._hp.image_width],
                    data_dir=self._hp.load_goal_image,
                    n_worker=0,
                    stack_goal_images=self._hp.stack_goal_images,
                    filtering_function=[filtering_cam1, ]
                )
                goal_dataset = LMDB_Dataset_Goal(hp, phase='train')
                random_index = randrange(len(goal_dataset))
                self._goal_image = goal_dataset.__getitem__(random_index)
                self._goal_image = [np.expand_dims((image + 1) / 2, axis=0) for image in self._goal_image]
            else:
                goal_dims = (1, self.ncam, self._hp.image_height, self._hp.image_width, 3)
                self._goal_image = np.zeros(goal_dims, dtype=np.uint8)
                for n in range(self.ncam):
                    base_folder, tstep = self._hp.load_goal_image
                    im = cv2.imread(base_folder + '/images{}/im_{}.jpg'.format(n, tstep))
                    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                    self._goal_image[0, n] = cv2.resize(im, (self._hp.image_width, self._hp.image_height),
                                                        interpolation=cv2.INTER_AREA)
                self._goal_image = self._goal_image.astype(np.float32) / 255.
        self.gif_images_traj, self.traj_points = [], None

    def cleanup(self):
        if self._hp.use_save_thread:
            print('Cleaning up file saver....')
            self._save_worker.put(None)
            self._save_worker.join()

    @property
    def record_path(self):
        return self._hp.log_dir+ '/record/'


class TimedLoop(BlockingLoop):
    def __init__(self, hyperparams):
        self._hp = self._default_hparams()
        self._override_defaults(hyperparams)
        super(TimedLoop, self).__init__(hyperparams)

    def _default_hparams(self):
        default_dict = AttrDict(
            ask_confirmation=True,
            absolute_grasp_action=True,  # if False use relative grasp action instead
            ask_traj_ok=True
        )
        # add new params to parent params
        parent_params = super(TimedLoop, self)._default_hparams()
        parent_params.update(default_dict)
        return parent_params

    def rollout(self, policy, i_trial, i_traj):
        """
        Rolls out policy for T timesteps
        :param policy: Class extending abstract policy class. Must have act method (see arg passing details)
        :param i_trial: Rollout attempt index (increment each time trajectory fails rollout)
        :return: - agent_data: Dictionary of extra statistics/data collected by agent during rollout
                 - obs: dictionary of environment's observations. Each key maps to that values time-history
                 - policy_ouputs: list of policy's outputs at each timestep.
                 Note: tfrecord saving assumes all keys in agent_data/obs/policy_outputs point to np arrays or primitive int/float
        """
        self._init()

        agent_data, policy_outputs = {}, []

        # Take the sample.
        done = False
        policy.reset()
        initial_env_obs = self.reset_env(i_traj)
        obs = self._post_process_obs(initial_env_obs, agent_data, True)
        

        self.traj_log_dir = self._hp.log_dir + '/verbose/traj{}'.format(i_traj)
        if not os.path.exists(self.traj_log_dir):
            os.makedirs(self.traj_log_dir)
        policy.set_log_dir(self.traj_log_dir)

        self.env.start()
        step_duration = self.env._hp.move_duration
        last_tstep = time.time()
        while self._cache_cntr < self._hp.T and not done:
            """
            Every time step send observations to policy, acts in environment, and records observations

            Policy arguments are created by
                - populating a kwarg dict using get_policy_arg
                - calling policy.act with given dictionary

            Policy returns an object (pi_t) where pi_t['actions'] is an action that can be fed to environment
            Environment steps given action and returns an observation
            """
            if time.time() > last_tstep + step_duration:
                # print('actual delta t', time.time() - last_tstep)
                if (time.time() - last_tstep) > step_duration*1.05:
                    print('###########################')
                    print('Warning, loop takes too long: {}s!!!'.format(time.time() - last_tstep))
                    print('###########################')
                print('loop  {}s'.format(time.time() - last_tstep))
                last_tstep = time.time()



                print('tstep', self._cache_cntr - 1)
                pi_t = policy.act(**get_policy_args(policy, obs, self._cache_cntr - 1, i_traj, agent_data))
                
                if 'done' in pi_t:
                    done = pi_t['done']
                try:
                    tstamp_return_obs = last_tstep + step_duration
                    obs = self.env.step(pi_t['actions'], tstamp_return_obs, blocking=False)
                    obs = self._post_process_obs(obs, agent_data)
                except Environment_Exception as e:
                    print(e)
                    return {'traj_ok': False}, None, None

                policy_outputs.append(pi_t)

                # import ipdb; ipdb.set_trace()
                if (self._hp.T - 1) == self._cache_cntr or obs['env_done'][-1]:   # environements can include the tag 'env_done' in the observations to signal that time is over
                    done = True
                # print('total exec time', time.time() - tstart)
        self.env.finish()

        traj_ok = self.env.valid_rollout()
        if self._hp.rejection_sample:
            assert self.env.has_goal(), 'Rejection sampling enabled but env has no goal'
            traj_ok = self.env.goal_reached()
            print('goal_reached', traj_ok)

        if hasattr(self.env, "target_img"):
            target_image = self.env.target_img
            obs['target_image'] = target_image

        if self._hp.ask_confirmation:
            traj_ok = self.env.ask_confirmation()
        agent_data['traj_ok'] = traj_ok

        if 'images' in obs:
            agent_data['camera_info'] = self.env.camera_info
        if 'depth_images' in obs:
            agent_data['depth_camera_info'] = self.env.depth_camera_info
        
        self._required_rollout_metadata(agent_data, self._cache_cntr, traj_ok)
        return agent_data, obs, policy_outputs
