import signal
import logging
import jax
import numpy as np

from widowx_envs.widowx_env_service import WidowXClient, WidowXConfigs
from octo.model.octo_model import OctoModel
from eval.envs.gym_wrappers import HistoryWrapper, ObsProcessingWrapper, ResizeImageWrapperDict, TemporalEnsembleWrapper
from eval.envs.widowx_env import WidowXGym

class DelayedKeyboardInterrupt:
    def __enter__(self):
        self.signal_received = False
        self.old_handler = signal.signal(signal.SIGINT, self.handler)
                
    def handler(self, sig, frame):
        self.signal_received = (sig, frame)
    
    def __exit__(self, type, value, traceback):
        signal.signal(signal.SIGINT, self.old_handler)
        if self.signal_received:
            self.old_handler(*self.signal_received)

# for debugging
class DummyClient:
    def __init__(self, im_sizes=None):
        if im_sizes is None:
            im_sizes = {
                'image_0': (256, 256),
                'image_1': (128, 128),
                'digit_l': (224, 224),
                'digit_r': (224, 224),
            }
        self.im_sizes = im_sizes
    
    def init(self, *args, **kwargs):
        pass

    def step_action(self, *args, **kwargs):
        pass

    def get_observation(self):
        return {
            im_key: np.zeros(im_size + (3,)) for im_key, im_size in self.im_sizes.items()
        } | {
            'mic': np.zeros((44100,))
        } | {
            'xyz': [0, 0, 0],
        }

    def reset(self):
        pass

    def start_recording(self):
        pass

    def stop_recording(self, *args, **kwargs):
        pass

def get_env_params(FLAGS):
    if FLAGS.initial_eep is not None:
            assert isinstance(FLAGS.initial_eep, list)
            initial_eep = [float(e) for e in FLAGS.initial_eep]
            start_state = np.concatenate([initial_eep, [0, 0, 0, 1]])
    else:
        start_state = None
    default_env_params = WidowXConfigs.DefaultEnvParams
    env_params = default_env_params.copy() 
    env_params.update(FLAGS.config.env_params)
    env_params["start_state"] = list(start_state)
    return env_params


def initialize_widowx_env(FLAGS): 
    env_params = get_env_params(FLAGS)
    connection_success = False
    if FLAGS.debug_env:
        widowx_client = DummyClient()
    else:
        while not connection_success: 
            try: 
                widowx_client = WidowXClient(host=FLAGS.ip, port=FLAGS.port)
                connection_success = True
            except Exception as e: 
                logging.error(f"Received exception during environment initialization:\n{e}")
                logging.info("Retrying environment initialization...")

    widowx_client.init(env_params, image_size=256)
    env = WidowXGym(
        widowx_client,
        im_sizes=FLAGS.config.received_image_sizes,
        blocking=FLAGS.blocking, 
        sticky_gripper_num_steps=1,
    )
    env = ResizeImageWrapperDict(
        env,
        resize_size=FLAGS.config.resize_map,
    )
    env = ObsProcessingWrapper(
        env=env, 
        remap_keys=FLAGS.config.obs_key_map, 
        new_fields=FLAGS.config.calculated_fields, 
    )
    env = HistoryWrapper(env, horizon=2)
    env = TemporalEnsembleWrapper(env, pred_horizon=FLAGS.exec_horizon)
    return env


def sample_actions(
        pretrained_model: OctoModel,
        observations,
        tasks,
        rng, 
    ):
        observations = jax.tree_map(lambda x: x[None], observations) # add batch dim
        actions = pretrained_model.sample_actions(
            observations,
            tasks,
            unnormalization_statistics=pretrained_model.dataset_statistics["action"],
            rng=rng,
        )
        return actions[0]
