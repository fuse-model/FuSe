import datetime
from functools import partial
import os

from jax.experimental import multihost_utils
from typing import Union
import jax.numpy as jnp
from absl import app, flags, logging
import flax
from flax.traverse_util import flatten_dict, unflatten_dict
import jax
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from ml_collections import config_flags, ConfigDict
import importlib
from octo.data.utils.data_utils import AnnotationSelectionManager
from octo.model.components.tokenizers import BinTokenizer, LowdimObsTokenizer, ImageTokenizer, UnsqueezingImageTokenizer, ProjectionTokenizer, SiglipTokenizer
from octo.model.octo_module import OctoModule, OctoTransformer
from octo.utils.typing import Data
from octo.utils import jax_utils
import optax
import tensorflow as tf
import tqdm
import wandb
from octo.model.components.vit_encoders import ResNet26, SmallStem32
from octo.model.bcz_model import BczModel
from octo.model.bcz_module import BczModule

from octo.model.components.vit_encoders import SmallStem16
from octo.data.dataset import make_single_dataset
from octo.model.octo_model import OctoModel
from octo.utils.jax_utils import initialize_compilation_cache
from octo.utils.spec import ModuleSpec
from octo.utils.train_callbacks import (
    RolloutVisualizationCallback,
    SaveCallback,
    ValidationCallback,
    VisualizationCallback,
    LanguageCallback,
    GradCAMVisualizationCallback,
)
from octo.utils.train_utils import (
    check_config_diff,
    create_optimizer,
    format_name_with_config,
    merge_params,
    process_text,
    process_lang_list,
    process_dropout_annotations,
    process_fully_batched_rephrase,
    process_fully_batched_rephrase_targets,
    Timer,
    TrainState,
)
import numpy as np

try:
    from jax_smi import initialise_tracking  # type: ignore

    initialise_tracking()
except ImportError:
    pass


#TEMP
from octo.data.utils.text_processing import HFTokenizer

FLAGS = flags.FLAGS

flags.DEFINE_string("name", "experiment", "Experiment name.")
flags.DEFINE_bool("debug", False, "Debug config (no wandb logging)")

default_config_file = os.path.join(
    os.path.dirname(__file__), "configs/finetune_config.py"
)
config_flags.DEFINE_config_file(
    "config",
    default_config_file,
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)

MAX_KEY_LEN = 15
INDENT_SIZE = MAX_KEY_LEN + 4
INDENT = ''.join([' ' for _ in range(INDENT_SIZE)])
def recursive_dict_print(dictionary, prefix="", log_func=print): 
    lines = [] 
    def dfs(dictionary, prefix=""):
        for key, val in dictionary.items(): 
            key = key[:MAX_KEY_LEN]
            if isinstance(val, dict): 
                lines.append(f'{prefix}{key}')
                new_prefix = prefix + INDENT
                dfs(val, new_prefix)
            else: 
                indent = ''.join([' ' for _ in range(INDENT_SIZE - len(key))])
                lines.append(f'{prefix}{key}:{indent}{val.shape}  {val.dtype}')
    dfs(dictionary, prefix)
    return lines


def main(_):
    initialize_compilation_cache()
    devices = jax.devices()
    assert FLAGS.config.batch_size % jax.device_count() == 0
    assert FLAGS.config.batch_size % jax.process_count() == 0
    
    logging.info(
        f"""
        Octo Finetuning Script
        ======================
        Finetuning Dataset: {FLAGS.config.dataset_kwargs.name}
        Data dir: {FLAGS.config.dataset_kwargs.data_dir}
        Task Modality: {FLAGS.config.modality}
        
        # Workers: {jax.process_count()}
        # Devices: {jax.device_count()}
        Batch size: {FLAGS.config.batch_size} ({FLAGS.config.batch_size // len(devices) } per device)
        # Steps: {FLAGS.config.num_steps}
        # Window size: {FLAGS.config.window_size}
    """
    )

    #########
    #
    # Setup Jax Data Parallelism
    #
    #########

    # create a 1D mesh with a single axis named "batch"
    mesh = Mesh(jax.devices(), axis_names="batch")
    
    # replicated sharding -- does not shard arrays
    replicated_sharding = NamedSharding(mesh, PartitionSpec())
    # Our batches will be data-parallel sharded -- each device will get a slice of the batch
    dp_sharding = NamedSharding(mesh, PartitionSpec("batch"))

    def shard(batch):
        return multihost_utils.host_local_array_to_global_array(
            batch, mesh, PartitionSpec("batch")
        )


    # prevent tensorflow from using GPU memory since it's only used for data loading
    tf.config.set_visible_devices([], "GPU")
    
    # make sure each process loads different data
    tf.random.set_seed(FLAGS.config.seed + jax.process_index())

    #########
    #
    # Setup WandB
    #
    #########

    name = format_name_with_config(
        FLAGS.name,
        FLAGS.config.to_dict(),
    )
    wandb_id = "{name}_{time}".format(
        name=name,
        time=datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
    )
    wandb_id = jax_utils.host_broadcast_str(wandb_id)
    if jax.process_index() == 0:
        wandb.init(
            config=FLAGS.config.to_dict(),
            id=wandb_id,
            name=name,
            mode="disabled" if FLAGS.debug else None,
            **FLAGS.config.wandb,
        )


    #########
    #
    # Load Pretrained model + optionally modify config
    #
    #########
    Model = BczModel
    text_processor = ModuleSpec.instantiate(FLAGS.config.text_processor)()

    #########
    #
    # Setup Data Loader
    #
    #########
    dataset = make_single_dataset(
        FLAGS.config.dataset_kwargs,
        traj_transform_kwargs=FLAGS.config.traj_transform_kwargs,
        frame_transform_kwargs=FLAGS.config.frame_transform_kwargs,
        train=True,
    )

    annotation_manager: AnnotationSelectionManager = dataset.supp_info['annotation_manager']

    FLAGS.config.batch_size //= jax.process_count()
    # create text processor
    
    if FLAGS.config['dataset_kwargs']['language_key'] == 'multimodal_annotations':  
        process_text_func = partial(process_dropout_annotations, batch_size=FLAGS.config.batch_size, keys=np.array(dataset.annotation_keys), probabilities=dataset.annotation_probabilities)
    elif FLAGS.config['dataset_kwargs']['language_key'] in  {'all_lang_list', 'rephrase'}: 
        process_text_func = partial(process_lang_list, batch_size=FLAGS.config.batch_size, annotation_manager=annotation_manager)
    elif FLAGS.config['dataset_kwargs']['language_key'] == 'rephrase_batch_full': 
        process_text_func = partial(process_fully_batched_rephrase, batch_size=FLAGS.config.batch_size, annotation_manager=annotation_manager)
    elif FLAGS.config['dataset_kwargs']['language_key'] == 'rephrase_batch_full_target': 
        process_text_func = partial(process_fully_batched_rephrase_targets, batch_size=FLAGS.config.batch_size, annotation_manager=annotation_manager)
    else: 
        # process_text_func = process_text
        raise ValueError(FLAGS.config['dataset_kwargs']['language_key'] )

    def process_batch(batch):
        batch = process_text_func(batch, text_processor)
        del batch["dataset_name"]
        return batch


    train_data_iter = (
        dataset.repeat()
        .unbatch()
        .shuffle(FLAGS.config.shuffle_buffer_size)
        .batch(FLAGS.config.batch_size)
        .iterator()
    )
    train_data_iter = map(process_batch, train_data_iter)
    train_data_iter = map(shard, train_data_iter)
    example_batch = next(train_data_iter)
    logging.info(f"Batch size: {example_batch['action'].shape[0]}")
    logging.info(f"Number of devices: {jax.device_count()}")
    logging.info(
        f"Batch size per device: {example_batch['action'].shape[0] // jax.device_count()}"
    )

    # print example batch 
    example_lines = []
    example_lines.append("############################################")
    example_lines.append('Example batch:')
    example_lines.append('\n\n')
    example_lines.extend(recursive_dict_print(example_batch))

    example_lines.append('\n\n')
    example_lines.append("############################################")
    logging.info('\n'.join(example_lines))

    #########
    #
    # Generation parameters
    #
    #########
    if 'gen_modes' in FLAGS.config:
        gen_modes = FLAGS.config['gen_modes']
    else:
        gen_modes = [('visual',), ('tactile',), ('visual', 'tactile')]

       
    csv_modes = [','.join(modality_tuple) for modality_tuple in gen_modes]
    gen_mode_lang_names = [
        annotation_manager.key_map[mode] for mode in csv_modes
    ]
    modality_obs_keys = {
        'visual': ['image_primary', 'image_wrist'], 
        'tactile': ['image_digit_right', 'image_digit_left', 'asym_tac', 'uniform_tac'],
        'audio': ['mic', 'mel_spectro']
    } 
    
    modality_specific_keys = []
    for v in modality_obs_keys.values(): 
        modality_specific_keys.extend(v)
    modality_specific_keys = set(modality_specific_keys)
    
    includes = ['pad_mask_dict', 'task_completed', 'timestep', 'timestep_pad_mask']


    #########
    #
    # Add in the necessary language heads
    #
    #########
    # if FLAGS.config['lang_head'] and hasattr(annotation_manager, 'reconstruction_loss_keys') and annotation_manager.reconstruction_loss_keys:
    #     reconstruction_head_names = [
    #         f'language_{annotation_manager.key_map[key]}' for key in annotation_manager.reconstruction_loss_keys
    #     ]
    #     for name in reconstruction_head_names: 
    #         config['model']['heads'][name] = FLAGS.config['lang_head']
    if FLAGS.config['lang_head']:
        FLAGS.config['model']['heads']['clip'] = FLAGS.config['lang_head']    

    if FLAGS.config['gen_head']: 
        if FLAGS.config['multi_head']:
            generation_head_names = [f'gen_{lang_name}' for lang_name in csv_modes]
            for name in generation_head_names: 
                FLAGS.config['model']['heads'][name] = FLAGS.config['gen_head']
        else:
            FLAGS.config['model']['heads']['gen'] = FLAGS.config['gen_head']


    rng = jax.random.PRNGKey(FLAGS.config.seed)
    rng, init_rng = jax.random.split(rng)
    model = BczModel.from_config(
        FLAGS.config.to_dict(),
        example_batch,
        text_processor,
        verbose=True,
        rng=init_rng,
        dataset_statistics=dataset.dataset_statistics,
    )
    
    for loader in FLAGS.config.pretrained_loaders:
        if not callable(loader):  # Means that it is a ModuleSpec
            loader = ModuleSpec.instantiate(loader)
        model = model.replace(params=loader(model.params))
    

    #########
    #
    # Setup Optimizer and Train State
    #
    #########



    params = model.params
    if FLAGS.config.optimizer.frozen_keys is None:
        FLAGS.config.optimizer.frozen_keys = model.config["optimizer"]["frozen_keys"]

    tx, lr_callable, param_norm_callable = create_optimizer(
        params,
        **FLAGS.config.optimizer.to_dict(),
    )
    train_state = TrainState.create(
        model=model,
        tx=tx,
        rng=rng,
    )
    train_state = jax.device_get(train_state)

    #########
    #
    # Save all metadata
    #
    #########

    if FLAGS.config.save_dir is not None:
        save_dir = tf.io.gfile.join(
            FLAGS.config.save_dir,
            FLAGS.config.wandb.project,
            FLAGS.config.wandb.group or "",
            wandb_id,
        )
        if jax.process_index() == 0:
            wandb.config.update(dict(save_dir=save_dir), allow_val_change=True)
        logging.info("Saving to %s", save_dir)
        save_callback = SaveCallback(save_dir)

        # Add window_size to top of config, to make eval easier
        new_config = ConfigDict(model.config)
        new_config.annotation_manager = annotation_manager
        new_config["window_size"] = example_batch["observation"][
            "timestep_pad_mask"
        ].shape[1]
        model = model.replace(config=new_config)

        # Save finetuning config since it's not saved by SaveCallback, i.e. as part of model.save_pretrained()
        with tf.io.gfile.GFile(
            tf.io.gfile.join(save_dir, "finetune_config.json"), "w"
        ) as config_file:
            config_file.write(FLAGS.config.to_json_best_effort())
    else:
        save_dir = None
        save_callback = SaveCallback(None)
        logging.warning("save_dir not passed in, not saving checkpoints")

    example_batch_spec = jax.tree_map(
        lambda arr: (arr.shape, str(arr.dtype)), example_batch
    )
    if jax.process_index() == 0:
        wandb.config.update(
            dict(example_batch_spec=example_batch_spec), allow_val_change=True
        )

    #########
    #
    # Define loss, train_step, and eval_step
    #
    #########

    def append_identity_to_metrics(metrics: dict, identity_suffix: str) -> dict: 
        processed_metrics = {}
        for key, val in metrics.items(): 
            processed_metrics[f'{key}_{identity_suffix}'] = val
        return processed_metrics

    def loss_fn_ac(bound_module: BczModule, batch: Data, train: bool = True, specify_lang_key: str = ''):
        encoding = bound_module.bcz_encoder(
                batch["observation"],
                batch["task"],
                batch["observation"]["timestep_pad_mask"],
                train=train,
            )
        action_loss, action_metrics = bound_module.heads["action"].loss(
            encoding,  # action head knows to pull out the "action" readout_key
            batch["action"],
            batch["observation"]["timestep_pad_mask"],
            batch["action_pad_mask"],
            train=train,
        )
        metrics = append_identity_to_metrics(action_metrics, 'ac')
        lang_loss = 0
        if 'language' in bound_module.heads:
            lang_loss, lang_metrics = bound_module.heads['language'].loss(
                encoding, 
                batch['task']['language_instruction'], 
                batch['observation']['timestep_pad_mask'], 
                train=train
            )
            metrics.update(append_identity_to_metrics(lang_metrics, 'bcz_lang'))
        loss = action_loss + lang_loss
        return loss, metrics
    # lang_loss_keys = []
    EVAL_COMBINED = FLAGS.config.get('lang_combined', False)
    lang_loss_keys = [annotation_manager.key_map[key] for key in annotation_manager.reconstruction_loss_keys] if annotation_manager.reconstruction_loss_keys else []
    num_different_annotation_types =  len(lang_loss_keys) + int(EVAL_COMBINED)
    reconstruction_weight = FLAGS.config.get('reconstruction_loss_weight', 0)
    effective_weight =  reconstruction_weight * 1.0 / num_different_annotation_types if num_different_annotation_types else 0.0
    
    def loss_fn_lang(bound_module: BczModule, batch: Data, train: bool = True, eval_unseparated: bool = False, **kwargs): 

        total_loss = 0.0
        info = {}
        cached_lang = batch['task'].pop('language_instruction')
        transformer_embeddings_no_lang = bound_module.octo_transformer(
            batch["observation"],
            batch["task"],
            batch["observation"]["timestep_pad_mask"],
            train=train,
        )
        
        for lang_key in lang_loss_keys: 
            batch['task']['language_instruction'] = batch['task'][lang_key]
            true_language_embeddings = bound_module.octo_transformer.embed_language(batch['task'], train=train)
            # lang_loss, lang_metrics = bound_module.heads[f"language_{lang_key}"].loss(
            #     transformer_embeddings_no_lang,
            #     true_language_embeddings,
            #     batch["observation"]["timestep_pad_mask"],
            #     train=train,
            # )
            lang_loss, lang_metrics = bound_module.heads[f"clip"].loss(
                transformer_embeddings_no_lang,
                true_language_embeddings,
                batch["observation"]["timestep_pad_mask"],
                train=train,
            )
            lang_metrics = append_identity_to_metrics(lang_metrics, identity_suffix=f'contrastive_{annotation_manager.rev_key_map[lang_key]}')      
            total_loss += lang_loss
            info.update(lang_metrics)
        
        batch['task']['language_instruction'] = cached_lang 
        if eval_unseparated:
            true_language_embeddings = bound_module.octo_transformer.embed_language(batch['task'], train=train)
            lang_loss, lang_metrics = bound_module.heads[f"clip"].loss(
                transformer_embeddings_no_lang,
                true_language_embeddings,
                batch["observation"]["timestep_pad_mask"],
                train=train,
            )
            lang_metrics = append_identity_to_metrics(lang_metrics, identity_suffix=f'contrastive_all')      
            total_loss += lang_loss
            info.update(lang_metrics)
        total_loss *= effective_weight
        return total_loss, info

    def create_batch(batch, obs, gen_mode): 
        modality_obs = {}
        for modality_key in gen_mode:
            for obs_key in modality_obs_keys[modality_key]:
                if obs_key in obs: 
                    modality_obs[obs_key] = obs[obs_key]
        for key in includes:
            modality_obs[key] = obs[key]
        batch['observation'] = modality_obs
        return batch
    
    USE_TARGETS = FLAGS.config['dataset_kwargs']['language_key'] == 'rephrase_batch_full_target'

    def get_language_decode_ids_single_head(params, batch, rng, use_targets: bool = USE_TARGETS, train=True): 
        obs = batch['observation']
        cache_lang = batch['task'].pop('language_instruction')
        batch['task']['language_instruction'] = batch['task']['null']
        info = {}
        bound_module = model.module.bind({"params": params}, rngs={"dropout": rng})
        for gen_mode, csv_mode, gen_mode_lang_name in zip(gen_modes, csv_modes, gen_mode_lang_names): 
            batch = create_batch(batch, obs, gen_mode)
            modality_transformer_embedding = bound_module.octo_transformer(
                batch['observation'],
                batch["task"],
                batch["observation"]["timestep_pad_mask"],
                train=train,
            )
            target_key = f'target_{gen_mode_lang_name}' if use_targets else gen_mode_lang_name
            target_lang = batch['task'][target_key]['input_ids']
            decode_ids = bound_module.heads[f"gen"].reconstruct_lang(
                modality_transformer_embedding,
                mode=csv_mode,
                train=train,
            )
            info[f'gen_{csv_mode}'] = [decode_ids, target_lang]
        batch['task']['language_instruction'] = cache_lang
        batch['observation'] = obs
        return info 
        
    def loss_fn_lang_gen_single_head(bound_module: BczModule, batch: Data, train: bool = True, use_targets: bool = USE_TARGETS, **kwargs): 
        total_loss = 0.0
        obs = batch['observation']
        cache_lang = batch['task'].pop('language_instruction')
        batch['task']['language_instruction'] = batch['task']['null']
        info = {}
        for gen_mode, csv_mode, gen_mode_lang_name in zip(gen_modes, csv_modes, gen_mode_lang_names): 
            if 'audio' in gen_mode:
                mask = obs['mic_mask'][:, 0] # remove window dimension
            else:
                mask = None
            batch = create_batch(batch, obs, gen_mode)
            example_lines = []
            example_lines.append("############################################")
            example_lines.append('Example batch:')
            example_lines.append('\n\n')
            example_lines.extend(recursive_dict_print(batch))

            example_lines.append('\n\n')
            example_lines.append("############################################")
            logging.info('\n'.join(example_lines))
            modality_transformer_embedding = bound_module.octo_transformer(
                batch['observation'],
                batch["task"],
                batch["observation"]["timestep_pad_mask"],
                train=train,
            )
            target_key = f'target_{gen_mode_lang_name}' if use_targets else gen_mode_lang_name
            target_lang = batch['task'][target_key]['input_ids']
            gen_loss, gen_metrics = bound_module.heads[f"gen"].loss(
                modality_transformer_embedding,
                target_lang,
                csv_mode, 
                batch["observation"]["timestep_pad_mask"],
                mask=mask,
                train=train,
            )
            gen_metrics = append_identity_to_metrics(gen_metrics, identity_suffix=f'gen_{csv_mode}')      
            total_loss += gen_loss
            info.update(gen_metrics)
        total_loss /= len(gen_modes)
        batch['task']['language_instruction'] = cache_lang
        batch['observation'] = obs
        return total_loss, info
    

    loss_fn_lang_gen = loss_fn_lang_gen_single_head
    get_language_decode_ids = get_language_decode_ids_single_head

    def loss_fn(params, batch, rng, train=True, eval_ac=True, eval_lang=True, gen_lang=True, **kwargs): 
        info = {}
        loss = 0.0
        bound_module = model.module.bind({"params": params}, rngs={"dropout": rng})
        if eval_ac:
            ac_loss, ac_metrics = loss_fn_ac(bound_module, batch, train, **kwargs)
            info.update(ac_metrics)
            loss += ac_loss
            
        if eval_lang: 
            lang_loss, lang_metrics = loss_fn_lang(bound_module, batch, train, **kwargs)
            info.update(lang_metrics)
            loss += lang_loss
        
        if gen_lang:
            gen_loss, gen_metrics = loss_fn_lang_gen(bound_module, batch, train, **kwargs)
            info.update(gen_metrics)
            loss += gen_loss
        
        info['loss_total'] = loss
        return loss, info


    # eval_language = FLAGS.config.get('eval_lang', False)
    eval_language = FLAGS.config['lang_head'] and lang_loss_keys
    gen_language = FLAGS.config['gen_head'] is not None
    # Data parallelism
    # Model is replicated across devices, data is   split across devices
    @partial(
        jax.jit,
        # in_shardings=[replicated_sharding, dp_sharding],
        in_shardings=(replicated_sharding, dp_sharding),
        out_shardings=(replicated_sharding, replicated_sharding),
        # allows jax to modify `state` in-place, saving a lot of memory
        donate_argnums=0,
    )
    def train_step(state: TrainState, batch):
        rng, dropout_rng = jax.random.split(state.rng)
        (loss, info), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            state.model.params, batch, dropout_rng, train=True, eval_lang=eval_language, gen_lang=gen_language,
        )
        grad_norm = optax.global_norm(grads)
        updates, _ = state.tx.update(grads, state.opt_state, state.model.params)
        update_norm = optax.global_norm(updates)
        info.update(
            {
                "grad_norm": grad_norm,
                "update_norm": update_norm,
                "param_norm": param_norm_callable(state.model.params),
                "learning_rate": lr_callable(state.step),
            }
        )
        new_state = state.apply_gradients(grads=grads, rng=rng)
        return new_state, info


    #########
    #
    # Train loop
    #
    #########
    def wandb_log(info, step):
        if jax.process_index() == 0:
            wandb.log(flatten_dict(info, sep="/"), step=step)

    timer = Timer()
    for i in tqdm.tqdm(
        range(0, int(FLAGS.config.num_steps)),
        total=int(FLAGS.config.num_steps),
        dynamic_ncols=True,
    ):
        timer.tick("total")

        with timer("dataset"):
            batch = next(train_data_iter)
        

        with timer("train"):
            train_state, update_info = train_step(train_state, batch)

        timer.tock("total")

        if (i + 1) % FLAGS.config.log_interval == 0:
            update_info = jax.device_get(update_info)
            wandb_log(
                {"training": update_info, "timer": timer.get_average_times()}, step=i
            )

        if (i + 1) % FLAGS.config.save_interval == 0 and save_dir is not None:
            logging.info("Saving checkpoint...")
            save_callback(train_state, i + 1)


if __name__ == "__main__":
    app.run(main)
