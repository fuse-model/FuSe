import datetime
from functools import partial
import os

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
import optax
import tensorflow as tf
import tqdm
import wandb
from octo.model.components.vit_encoders import ResNet26, SmallStem32

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
    process_batched_rephrase,
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
flags.DEFINE_integer("o_window_size", -1, "override window size")
flags.DEFINE_integer("o_batch_size", -1, "override batch size")
flags.DEFINE_integer("o_steps", -1, "override step ct")
flags.DEFINE_bool('unfreeze_hf', False, 'whether to unfreeze language model')

# # TEMP memory testing
flags.DEFINE_string("mode", "SmallStem16", "temp")
# flags.DEFINE_bool("resnet", False, "temp")
# flags.DEFINE_bool("vit32", False, "temp")
flags.DEFINE_string("log_file", "", "temp")
# # TEMP

default_config_file = os.path.join(
    os.path.dirname(__file__), "configs/finetune_config.py"
)
config_flags.DEFINE_config_file(
    "config",
    default_config_file,
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)


# def config_to_dict(dic: Union[dict, ConfigDict]): 
#     def helper(d): 
#         for k, v in d.items(): 
#             if isinstance(v, Union)
#         pass 
    


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
    if FLAGS.o_batch_size > 0: 
        FLAGS.config.batch_size = FLAGS.o_batch_size
    if FLAGS.o_window_size > 0: 
        FLAGS.config.window_size = FLAGS.o_window_size
 
    logging.info(
        f"""
        Octo Finetuning Script
        ======================
        Pretrained model: {FLAGS.config.pretrained_path}
        Finetuning Dataset: {FLAGS.config.dataset_kwargs.name}
        Data dir: {FLAGS.config.dataset_kwargs.data_dir}
        Task Modality: {FLAGS.config.modality}
        Finetuning Mode: {FLAGS.config.finetuning_mode}

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

    assert (
        FLAGS.config.batch_size % len(devices) == 0
    ), f"Batch size ({FLAGS.config.batch_size}) must be divisible by the number of devices ({len(devices)})"
    assert (
        FLAGS.config.viz_kwargs.eval_batch_size % len(devices) == 0
    ), f"Eval batch size ({FLAGS.config.viz_kwargs.eval_batch_size}) must be divisible by the number of devices ({len(devices)})"

    # create a 1D mesh with a single axis named "batch"
    mesh = Mesh(jax.devices(), axis_names="batch")
    # Our batches will be data-parallel sharded -- each device will get a slice of the batch
    dp_sharding = NamedSharding(mesh, PartitionSpec("batch"))
    # Our model will be replicated across devices (we are only doing data parallelism, not model parallelism)
    replicated_sharding = NamedSharding(mesh, PartitionSpec())

    # prevent tensorflow from using GPU memory since it's only used for data loading
    tf.config.set_visible_devices([], "GPU")

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
    wandb.init(
        config=FLAGS.config.to_dict(),
        id=wandb_id,
        name=name,
        mode="disabled" if FLAGS.debug else None,
        **FLAGS.config.wandb,
    )

    dataset = make_single_dataset(
        FLAGS.config.dataset_kwargs,
        traj_transform_kwargs=FLAGS.config.traj_transform_kwargs,
        frame_transform_kwargs=FLAGS.config.frame_transform_kwargs,
        train=True,
    )

    annotation_manager: AnnotationSelectionManager = dataset.supp_info['annotation_manager']

text_processor_cfg=ModuleSpec.create(
        HFTokenizer,
        tokenizer_name="t5-base",
        encode_with_model=False,
        tokenizer_kwargs={
            "max_length": 32,
            "padding": "max_length",
            "truncation": True,
            "return_tensors": "np",
        },
    )
text_processor = ModuleSpec.instantiate(text_processor_cfg)()

def test_string(string, p): 
    enc = p.encode([string])['input_ids'][0]
    recon = p.decode(enc)
    print(recon)
    return recon

    if FLAGS.config['dataset_kwargs']['language_key'] == 'multimodal_annotations':  
        process_text_func = partial(process_dropout_annotations, batch_size=FLAGS.config.batch_size, keys=np.array(dataset.annotation_keys), probabilities=dataset.annotation_probabilities)
    elif FLAGS.config['dataset_kwargs']['language_key'] in {'all_lang_list', 'rephrase'}: 
        process_text_func = partial(process_lang_list, batch_size=FLAGS.config.batch_size, annotation_manager=annotation_manager)
    elif FLAGS.config['dataset_kwargs']['language_key'] == 'rephrase_batch': 
        process_text_func = partial(process_batched_rephrase, batch_size=FLAGS.config.batch_size, annotation_manager=annotation_manager)
    elif FLAGS.config['dataset_kwargs']['language_key'] == 'language_instruction':
        process_text_func = process_text
    else: 
        raise ValueError(FLAGS.config['dataset_kwargs']['language_key'])
    
    
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
    # train_data_iter = map(process_batch, train_data_iter)
    

    
    example_batch = next(train_data_iter)

    print(example_batch)

    # print example batch 
    example_lines = []
    example_lines.append("\n\n\n\n\n############################################")
    example_lines.append('Example batch:')
    example_lines.append('\n\n')
    example_lines.extend(recursive_dict_print(example_batch))

    example_lines.append('\n\n')
    example_lines.append("############################################")
    save_ex = example_lines
    


    train_data_iter = map(process_batch, train_data_iter)
    example_batch = next(train_data_iter)
    example_lines = []
    example_lines.append("############################################")
    example_lines.append('Example batch:')
    example_lines.append('\n\n')
    example_lines.extend(recursive_dict_print(example_batch))

    example_lines.append('\n\n')
    example_lines.append("############################################")
    logging.info('\n'.join(example_lines))
    logging.info('\n'.join(save_ex))
    exit(0)

    #########
    #
    # Load Pretrained model + optionally modify config
    #
    #########
    pretrained_model_kwargs = {"checkpoint_path": FLAGS.config.pretrained_path}
    if hasattr(FLAGS.config, "pretrained_step"): 
        pretrained_model_kwargs["step"] = step=FLAGS.config.pretrained_step

    pretrained_model = OctoModel.load_pretrained(
        **pretrained_model_kwargs
    )
    flat_config = flax.traverse_util.flatten_dict(
        pretrained_model.config, keep_empty_nodes=True
    )
    for d_key in flax.traverse_util.flatten_dict(
        FLAGS.config.get("config_delete_keys", ConfigDict()).to_dict()
    ):
        for c_key in list(flat_config.keys()):
            if ".".join(c_key).startswith(".".join(d_key)):
                del flat_config[c_key]

    
    update_config = FLAGS.config.get('update_config', ConfigDict())
    flat_update_config = flatten_dict(update_config.to_dict())
    flat_config.update(flat_update_config)

    for k, v in flat_config.items(): 
        print(k, v)
    pop_keys = FLAGS.config.get('pop_keys', [])
    for key in pop_keys: 
        if key not in flat_config: 
            logging.warning(f'{key} not found in flat config, so can\'t pop. Skipping...')
        else: 
            del flat_config[key]
    

    config = ConfigDict(flax.traverse_util.unflatten_dict(flat_config))
    config = config.to_dict()
    check_config_diff(config, pretrained_model.config)

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




    # create text processor
    if config["text_processor"] is None:
        text_processor = None
    else:
        text_processor = ModuleSpec.instantiate(config["text_processor"])()

    if FLAGS.config['dataset_kwargs']['language_key'] == 'multimodal_annotations':  
        process_text_func = partial(process_dropout_annotations, batch_size=FLAGS.config.batch_size, keys=np.array(dataset.annotation_keys), probabilities=dataset.annotation_probabilities)
    elif FLAGS.config['dataset_kwargs']['language_key'] == 'all_lang_list': 
        process_text_func = partial(process_lang_list, batch_size=FLAGS.config.batch_size, annotation_manager=annotation_manager)
    else: 
        process_text_func = process_text

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
    example_batch = next(train_data_iter)


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
    # Add in the necessary language heads
    #
    #########
    reconstruction_head_names = [
        f'language_{annotation_manager.key_map[key]}' for key in annotation_manager.reconstruction_loss_keys
    ]
    for name in reconstruction_head_names: 
        config['model']['heads'][name] = FLAGS.config['lang_head']

    # print('\nConfig heads\n\n')
    # print(*list(config['model']['heads'].items()), sep='\n\n')
    # input('Hit enter:   ')


    #########
    #
    # Create new model and merge parameters
    #
    #########
    rng = jax.random.PRNGKey(FLAGS.config.seed)
    rng, init_rng = jax.random.split(rng)
    config['max_horizon'] = 100 
    config['model']['max_horizon'] = 100
    model = OctoModel.from_config(
        config,
        example_batch,
        text_processor,
        rng=init_rng,
        dataset_statistics=dataset.dataset_statistics,
    )
    merged_params = merge_params(model.params, pretrained_model.params)
    model = model.replace(params=merged_params)
    del pretrained_model
    for loader in FLAGS.config.pretrained_loaders:
        if not callable(loader):  # Means that it is a ModuleSpec
            loader = ModuleSpec.instantiate(loader)
        model = model.replace(params=loader(model.params))


    flattened = flatten_dict(model.params)
    for key in flattened.keys():
        print(key)


    #########
    #
    # Setup Optimizer and Train State
    #
    #########



    params = model.params
    if FLAGS.config.optimizer.frozen_keys is None:
        FLAGS.config.optimizer.frozen_keys = model.config["optimizer"]["frozen_keys"]

    should_unfreeze_hf = (not FLAGS.config.get('freeze_hf', True)) or FLAGS.unfreeze_hf
    if should_unfreeze_hf: 
        FLAGS.config.optimizer.frozen_keys = None

    tx, lr_callable, param_norm_callable = create_optimizer(
        params,
        **FLAGS.config.optimizer.to_dict(),
    )
    train_state = TrainState.create(
        model=model,
        tx=tx,
        rng=rng,
    )

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

    def loss_fn_ac(bound_module: OctoModule, batch: Data, train: bool = True, specify_lang_key: str = ''):
        cache_lang = batch['task']['language_instruction']
        if specify_lang_key: 
            batch['task']['language_instruction'] = batch['task'][annotation_manager.key_map[specify_lang_key]]
        transformer_embeddings = bound_module.octo_transformer(
            batch["observation"],
            batch["task"],
            batch["observation"]["timestep_pad_mask"],
            train=train,
        )
        if specify_lang_key: 
            batch['task']['language_instruction'] = cache_lang
        action_loss, action_metrics = bound_module.heads["action"].loss(
            transformer_embeddings,  # action head knows to pull out the "action" readout_key
            batch["action"],
            batch["observation"]["timestep_pad_mask"],
            batch["action_pad_mask"],
            train=train,
        )
        action_metrics = append_identity_to_metrics(action_metrics, 'ac')
        return action_loss, action_metrics

    lang_loss_keys = [annotation_manager.key_map[key] for key in annotation_manager.reconstruction_loss_keys ]
    num_different_annotation_types =  len(lang_loss_keys)
    reconstruction_weight = FLAGS.config['reconstruction_loss_weight']
    effective_weight =  reconstruction_weight * 1.0 / num_different_annotation_types if num_different_annotation_types else 0.0
    
    def loss_fn_lang(bound_module: OctoModule, batch: Data, train: bool = True, **kwargs): 
        if not lang_loss_keys: 
            return 0.0, {}
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
            lang_loss, lang_metrics = bound_module.heads[f"language_{lang_key}"].loss(
                transformer_embeddings_no_lang,
                true_language_embeddings,
                batch["observation"]["timestep_pad_mask"],
                train=train,
            )
            lang_metrics = append_identity_to_metrics(lang_metrics, identity_suffix=annotation_manager.rev_key_map[lang_key])      
            total_loss += lang_loss
            info.update(lang_metrics)
        batch['task']['language_instruction'] = cached_lang 
        total_loss *= effective_weight
        return total_loss, info


    def loss_fn(params, batch, rng, train=True, eval_ac=True, eval_lang=True, **kwargs): 
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
        
        info['loss_total'] = loss
        return loss, info


    eval_language = FLAGS.config.get('eval_lang', True)
    # Data parallelism
    # Model is replicated across devices, data is split across devices
    @partial(
        jax.jit,
        in_shardings=[replicated_sharding, dp_sharding],
    )
    def train_step(state: TrainState, batch):
        rng, dropout_rng = jax.random.split(state.rng)
        (loss, info), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            state.model.params, batch, dropout_rng, train=True, eval_lang=eval_language
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
    # Build validation & visualization callbacks
    #
    #########

    if FLAGS.config.modality == "image_conditioned":
        modes_to_evaluate = ["image_conditioned"]
    elif FLAGS.config.modality == "text_conditioned":
        modes_to_evaluate = ["text_conditioned"]
    elif FLAGS.config.modality == "multimodal":
        modes_to_evaluate = ["image_conditioned", "text_conditioned"]
    else:
        modes_to_evaluate = ["base"]

    dataset_kwargs_list = [FLAGS.config.dataset_kwargs]
    
    if FLAGS.config.get('skip_val', False):
        val_callback = lambda a,b: {}
    else: 
        val_callback = ValidationCallback(
            loss_fns={ 
                'ac_only': partial(loss_fn, eval_ac=True, eval_lang=False), 
                'combined': loss_fn
            },
            lang_modes = {
                'ac_only': annotation_manager.valid_keys
            },
            process_batch_fn=process_batch,
            text_processor=text_processor,
            val_dataset_kwargs_list=dataset_kwargs_list,
            dataset_kwargs=FLAGS.config,
            modes_to_evaluate=modes_to_evaluate,
            **FLAGS.config.val_kwargs,
        )

    viz_callback = VisualizationCallback(
        text_processor=text_processor,
        val_dataset_kwargs_list=dataset_kwargs_list,
        dataset_kwargs=FLAGS.config,
        modes_to_evaluate=modes_to_evaluate,
        **FLAGS.config.viz_kwargs,
    )

    # gradcam_callback = GradCAMVisualizationCallback(
    #     text_processor=text_processor,
    #     val_dataset_kwargs_list=dataset_kwargs_list, 
    #     dataset_kwargs=FLAGS.config, 
    #     **FLAGS.config.gradcam_kwargs
    # )
    #########
    #
    # Optionally build visualizers for sim env evals
    #
    #########

    if "rollout_kwargs" in FLAGS.config:
        rollout_callback = RolloutVisualizationCallback(
            text_processor=text_processor,
            history_length=FLAGS.config["window_size"],
            model_pred_horizon=config["model"]["heads"]["action"]["kwargs"].get(
                "pred_horizon", 1
            ),
            **FLAGS.config.rollout_kwargs.to_dict(),
        )
    else:
        rollout_callback = None

    #########
    #
    # Train loop
    #
    #########

    def wandb_log(info, step):
        wandb.log(flatten_dict(info, sep="/"), step=step)

    if FLAGS.o_steps > 0: 
        FLAGS.config.num_steps = FLAGS.o_steps
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

        if (i + 1) % FLAGS.config.eval_interval == 0:
            logging.info("Evaluating...")

            with timer("val"):
                val_metrics = val_callback(train_state, i + 1)
                wandb_log(val_metrics, step=i)

            # with timer("visualize"):
            #     viz_metrics = viz_callback(train_state, i + 1)
            #     wandb_log(viz_metrics, step=i)

            # with timer('gradcam'):
            #     gradcam_metrics = gradcam_callback(train_state, i + 1) 
            #     wandb_log(gradcam_metrics, step=i)

        #     if rollout_callback is not None:
        #         with timer("rollout"):
        #             rollout_metrics = rollout_callback(train_state, i + 1)
        #             wandb_log(rollout_metrics, step=i)

        if (i + 1) % FLAGS.config.save_interval == 0 and save_dir is not None:
            logging.info("Saving checkpoint...")
            save_callback(train_state, i + 1)
    if FLAGS.log_file != '': 
        log_str = f"batch_size={FLAGS.config.batch_size}, window_size={FLAGS.config.window_size}, mode={FLAGS.mode}\n"
        with open(FLAGS.log_file, 'a') as file: 
            file.write(log_str)


if __name__ == "__main__":
    app.run(main)
