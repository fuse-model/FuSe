from dataclasses import dataclass
from functools import partial
import logging
import os
from typing import Callable, Iterable, Mapping, Optional, Tuple

import flax
from flax.training import orbax_utils
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint
import tensorflow as tf
import tqdm

from octo.data.dataset import make_single_dataset
from octo.data.utils.text_processing import TextProcessor
from octo.utils.train_utils import batched_apply, TrainState, process_and_save_text
from octo.utils.typing import Any, Data, Sequence
from octo.utils.visualization_lib import RolloutVisualizer, Visualizer
from octo.utils.gradcam import gradCAM, get_overlaid_attention_map
import ml_collections
from wandb.data_types import Table

class Callback:
    def __call__(self, train_state: TrainState, step: int):
        raise NotImplementedError


def create_validation_dataset(
    dataset_kwargs: dict,
    traj_transform_kwargs: dict,
    frame_transform_kwargs: dict,
    train: bool = False,
):
    """Creates a dataset for validation and visualization purposes.

    Takes the training configuration and overwrites default parameters with more conservative
    options to ensure stable memory consumption.
    """
    return make_single_dataset(
        dataset_kwargs={
            **dataset_kwargs,
            "num_parallel_reads": 4,
            "num_parallel_calls": 4,
            "shuffle": False,
        },
        traj_transform_kwargs={
            **traj_transform_kwargs,
            "num_parallel_calls": 4,
        },
        frame_transform_kwargs={
            **frame_transform_kwargs,
            "num_parallel_calls": 16,
        },
        train=train,
    )


@dataclass
class SaveCallback(Callback):
    """Callback that saves checkpoints to `save_dir`. If `save_dir` is None, does nothing."""

    save_dir: Optional[str]

    def __post_init__(self):
        if self.save_dir is not None:
            if not self.save_dir.startswith("gs://"):
                self.save_dir = os.path.abspath(self.save_dir)
            if jax.process_index() == 0:
                tf.io.gfile.makedirs(self.save_dir)
                logging.info(f"Created {self.save_dir}")
            # make checkpointers
            # only keep latest full TrainState
            self.state_checkpointer = orbax.checkpoint.CheckpointManager(
                tf.io.gfile.join(self.save_dir, "state"),
                orbax.checkpoint.PyTreeCheckpointer(),
                options=orbax.checkpoint.CheckpointManagerOptions(
                    max_to_keep=1,
                ),
            )
            # keep every params checkpoint
            self.params_checkpointer = orbax.checkpoint.CheckpointManager(
                self.save_dir,
                orbax.checkpoint.PyTreeCheckpointer(),
            )

    def __call__(self, train_state: TrainState, step: int):
        if self.save_dir is not None:
            cfg = train_state.model.config
            if isinstance(cfg, ml_collections.config_dict.config_dict.ConfigDict):
                cfg = cfg.to_dict()
            def recursive_checker(dic): 
                for key, val in dic.items(): 
                    if isinstance(val, ml_collections.config_dict.config_dict.ConfigDict): 
                        print("ERROR:  key was ConfigDict, not dict. Converting to dict", key)
                        dic[key] = val.to_dict()
                        recursive_checker(val)
                    elif isinstance(val, dict): 
                        recursive_checker(val)
            
            recursive_checker(cfg)
            train_state.model.save_pretrained(
                step, checkpoint_manager=self.params_checkpointer
            )

                        

            self.state_checkpointer.save(
                step,
                train_state,
                {"save_args": orbax_utils.save_args_from_target(train_state)},
            )


def remove_text(tasks: Data, zero_text_encoding: Optional[Data]):
    """Replaces language encoding inside task dict with that of the empty string.

    zero_text_encoding = jax.tree_map(lambda x: x[0], text_processor.encode([""]))
    """
    if zero_text_encoding is None:
        zero_text_encoding = jnp.zeros((1,))
    if "language_instruction" in tasks:
        new_language = jax.tree_map(
            lambda x, example: jnp.broadcast_to(example[None], x.shape),
            tasks["language_instruction"],
            zero_text_encoding,
        )
        new_pad_dict = flax.core.copy(
            tasks["pad_mask_dict"],
            {
                "language_instruction": jnp.zeros_like(
                    tasks["pad_mask_dict"]["language_instruction"]
                )
            },
        )
        tasks = flax.core.copy(
            tasks, {"language_instruction": new_language, "pad_mask_dict": new_pad_dict}
        )
    return tasks


def remove_images(tasks: Data):
    """Replaces images inside task dict with zero (black) images."""
    updates = {k: jnp.zeros_like(v) for k, v in tasks.items() if "image" in k}
    updates["pad_mask_dict"] = flax.core.copy(
        tasks["pad_mask_dict"],
        {
            k: jnp.zeros_like(v)
            for k, v in tasks["pad_mask_dict"].items()
            if "image" in k
        },
    )
    return flax.core.copy(tasks, updates)


def supply_rng(f, rng=jax.random.PRNGKey(0)):
    def wrapped(*args, **kwargs):
        nonlocal rng
        rng, key = jax.random.split(rng)
        return f(*args, rng=key, **kwargs)

    return wrapped


def get_policy_sampled_actions(
    state: TrainState,
    observations,
    tasks,
    zero_text,
    samples_per_state,
    rng,
    unnormalization_statistics=None,
    policy_mode=None,
):
    if policy_mode == "text_conditioned":
        tasks = remove_images(tasks)
    elif policy_mode == "image_conditioned":
        tasks = remove_text(tasks, zero_text)
    elif policy_mode == "unconditioned":
        tasks = remove_text(remove_images(tasks), zero_text)

    actions = state.model.sample_actions(
        observations,
        tasks,
        unnormalization_statistics=unnormalization_statistics,
        train=False,
        argmax=False,
        sample_shape=(samples_per_state,),
        rng=rng,
    )

    # viz expects (batch_size, n_samples, action_dim)
    actions = jnp.moveaxis(actions, 0, 1)
    return actions


@dataclass
class ValidationCallback(Callback):
    loss_fns: Mapping[str, Callable]
    lang_modes: Mapping[str, Iterable[str]]
    process_batch_fn: Callable[[Data], Data]
    text_processor: Optional[TextProcessor]
    val_dataset_kwargs_list: Sequence[Mapping[str, Any]]
    dataset_kwargs: Mapping[str, Any]
    val_shuffle_buffer_size: int
    num_val_batches: int
    modes_to_evaluate: Sequence[str] = ("text_conditioned", "image_conditioned")
    train: bool = False

    def __post_init__(self):
        if self.text_processor is not None:
            self.zero_text = jax.tree_map(
                lambda x: x[0], self.text_processor.encode("")
            )
        else:
            self.zero_text = None
        self.val_iterators = {}
        for single_dataset_kwargs in self.val_dataset_kwargs_list:
            val_dataset = create_validation_dataset(
                single_dataset_kwargs,
                self.dataset_kwargs["traj_transform_kwargs"],
                self.dataset_kwargs["frame_transform_kwargs"],
                train=self.train,
            )
            val_iterator = (
                val_dataset.unbatch()
                .shuffle(self.val_shuffle_buffer_size)
                .repeat()
                .batch(self.dataset_kwargs["batch_size"])
                .iterator(prefetch=0)
            )
            val_iterator = map(self.process_batch_fn, val_iterator)
            self.val_iterators[single_dataset_kwargs["name"]] = val_iterator
        self.loss_fn_partials = [] 
        for key, loss_fn in self.loss_fns.items(): 
            if key not in self.lang_modes: 
                self.loss_fn_partials.append(loss_fn)
            else: 
                for lang_key in self.lang_modes[key]: 
                    self.loss_fn_partials.append(partial(loss_fn, specify_lang_key=lang_key))
        @partial(
            jax.jit,
            out_shardings=jax.sharding.PositionalSharding(jax.devices()).replicate(),
        )
        def eval_step(state: TrainState, batch: Data):
            
            
            loss_fn_partials = [
                partial(loss_fn, params=state.model.params, rng=state.rng, train=False) for loss_fn in self.loss_fn_partials
            ]
            all_tasks = {}

            if "base" in self.modes_to_evaluate:
                all_tasks["base"] = batch["task"]
            if "image_conditioned" in self.modes_to_evaluate:
                all_tasks["image_conditioned"] = remove_text(
                    batch["task"], self.zero_text
                )
            if "text_conditioned" in self.modes_to_evaluate:
                all_tasks["text_conditioned"] = remove_images(batch["task"])

            if "unconditioned" in self.modes_to_evaluate:
                all_tasks["unconditioned"] = remove_text(
                    remove_images(batch["task"]), self.zero_text
                )
            
            out = {}
            i = 0
            for k, tasks in all_tasks.items(): 
                for loss_fn in loss_fn_partials: 
                    out[f'{k}_{i}'] = loss_fn(batch=flax.core.copy(batch, {"task": tasks}))[1]
            return out
        

        self.eval_step = eval_step

    def __call__(self, train_state: TrainState, step: int):
        wandb_metrics = {}
        for name, val_data_iter in self.val_iterators.items():
            metrics = []
            for _, batch in tqdm.tqdm(
                zip(range(self.num_val_batches), val_data_iter),
                total=self.num_val_batches,
                desc=name,
            ):
                metrics.append(self.eval_step(train_state, batch))
            metrics = jax.tree_map(lambda *xs: np.mean(xs), *metrics)
            wandb_metrics[f"validation_{name}"] = metrics
        return wandb_metrics
    
    
@dataclass
class LanguageCallback(Callback):
    get_ids_fn: Callable
    process_batch_fn: Callable[[Data], Data]
    text_processor: Optional[TextProcessor]
    val_dataset_kwargs_list: Sequence[Mapping[str, Any]]
    dataset_kwargs: Mapping[str, Any]
    val_shuffle_buffer_size: int
    num_val_batches: int
    modes_to_evaluate: Sequence[str] = ("text_conditioned", "image_conditioned")
    train: bool = False

    def __post_init__(self):
        if self.text_processor is not None:
            self.zero_text = jax.tree_map(
                lambda x: x[0], self.text_processor.encode("")
            )
        else:
            self.zero_text = None
        self.val_iterators = {}
        for single_dataset_kwargs in self.val_dataset_kwargs_list:
            val_dataset = create_validation_dataset(
                single_dataset_kwargs,
                self.dataset_kwargs["traj_transform_kwargs"],
                self.dataset_kwargs["frame_transform_kwargs"],
                train=self.train,
            )
            val_iterator = (
                val_dataset.unbatch()
                .shuffle(self.val_shuffle_buffer_size)
                .repeat()
                .batch(self.dataset_kwargs["batch_size"])
                .iterator(prefetch=0)
            )
            val_iterator = map(self.process_batch_fn, val_iterator)
            self.val_iterators[single_dataset_kwargs["name"]] = val_iterator
            
        @partial(
            jax.jit,
            out_shardings=jax.sharding.PositionalSharding(jax.devices()).replicate(),
        )
        def eval_step(state: TrainState, batch: Data):
            get_ids_partial = partial(self.get_ids_fn, params=state.model.params, rng=state.rng, train=False)
        
            out = get_ids_partial(batch=batch)
            return out
        

        self.eval_step = eval_step

    def __call__(self, train_state: TrainState, step: int):
        wandb_metrics = {}
        for name, val_data_iter in self.val_iterators.items():
            metrics = {}
            recon_lang_all = []
            true_lang_all = []
            for _, batch in tqdm.tqdm(
                zip(range(self.num_val_batches), val_data_iter),
                total=self.num_val_batches,
                desc=name,
            ):
                out = self.eval_step(train_state, batch)
                for k, v in out.items(): 
                    recon_ids, true_ids = v
                    true_ids = np.array(true_ids)
                    recon_ids = np.array(recon_ids)
                    num_correct = np.sum(np.all(recon_ids == true_ids, axis=-1))
                    accuracy = 1.0 * num_correct / len(recon_ids)
                    metrics[k] = accuracy
                    recon_lang = self.text_processor.decode(recon_ids[0]) # remove batch for examples
                    true_lang = self.text_processor.decode(true_ids[0])
                    recon_lang_all.append(recon_lang)
                    true_lang_all.append(true_lang)
                
            rows = [list(zipped_lang) for zipped_lang in zip(true_lang_all, recon_lang_all)]
            
            wandb_metrics[f"gen_{step}"] = Table(
                columns=['True', 'Reconstructed'], 
                data=rows
            )
            wandb_metrics.update(metrics)
        return wandb_metrics

@dataclass
class VisualizationCallback(Callback):
    text_processor: TextProcessor
    val_dataset_kwargs_list: Sequence[Mapping[str, Any]]
    dataset_kwargs: Mapping[str, Any]
    eval_batch_size: int
    trajs_for_metrics: int
    trajs_for_viz: int
    samples_per_state: int
    modes_to_evaluate: str = ("text_conditioned", "image_conditioned")
    train: bool = False

    def __post_init__(self):
        if self.text_processor is not None:
            self.zero_text = jax.tree_map(
                lambda x: x[0], self.text_processor.encode("")
            )
        else:
            self.zero_text = None

        self.visualizers = {}
        for single_dataset_kwargs in self.val_dataset_kwargs_list:
            val_dataset = create_validation_dataset(
                single_dataset_kwargs,
                self.dataset_kwargs["traj_transform_kwargs"],
                self.dataset_kwargs["frame_transform_kwargs"],
                train=self.train,
            )
            self.visualizers[single_dataset_kwargs["name"]] = Visualizer(
                val_dataset, text_processor=self.text_processor, freeze_trajs=False
            )

    def __call__(self, train_state: TrainState, step: int):
        wandb_metrics = {}
        modal_policy_fns = {
            mode: batched_apply(
                supply_rng(
                    partial(
                        get_policy_sampled_actions,
                        train_state,
                        zero_text=self.zero_text,
                        samples_per_state=self.samples_per_state,
                        policy_mode=mode,
                    )
                ),
                self.eval_batch_size,
            )
            for mode in self.modes_to_evaluate
        }

        for name, visualizer in self.visualizers.items():
            for mode, policy_fn in modal_policy_fns.items():
                if self.trajs_for_metrics > 0:
                    raw_infos = visualizer.raw_evaluations(
                        policy_fn, max_trajs=self.trajs_for_metrics
                    )
                    metrics = visualizer.metrics_for_wandb(raw_infos)
                    wandb_metrics[f"offline_metrics_{name}/{mode}"] = metrics
                if self.trajs_for_viz > 0:
                    images = visualizer.visualize_for_wandb(
                        policy_fn, max_trajs=self.trajs_for_viz
                    )
                    wandb_metrics[f"visualizations_{name}/{mode}"] = images
        return wandb_metrics

@dataclass
class RolloutVisualizationCallback(Callback):
    visualizer_kwargs_list: Sequence[Mapping[str, Any]]
    text_processor: TextProcessor
    trajs_for_rollouts: int
    unnormalization_statistics: dict
    modes_to_evaluate: str = ("text_conditioned", "image_conditioned")

    def __post_init__(self):
        if self.text_processor is not None:
            self.zero_text = jax.tree_map(
                lambda x: x[0], self.text_processor.encode("")
            )
        else:
            self.zero_text = None

        self.rollout_visualizers = [
            RolloutVisualizer(
                **kwargs,
            )
            for kwargs in self.visualizer_kwargs_list
        ]

    def __call__(self, train_state: TrainState, step: int):
        wandb_metrics = {}
        modal_policy_fns = {
            mode: supply_rng(
                partial(
                    get_policy_sampled_actions,
                    train_state,
                    unnormalization_statistics=self.unnormalization_statistics,
                    zero_text=self.zero_text,
                    samples_per_state=1,
                    policy_mode=mode,
                )
            )
            for mode in self.modes_to_evaluate
        }
        for rollout_visualizer in self.rollout_visualizers:
            for mode, policy_fn in modal_policy_fns.items():
                logging.info(f"Running rollouts for {rollout_visualizer.env_name}")
                rollout_infos = rollout_visualizer.run_rollouts(
                    policy_fn, train_state, mode, n_rollouts=self.trajs_for_rollouts
                )
                wandb_metrics[
                    f"rollouts_{rollout_visualizer.name}/{mode}"
                ] = rollout_infos

        return wandb_metrics


@dataclass
class GradCAMVisualizationCallback(Callback):
    text_processor: TextProcessor
    val_dataset_kwargs_list: Sequence[Mapping[str, Any]]
    dataset_kwargs: Mapping[str, Any] 
    eval_batch_size: int
    shuffle_buffer_size: int 
    train: bool = False
    gradcam_kwargs_list: Sequence[Tuple[str, Mapping[str, str | int]]] = None 
    def __post_init__(self):
        if self.gradcam_kwargs_list is None: 
            self.gradcam_kwargs_list = (
                    ('obs_primary', {'psuedo_loss_type': 'loss'}),
                    ('obs_wrist', {'psuedo_loss_type': 'loss'})
            )
        def process_batch(batch):
            batch = process_and_save_text(batch, self.text_processor)
            del batch["dataset_name"]
            return batch
        self.datasets = []
        for single_dataset_kwargs in self.val_dataset_kwargs_list:
            val_dataset = create_validation_dataset(
                single_dataset_kwargs,
                self.dataset_kwargs["traj_transform_kwargs"],
                self.dataset_kwargs["frame_transform_kwargs"],
                train=self.train,
            )
            val_data_iter = (
                val_dataset.repeat()
                .unbatch()
                .shuffle(self.shuffle_buffer_size)
                .batch(self.eval_batch_size)
                .iterator()
            )
            val_data_iter = map(process_batch, val_data_iter)
            self.datasets.append(val_data_iter)

    def __call__(self, train_state: TrainState, step: int): 
        wandb_metrics = {} 
        rng, dropout_rng = jax.random.split(train_state.rng)
        for i, dataset in enumerate(self.datasets): 
            batch = next(dataset)
            for (obs_key, psuedo_loss_kwargs) in self.gradcam_kwargs_list: 
                original_image, resized_gradcam = gradCAM(
                    train_state.model, train_state.model.params, train_state.model.perturbations, obs_key, batch, dropout_rng, rng=rng, **psuedo_loss_kwargs
                )
                if psuedo_loss_kwargs['psuedo_loss_type'] == 'loss': 
                    psuedo_loss_suffix = 'mse'
                else: 
                    psuedo_loss_suffix = f"pred_{psuedo_loss_kwargs['pred_horizon_step']}_{psuedo_loss_kwargs['action_dim']}"

                wandb_name = f"gradcam_{i}/{obs_key.replace('obs_', '')}_{psuedo_loss_suffix}"

                overlaid_heatmap = get_overlaid_attention_map(original_image, resized_gradcam)
                wandb_metrics[f"{wandb_name}/img"] =  original_image
                wandb_metrics[f"{wandb_name}/gradcam"] = overlaid_heatmap
                wandb_metrics[f"{wandb_name}/command"] = batch['natural_language_instruction']
        return wandb_metrics
                    
