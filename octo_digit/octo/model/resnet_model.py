from functools import partial
import json
import logging
from typing import Any, Dict, Optional, Tuple

import flax
from flax import struct
import flax.linen as nn
from flax.training import orbax_utils
import jax
from jax.experimental import multihost_utils
import jax.numpy as jnp
from jax.typing import ArrayLike
import numpy as np
from octo.data.utils.text_processing import TextProcessor
from octo.model.components.action_heads import ActionHead
from octo.model.components.vit_encoders import StdConv, ViTResnet
from octo.model.octo_module import OctoModule
from octo.utils.spec import ModuleSpec
from octo.utils.typing import Config, Data, Params, Perturbations, PRNGKey, Sequence
import orbax.checkpoint
import tensorflow as tf


class ResnetModule(nn.Module):
    mlp_widths: tuple[int]
    image_embedding_size: int
    image_encoder_stages: Sequence[tuple[str, tuple]]
    language_key: Optional[str] = ("language_instruction",)
    action_dim: int = (7,)
    action_pred_horizon: int = (1,)

    @nn.compact
    def __call__(
        self,
        observations,
        tasks,
    ):
        # observations = batch['observation']
        b, w = observations[self.image_encoder_stages[0][0]].shape[:2]
        embeddings = []
        for observation_key, encoder_stages in self.image_encoder_stages:
            embedding = ViTResnet(num_layers=encoder_stages)(
                observations[observation_key]
            )
            embedding = StdConv(self.image_embedding_size, (3, 3))(embedding)
            embedding = jnp.mean(embedding, axis=(-2, -3))  # GAP
            embeddings.append(embedding)

        lang = jnp.tile(
            tasks[self.language_key][:, None, ...], (1, w, 1)
        )  # repeat task embedding over window: b, w, dim
        embeddings.append(lang)
        x = jnp.concatenate(embeddings, axis=-1)
        for width in self.mlp_widths:
            x = nn.Dense(width)(x)
        x = nn.Dense(self.action_dim * self.action_pred_horizon)(x)
        x = jnp.reshape(x, (b, w, self.action_pred_horizon, self.action_dim))
        return x

    @classmethod
    def create(
        cls,
        mlp_widths: tuple[int],
        image_embedding_size: int,
        image_encoder_stages: Optional[Sequence[tuple[str, tuple]]] = None,
        language_key: Optional[str] = "language_instruction",
        action_dim: int = 7,
        action_pred_horizon: int = 1,
    ) -> "OctoModule":

        if image_encoder_stages is None:
            image_encoder_stages = (
                ("image_primary", (2, 2, 2, 2)),
                ("image_wrist", (2, 2, 2, 2)),
            )

        return cls(
            mlp_widths,
            image_embedding_size,
            image_encoder_stages,
            language_key,
            action_dim,
            action_pred_horizon,
        )


@struct.dataclass
class ResnetModel:
    module: ResnetModule = struct.field(pytree_node=False)
    text_processor: TextProcessor = struct.field(pytree_node=False)
    config: Config = struct.field(pytree_node=False)
    params: Params
    perturbations: Perturbations
    example_batch: Data
    dataset_statistics: Optional[Data]

    def create_tasks(self, texts: Sequence[str] = None):
        tasks = {}

        assert self.text_processor is not None
        tasks["language_instruction"] = texts
        # tasks["pad_mask_dict"]["language_instruction"] = np.ones(
        #     len(texts), dtype=bool
        # )

        tasks["language_instruction"] = self.text_processor.encode(
            tasks["language_instruction"]
        )

        _verify_shapes(tasks, "tasks", self.example_batch["task"], starting_dim=1)
        return tasks

    @partial(jax.jit, static_argnames=())
    def run_resnet(
        self,
        observations: Data,
        tasks: Data,
    ):

        # print('hi')
        # _verify_shapes(
        #     observations,
        #     "observations",
        #     self.example_batch["observation"],
        #     starting_dim=2,
        # )
        # _verify_shapes(tasks, "tasks", self.example_batch["task"], starting_dim=1)
        # print('yo')
        _verify_shapes(
            observations,
            "observations",
            self.example_batch["observation"],
            starting_dim=2,
        )
        _verify_shapes(tasks, "tasks", self.example_batch["task"], starting_dim=1)

        params_kwargs = {"params": params, "perturbations": perturbations}
        return self.module.apply(
            {"params": self.params}, observations, tasks, method="self"
        )

    @partial(jax.jit, static_argnames=())
    def sample_actions(
        self,
        observations: Data,
        tasks: Data,
        unnormalization_statistics: Optional[Data] = None,
    ):
        # print("here")
        return jnp.zeros((1, 2))
        # action = self.run_resnet(
        #     observations, tasks
        # )
        # print(action.shape)
        # exit(0)
        if unnormalization_statistics is not None:
            mask = unnormalization_statistics.get(
                "mask", jnp.ones_like(unnormalization_statistics["mean"], dtype=bool)
            )
            action = action[..., : len(mask)]
            action = jnp.where(
                mask,
                (action * unnormalization_statistics["std"])
                + unnormalization_statistics["mean"],
                action,
            )
        return action

    @classmethod
    def load_pretrained(
        cls,
        checkpoint_path: str,
        text_processor: TextProcessor,
        dataset_statistics,
        step: Optional[int] = None,
    ) -> "ResnetModel":
        """Loads a model from a checkpoint that was saved via `save_pretrained`.

        Args:
            checkpoint_path (str): A path to either a directory of checkpoints or a single checkpoint.
            step (int, optional): If multiple checkpoints are present, which one to load. Defaults to the latest.
        """
        # load config
        with tf.io.gfile.GFile(
            tf.io.gfile.join(checkpoint_path, "config.json"), "r"
        ) as f:
            config = json.load(f)

        # load example batch
        with tf.io.gfile.GFile(
            tf.io.gfile.join(checkpoint_path, "example_batch.msgpack"), "rb"
        ) as f:
            example_batch = flax.serialization.msgpack_restore(f.read())
        # shim for migrating from "tasks" to "task"
        if "tasks" in example_batch:
            example_batch["task"] = example_batch.pop("tasks")

        logging.debug(
            "Model was trained with observations: %s",
            flax.core.pretty_repr(
                jax.tree_map(jnp.shape, example_batch["observation"])
            ),
        )
        logging.debug(
            "Model was trained with tasks: %s",
            flax.core.pretty_repr(jax.tree_map(jnp.shape, example_batch["task"])),
        )

        # load dataset statistics
        with tf.io.gfile.GFile(
            tf.io.gfile.join(checkpoint_path, "dataset_statistics.json"), "r"
        ) as f:
            dataset_statistics = json.load(f)
            dataset_statistics = jax.tree_map(
                np.array, dataset_statistics, is_leaf=lambda x: not isinstance(x, dict)
            )

        # create model def (an OctoModule)
        module = ResnetModule.create(**config["model"])
        # infer params shape without actually doing any computation

        init_args = (example_batch["observation"], example_batch["task"])
        perturbations = module.init(jax.random.PRNGKey(0), *init_args).get(
            "perturbations", None
        )
        params_shape = jax.eval_shape(
            partial(module.init), jax.random.PRNGKey(0), *init_args
        )["params"]
        # restore params, checking to make sure the shape matches
        checkpointer = orbax.checkpoint.CheckpointManager(
            checkpoint_path, orbax.checkpoint.PyTreeCheckpointer()
        )
        step = step if step is not None else checkpointer.latest_step()
        params = checkpointer.restore(step, params_shape)

        # if config["text_processor"] is not None:
        #     text_processor = ModuleSpec.instantiate(config["text_processor"])()
        # else:
        #     text_processor = None

        return cls(
            module=module,
            params=params,
            perturbations=perturbations,
            text_processor=text_processor,
            example_batch=example_batch,
            config=config,
            dataset_statistics=dataset_statistics,
        )

    def save_pretrained(
        self,
        step: int,
        checkpoint_path: Optional[str] = None,
        checkpoint_manager: Optional[orbax.checkpoint.CheckpointManager] = None,
    ):
        """Saves a model, as well as corresponding metadata needed for `load_pretrained`. Takes either a
        pre-existing checkpoint manager (which already knows where to save the checkpoint) or a path to a
        directory to save the checkpoint to.

        Args:
            step (int): Step number.
            checkpoint_path (str, optional): Path to save the checkpoint.
            checkpoint_manager (optional): Checkpoint manager to save the checkpoint.
            params (optional): Params to save. If None, uses self.params.
        """
        if (checkpoint_path is None) == (checkpoint_manager is None):
            raise ValueError(
                "Must provide exactly one of checkpoint_path or checkpoint_manager."
            )
        if checkpoint_manager is None:
            # checkpoint_manager = orbax.checkpoint.CheckpointManager(
            #     checkpoint_path, orbax.checkpoint.PyTreeCheckpointer()
            # )
            raise RuntimeError
        if checkpoint_path is None:
            checkpoint_path = str(checkpoint_manager._directory)

        # save params
        checkpoint_manager.save(
            step,
            self.params,
            {"save_args": orbax_utils.save_args_from_target(self.params)},
        )

        if jax.process_index() == 0:
            # save config
            config_path = tf.io.gfile.join(checkpoint_path, "config.json")
            if not tf.io.gfile.exists(config_path):
                with tf.io.gfile.GFile(config_path, "w") as f:
                    json.dump(self.config, f)

            # save example batch
            example_batch_path = tf.io.gfile.join(
                checkpoint_path, "example_batch.msgpack"
            )
            if not tf.io.gfile.exists(example_batch_path):
                with tf.io.gfile.GFile(example_batch_path, "wb") as f:
                    f.write(flax.serialization.msgpack_serialize(self.example_batch))

            # save dataset statistics
            dataset_statistics_path = tf.io.gfile.join(
                checkpoint_path, "dataset_statistics.json"
            )
            if not tf.io.gfile.exists(dataset_statistics_path):
                with tf.io.gfile.GFile(dataset_statistics_path, "w") as f:
                    json.dump(
                        jax.tree_map(lambda x: x.tolist(), self.dataset_statistics),
                        f,
                    )

    @classmethod
    def from_config(
        cls,
        config: Config,
        example_batch: Data,
        text_processor: Optional[Any] = None,
        verbose: bool = False,
        rng: Optional[PRNGKey] = None,
        dataset_statistics: Optional[Data] = None,
    ):

        module = ResnetModule.create(**config["model"])
        rng = rng if rng is not None else jax.random.PRNGKey(0)
        example_batch = multihost_utils.process_allgather(example_batch)
        example_batch = jax.tree_map(lambda x: x[:1], example_batch)

        init_args = (example_batch["observation"], example_batch["task"])

        if verbose:
            print(
                module.tabulate(rng, *init_args, verbose=True, depth=2)
            )  # Prints out the parameter count of our model, and tokenizer details

        @jax.jit
        def _init(rng):
            return module.init(rng, *init_args)

        variables = _init(rng)
        params = variables["params"]
        perturbations = variables.get("perturbations", None)

        return cls(
            module=module,
            params=params,
            perturbations=perturbations,
            text_processor=text_processor,
            example_batch=example_batch,
            config=config,
            dataset_statistics=dataset_statistics,
        )


def _verify_shapes(
    pytree,
    name: str,
    example_pytree,
    starting_dim: int = 0,
    strict: bool = False,
    raise_error: bool = True,
    silent: bool = False,
):
    weak_fail, fail = False, False
    pytree_flat = flax.traverse_util.flatten_dict(pytree)
    example_pytree_flat = flax.traverse_util.flatten_dict(example_pytree)

    # Check that all elements are present
    if set(pytree_flat.keys()) != set(example_pytree_flat.keys()):
        if not silent:
            extra = set(pytree_flat.keys()) - set(example_pytree_flat.keys())
            if extra:
                logging.warning(
                    "'%s' contains extra items compared to example_batch: %s",
                    name,
                    {"/".join(x) for x in extra},
                )
            missing = set(example_pytree_flat.keys()) - set(pytree_flat.keys())
            if missing:
                logging.warning(
                    "'%s' is missing items compared to example_batch: %s",
                    name,
                    {"/".join(x) for x in missing},
                )
        weak_fail = True

    mismatched_keys = {
        k: f"{pytree_flat[k].shape} != {example_pytree_flat[k].shape}"
        for k in pytree_flat
        if k in example_pytree_flat
        and pytree_flat[k].shape[starting_dim:]
        != example_pytree_flat[k].shape[starting_dim:]
    }
    if mismatched_keys:
        if not silent:
            logging.error(
                "'%s' contains mismatched shapes compared to example_batch: %s",
                name,
                flax.core.pretty_repr(
                    {"/".join(k): v for k, v in mismatched_keys.items()}
                ),
            )
        fail = True

    if raise_error and (fail or (weak_fail and strict)):
        raise AssertionError(f"{name} does not match example batch.")

    return weak_fail or fail


SPEC_TEMPLATE = """
This model is trained with a window size of {window_size}, predicting {action_dim} dimensional actions {action_horizon} steps into the future.
Observations and tasks conform to the following spec:

Observations: {observation_space}
Tasks: {task_space}

At inference, you may pass in any subset of these observation and task keys, with a history window up to {window_size} timesteps.
"""
# from functools import partial
# import json
# import logging
# from typing import Any, Optional, Tuple, Dict

# import flax
# from flax import struct
# from flax.training import orbax_utils
# import jax
# from jax.experimental import multihost_utils
# import jax.numpy as jnp
# from jax.typing import ArrayLike
# import numpy as np
# import orbax.checkpoint
# import tensorflow as tf
# import flax.linen as nn

# from octo.data.utils.text_processing import TextProcessor
# from octo.model.components.action_heads import ActionHead
# from octo.model.octo_module import OctoModule
# from octo.utils.spec import ModuleSpec
# from octo.utils.typing import Config, Data, Params, PRNGKey, Perturbations, Sequence
# from octo.model.components.vit_encoders import StdConv, ViTResnet

# class ResnetModule(nn.Module):
#     mlp_widths: tuple[int]
#     image_embedding_size: int
#     image_encoder_stages: Sequence[tuple[str, tuple]]
#     language_key: Optional[str] = "language_instruction",
#     action_dim: int = 7,
#     action_pred_horizon: int = 1,

#     @nn.compact
#     def __call__(
#         self,
#         batch: Data,
#     ):
#         observations = batch['observation']
#         b, w = observations[self.image_encoder_stages[0][0]].shape[:2]
#         embeddings = []
#         for observation_key, encoder_stages in self.image_encoder_stages:
#             embedding = ViTResnet(num_layers=encoder_stages)(observations[observation_key])
#             embedding = StdConv(
#                 self.image_embedding_size,
#                 (3, 3)
#             )(embedding)
#             embedding = jnp.mean(embedding, axis = (-2, -3)) # GAP
#             embeddings.append(embedding)

#         lang = jnp.tile(batch['task'][self.language_key][:, None, ...], (1, w, 1)) # repeat task embedding over window
#         embeddings.append(lang)
#         x = jnp.concatenate(embeddings, axis=-1)
#         x = jnp.reshape(b, -1)
#         for width in self.mlp_widths:
#             x = nn.Dense(width)(x)
#         x = nn.Dense(self.action_dim * self.action_pred_horizon)(x)
#         x = jnp.reshape(x, (-1, self.action_pred_horizon, self.action_dim))
#         return x

#     @classmethod
#     def create(
#         cls,
#         mlp_widths: tuple[int],
#         image_embedding_size: int,
#         image_encoder_stages: Optional[Sequence[tuple[str, tuple]]] = None,
#         language_key: Optional[str] = "language_instruction",
#         action_dim: int = 7,
#         action_pred_horizon: int = 1,
#     ) -> "OctoModule":

#         if image_encoder_stages is None:
#            image_encoder_stages = (("image_primary", (2, 2, 2, 2)), ("image_wrist", (2, 2, 2, 2)))

#         return cls(
#             mlp_widths,
#             image_embedding_size,
#             image_encoder_stages,
#             language_key,
#             action_dim,
#             action_pred_horizon,
#         )


# @struct.dataclass
# class ResnetModel:
#     module: ResnetModule = struct.field(pytree_node=False)
#     text_processor: TextProcessor = struct.field(pytree_node=False)
#     config: Config = struct.field(pytree_node=False)
#     params: Params
#     perturbations: Perturbations
#     example_batch: Data
#     dataset_statistics: Optional[Data]

#     def create_tasks(
#         self, texts: Sequence[str] = None
#     ):
#         tasks = {}

#         assert self.text_processor is not None
#         tasks["language_instruction"] = texts
#         tasks["pad_mask_dict"]["language_instruction"] = np.ones(
#             len(texts), dtype=bool
#         )

#         tasks["language_instruction"] = self.text_processor.encode(
#             tasks["language_instruction"]
#         )

#         _verify_shapes(tasks, "tasks", self.example_batch["task"], starting_dim=1)
#         return tasks


#     @partial(jax.jit, static_argnames=("train",))
#     def run_resnet(
#         self,
#         observations: Data,
#         tasks: Data,
#     ):
#         _verify_shapes(
#             observations,
#             "observations",
#             self.example_batch["observation"],
#             starting_dim=2,
#         )
#         _verify_shapes(tasks, "tasks", self.example_batch["task"], starting_dim=1)

#         return self.module.apply(
#             {"params": self.params},
#             observations,
#             tasks
#         )


#     @partial(
#         jax.jit,
#         static_argnames=("train", "sample_shape", "argmax"),
#     )
#     def sample_actions(
#         self,
#         observations: Data,
#         tasks: Data,
#         unnormalization_statistics: Optional[Data] = None,
#     ):
#         action = self.run_resnet(
#             observations, tasks
#         )
#         if unnormalization_statistics is not None:
#             mask = unnormalization_statistics.get(
#                 "mask", jnp.ones_like(unnormalization_statistics["mean"], dtype=bool)
#             )
#             action = action[..., : len(mask)]
#             action = jnp.where(
#                 mask,
#                 (action * unnormalization_statistics["std"])
#                 + unnormalization_statistics["mean"],
#                 action,
#             )
#         return action

#     @classmethod
#     def load_pretrained(
#         cls,
#         checkpoint_path: str,
#         step: Optional[int] = None,
#     ) -> "ResnetModel":
#         """Loads a model from a checkpoint that was saved via `save_pretrained`.

#         Args:
#             checkpoint_path (str): A path to either a directory of checkpoints or a single checkpoint.
#             step (int, optional): If multiple checkpoints are present, which one to load. Defaults to the latest.
#         """

#         # load config
#         with tf.io.gfile.GFile(
#             tf.io.gfile.join(checkpoint_path, "config.json"), "r"
#         ) as f:
#             config = json.load(f)

#         # load example batch
#         with tf.io.gfile.GFile(
#             tf.io.gfile.join(checkpoint_path, "example_batch.msgpack"), "rb"
#         ) as f:
#             example_batch = flax.serialization.msgpack_restore(f.read())
#         # shim for migrating from "tasks" to "task"
#         if "tasks" in example_batch:
#             example_batch["task"] = example_batch.pop("tasks")

#         logging.debug(
#             "Model was trained with observations: %s",
#             flax.core.pretty_repr(
#                 jax.tree_map(jnp.shape, example_batch["observation"])
#             ),
#         )
#         logging.debug(
#             "Model was trained with tasks: %s",
#             flax.core.pretty_repr(jax.tree_map(jnp.shape, example_batch["task"])),
#         )

#         # load dataset statistics
#         with tf.io.gfile.GFile(
#             tf.io.gfile.join(checkpoint_path, "dataset_statistics.json"), "r"
#         ) as f:
#             dataset_statistics = json.load(f)
#             dataset_statistics = jax.tree_map(
#                 np.array, dataset_statistics, is_leaf=lambda x: not isinstance(x, dict)
#             )

#         # create model def (an OctoModule)
#         module = ResnetModule.create(**config["model"])
#         # infer params shape without actually doing any computation


#         init_args = (
#             example_batch["observation"],
#             example_batch["task"],
#             example_batch["observation"]["timestep_pad_mask"],
#         )
#         perturbations = module.init(jax.random.PRNGKey(0), *init_args, train=False)['perturbations']
#         params_shape = jax.eval_shape(
#             partial(module.init, train=False), jax.random.PRNGKey(0), *init_args
#         )["params"]
#         # restore params, checking to make sure the shape matches
#         checkpointer = orbax.checkpoint.CheckpointManager(
#             checkpoint_path, orbax.checkpoint.PyTreeCheckpointer()
#         )
#         step = step if step is not None else checkpointer.latest_step()
#         params = checkpointer.restore(step, params_shape)

#         if config["text_processor"] is not None:
#             text_processor = ModuleSpec.instantiate(config["text_processor"])()
#         else:
#             text_processor = None

#         return cls(
#             module=module,
#             params=params,
#             perturbations=perturbations,
#             text_processor=text_processor,
#             example_batch=example_batch,
#             config=config,
#             dataset_statistics=dataset_statistics,
#         )

#     def save_pretrained(
#         self,
#         step: int,
#         checkpoint_path: Optional[str] = None,
#         checkpoint_manager: Optional[orbax.checkpoint.CheckpointManager] = None,
#     ):
#         """Saves a model, as well as corresponding metadata needed for `load_pretrained`. Takes either a
#         pre-existing checkpoint manager (which already knows where to save the checkpoint) or a path to a
#         directory to save the checkpoint to.

#         Args:
#             step (int): Step number.
#             checkpoint_path (str, optional): Path to save the checkpoint.
#             checkpoint_manager (optional): Checkpoint manager to save the checkpoint.
#             params (optional): Params to save. If None, uses self.params.
#         """
#         if (checkpoint_path is None) == (checkpoint_manager is None):
#             raise ValueError(
#                 "Must provide exactly one of checkpoint_path or checkpoint_manager."
#             )
#         if checkpoint_manager is None:
#             checkpoint_manager = orbax.checkpoint.CheckpointManager(
#                 checkpoint_path, orbax.checkpoint.PyTreeCheckpointer()
#             )
#         if checkpoint_path is None:
#             checkpoint_path = str(checkpoint_manager._directory)

#         # save params
#         checkpoint_manager.save(
#             step,
#             self.params,
#             {"save_args": orbax_utils.save_args_from_target(self.params)},
#         )

#         if jax.process_index() == 0:
#             # save config
#             config_path = tf.io.gfile.join(checkpoint_path, "config.json")
#             if not tf.io.gfile.exists(config_path):
#                 with tf.io.gfile.GFile(config_path, "w") as f:
#                     json.dump(self.config, f)

#             # save example batch
#             example_batch_path = tf.io.gfile.join(
#                 checkpoint_path, "example_batch.msgpack"
#             )
#             if not tf.io.gfile.exists(example_batch_path):
#                 with tf.io.gfile.GFile(example_batch_path, "wb") as f:
#                     f.write(flax.serialization.msgpack_serialize(self.example_batch))

#             # save dataset statistics
#             dataset_statistics_path = tf.io.gfile.join(
#                 checkpoint_path, "dataset_statistics.json"
#             )
#             if not tf.io.gfile.exists(dataset_statistics_path):
#                 with tf.io.gfile.GFile(dataset_statistics_path, "w") as f:
#                     json.dump(
#                         jax.tree_map(lambda x: x.tolist(), self.dataset_statistics),
#                         f,
#                     )

#     @classmethod
#     def from_config(
#         cls,
#         config: Config,
#         example_batch: Data,
#         text_processor: Optional[Any] = None,
#         verbose: bool = False,
#         rng: Optional[PRNGKey] = None,
#         dataset_statistics: Optional[Data] = None,
#     ):

#         module = ResnetModule.create(**config["model"])
#         rng = rng if rng is not None else jax.random.PRNGKey(0)
#         example_batch = multihost_utils.process_allgather(example_batch)
#         example_batch = jax.tree_map(lambda x: x[:1], example_batch)

#         init_args = (
#             example_batch["observation"],
#             example_batch["task"],
#             example_batch["observation"]["timestep_pad_mask"],
#         )

#         if verbose:
#             print(
#                 module.tabulate(rng, *init_args, train=False, verbose=True, depth=2)
#             )  # Prints out the parameter count of our model, and tokenizer details

#         @jax.jit
#         def _init(rng):
#             return module.init(rng, *init_args, train=False)

#         variables = _init(rng)
#         params = variables["params"]
#         perturbations = variables.get('perturbations', None)

#         return cls(
#             module=module,
#             params=params,
#             perturbations=perturbations,
#             text_processor=text_processor,
#             example_batch=example_batch,
#             config=config,
#             dataset_statistics=dataset_statistics,
#         )


# def _verify_shapes(
#     pytree,
#     name: str,
#     example_pytree,
#     starting_dim: int = 0,
#     strict: bool = False,
#     raise_error: bool = True,
#     silent: bool = False,
# ):
#     weak_fail, fail = False, False
#     pytree_flat = flax.traverse_util.flatten_dict(pytree)
#     example_pytree_flat = flax.traverse_util.flatten_dict(example_pytree)

#     # Check that all elements are present
#     if set(pytree_flat.keys()) != set(example_pytree_flat.keys()):
#         if not silent:
#             extra = set(pytree_flat.keys()) - set(example_pytree_flat.keys())
#             if extra:
#                 logging.warning(
#                     "'%s' contains extra items compared to example_batch: %s",
#                     name,
#                     {"/".join(x) for x in extra},
#                 )
#             missing = set(example_pytree_flat.keys()) - set(pytree_flat.keys())
#             if missing:
#                 logging.warning(
#                     "'%s' is missing items compared to example_batch: %s",
#                     name,
#                     {"/".join(x) for x in missing},
#                 )
#         weak_fail = True

#     mismatched_keys = {
#         k: f"{pytree_flat[k].shape} != {example_pytree_flat[k].shape}"
#         for k in pytree_flat
#         if k in example_pytree_flat
#         and pytree_flat[k].shape[starting_dim:]
#         != example_pytree_flat[k].shape[starting_dim:]
#     }
#     if mismatched_keys:
#         if not silent:
#             logging.error(
#                 "'%s' contains mismatched shapes compared to example_batch: %s",
#                 name,
#                 flax.core.pretty_repr(
#                     {"/".join(k): v for k, v in mismatched_keys.items()}
#                 ),
#             )
#         fail = True

#     if raise_error and (fail or (weak_fail and strict)):
#         raise AssertionError(f"{name} does not match example batch.")

#     return weak_fail or fail


# SPEC_TEMPLATE = """
# This model is trained with a window size of {window_size}, predicting {action_dim} dimensional actions {action_horizon} steps into the future.
# Observations and tasks conform to the following spec:

# Observations: {observation_space}
# Tasks: {task_space}

# At inference, you may pass in any subset of these observation and task keys, with a history window up to {window_size} timesteps.
# """
