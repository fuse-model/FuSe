import importlib
import json
from typing import Any, Callable, Dict, Generic, Mapping, TypeVar

from flax import linen as nn
from flax import struct
from flax.core.frozen_dict import freeze, FrozenDict, unfreeze
import jax
import optax
import orbax.checkpoint as ocp
from palivla.types import Params
from palivla.utils import freeze_structure
from scalax.sharding import MeshShardingHelper, PartitionSpec
import tensorflow as tf

T = TypeVar("T")


@struct.dataclass
class CtorSpec(Generic[T]):
    ctor: Callable[..., T]
    config: FrozenDict[str, Any]
    load_fn: Callable[..., Params] | None = None
    load_kwargs: FrozenDict[str, Any] | None = None

    @classmethod
    def is_ctor_spec_dict(cls, data: Any) -> bool:
        return isinstance(data, Mapping) and "__ctor" in data and "config" in data

    @classmethod
    def create(
        cls,
        ctor: Callable[..., T] | str,
        config: Dict[str, Any],
        load_fn: Callable[..., Params] | None = None,
        load_kwargs: Dict[str, Any] | None = None,
    ) -> "CtorSpec[T]":
        config = jax.tree.map(
            lambda x: CtorSpec.from_dict(x) if CtorSpec.is_ctor_spec_dict(x) else x,
            config,
            is_leaf=CtorSpec.is_ctor_spec_dict,
        )
        config = freeze_structure(config)
        if load_kwargs is None:
            load_kwargs = {}
        load_kwargs = freeze_structure(load_kwargs)
        return cls(
            ctor=ctor,
            config=freeze(config),
            load_fn=load_fn,
            load_kwargs=freeze(load_kwargs),
        )

    @classmethod
    def from_name(cls, ctor_full_name: str, config: Dict[str, Any]):
        ctor_module = importlib.import_module(".".join(ctor_full_name.split(".")[:-1]))
        ctor_name = ctor_full_name.split(".")[-1]
        ctor = getattr(ctor_module, ctor_name)

        load_fn_str = config.pop("load_fn", None)
        load_kwargs = config.pop("load_kwargs", {})
        if load_fn_str:
            load_module = importlib.import_module(".".join(load_fn_str.split(".")[:-1]))
            load_fn_name = load_fn_str.split(".")[-1]
            load_fn = getattr(load_module, load_fn_name)
        else:
            load_fn = None

        return cls.create(ctor, config, load_fn, load_kwargs)

    def instantiate(self, **kwargs) -> T:
        return self.ctor(**self.config, **kwargs)

    def to_dict(self) -> Dict[str, Any]:
        config = jax.tree.map(
            lambda x: x.to_dict() if isinstance(x, CtorSpec) else x,
            unfreeze(self.config),
            is_leaf=lambda x: isinstance(x, CtorSpec),
        )
        return {
            "__ctor": self.ctor.__module__ + "." + self.ctor.__name__,
            "config": config,
        }

    @classmethod
    def from_dict(
        cls, data: Dict[str, Any], overrides: Dict[str, Any] | None = None
    ) -> "CtorSpec":
        if overrides:
            data["config"].update(overrides)

        return cls.from_name(ctor_full_name=data["__ctor"], config=data["config"])

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(
        cls, json_str: str, overrides: Dict[str, Any] | None = None
    ) -> "CtorSpec":
        data = json.loads(json_str)
        return cls.from_dict(data, overrides=overrides)


OptimizerSpec = CtorSpec[optax.GradientTransformation]
ModuleSpec = CtorSpec[nn.Module]


def restore_gluon_module(
    path: str,
    mesh: MeshShardingHelper,
    step: int | None = None,
    extra_kwargs: Dict[str, Any] = {},
):
    with tf.io.gfile.GFile(tf.io.gfile.join(path, "module_spec.json"), "r") as f:
        module_spec = ModuleSpec.from_json(f.read())

    module_spec.config.update(extra_kwargs)

    params_manager = ocp.CheckpointManager(
        directory=tf.io.gfile.join(path, "checkpoints"),
        item_handlers={"default": ocp.StandardCheckpointHandler()},
    )

    if step is None:
        step = params_manager.latest_step()

    params_metadata = params_manager.item_metadata(step)["default"]
    sharding = jax.sharding.NamedSharding(mesh, PartitionSpec())
    abstract_params = jax.tree.map(
        lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=sharding),
        params_metadata,
    )
    params = params_manager.restore(
        step,
        args=ocp.args.Composite(default=ocp.args.StandardRestore(abstract_params)),
    )["default"]["params"]
    return module_spec, params
