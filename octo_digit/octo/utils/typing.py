from typing import Any, Mapping, Sequence, Union

import jax

PRNGKey = jax.Array
PyTree = Union[jax.typing.ArrayLike, Mapping[str, "PyTree"]]
Config = Union[Any, Mapping[str, "Config"]]
Params = Mapping[str, PyTree]
Perturbations = Mapping[str, PyTree]
JaxArray = jax.typing.ArrayLike
Data = Mapping[str, PyTree]
Shape = Sequence[int]
Dtype = jax.typing.DTypeLike
