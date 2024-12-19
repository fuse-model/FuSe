import jax.config

jax.config.update("jax_default_prng_impl", "unsafe_rbg")
from functools import partial
from typing import Any, Dict

import flax
import jax
from jax import numpy as jnp
from jax.experimental import jax2tf
import numpy as np
import tensorflow as tf

from octo.model.octo_model import OctoModel

MODEL_NAME = ""
MODEL_PATH = ""

#################### HELPERS FOR SAVING ###############################

def model_jax(model, params, inputs, default_kwargs):
    return model.replace(params=params).sample_actions(
        **inputs,
        **default_kwargs,
    )


ExampleInputs = Any

def create_tf_model(
    model: OctoModel, possible_input_patterns: Dict[str, ExampleInputs], default_kwargs
):
    params_vars = tf.nest.map_structure(tf.Variable, model.params)
    pred_fn = partial(model_jax, model, default_kwargs=default_kwargs)
    prediction_tf = lambda inputs: jax2tf.convert(pred_fn)(params_vars, inputs)
    tf_model = tf.Module()
    # Tell the model saver what are the variables.
    tf_model._variables = tf.nest.flatten(params_vars)
    possible_specs = {
        k: jax.tree_map(
            lambda x: tf.TensorSpec(shape=x.shape, dtype=tf.as_dtype(x.dtype)), v
        )
        for k, v in possible_input_patterns.items()
    }

    for k, spec in possible_specs.items():
        print(
            "Adding function named",
            k,
            "with spec",
            flax.core.pretty_repr(spec),
            "to model",
        )
        setattr(
            tf_model,
            k,
            tf.function(
                prediction_tf, jit_compile=True, autograph=False, input_signature=[spec]
            ),
        )
    return tf_model


###################### MODEL SAVING #############################

model = OctoModel.load_pretrained(MODEL_PATH)


ws1_obs = {
    k: model.example_batch["observation"][k][:1, :1]
    for k in ["image_primary", "timestep_pad_mask"]
}
ws2_obs = {
    k: model.example_batch["observation"][k][:1, :2]
    for k in ["image_primary", "timestep_pad_mask"]
}

gc_task = {k: model.example_batch["task"][k][:1] for k in ["image_primary"]}
lc_task = {
    k: jax.tree_map(lambda x: x[:1], model.example_batch["task"][k])
    for k in ["language_instruction"]
}

example_rng = np.array(jax.random.PRNGKey(0))
print(example_rng.shape)
possible_input_patterns = {
    f"{task_type}_ws{ws}": {
        "observations": obs,
        "tasks": task,
        "rng": example_rng,
    }
    for (task_type, task) in [("gc", gc_task), ("lc", lc_task)]
    for (ws, obs) in [(1, ws1_obs), (2, ws2_obs)]
}

default_kwargs = dict(
    train=False,
    argmax=False,
    sample_shape=tuple(),
    temperature=1.0,
)

tf_model = create_tf_model(
    model, possible_input_patterns, default_kwargs=default_kwargs
)
tf.saved_model.save(tf_model, MODEL_NAME)


###################### EXAMPLE MODEL LOADING #############################

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("t5-base")
tokenizer_kwargs = {
    "max_length": 16,
    "padding": "max_length",
    "truncation": True,
    "return_tensors": "np",
}
instruction = "place coke can"
inputs = tokenizer(instruction, **tokenizer_kwargs)

restored_model = tf.saved_model.load(MODEL_NAME)
import time

while True:
    start = time.time()
    print(
        restored_model.lc_ws2(
            {
                "observations": {
                    "image_primary": np.zeros((1, 2, 256, 256, 3), dtype=np.uint8),
                    "timestep_pad_mask": np.ones(
                        (
                            1,
                            2,
                        ),
                        dtype=np.bool_
                    ),
                },
                "tasks": {
                    "language_instruction": inputs,
                },
                "rng": np.zeros((4,), dtype=np.uint32),
            }
        )
    )
    end = time.time()
    print(end - start)
