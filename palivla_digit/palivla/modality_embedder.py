import flax.linen as nn
import jax.numpy as jnp


# quick hack to make embedding w/ data type param json-serializable
class ModalityEmbedder(nn.Module):
    num_embeddings: int
    embedding_dim: int
    dtype_str: str = "float32"

    @nn.compact
    def __call__(self, x):
        return nn.Embed(
            num_embeddings=self.num_embeddings,
            features=self.embedding_dim,
            dtype=getattr(jnp, self.dtype_str),
        )(x)
