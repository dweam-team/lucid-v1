# DiT, (B, 3, 5, 256) or (T, 15, 256), we do diffusion transformer on this

import math
from typing import Any, Tuple
import flax.linen as nn
from flax.linen.initializers import xavier_uniform
import jax
from jax import lax
import jax.numpy as jnp
from einops import rearrange
import time
import tqdm
Array = Any
PRNGKey = Any
Shape = Tuple[int]
Dtype = Any

from typing import Any, Callable, Optional, Tuple, Type, Sequence, Union

# Port of https://github.com/facebookresearch/DiT/blob/main/models.py into jax.

#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    hidden_size: int
    frequency_embedding_size: int = 256
    max_period: int = 10000

    @nn.compact
    def __call__(self, t):
        B, T = t.shape
        t = t.reshape(-1)
        x = self.timestep_embedding(t, self.max_period)
        x = x.reshape((B, T, *x.shape[1:]))
        x = nn.Dense(self.hidden_size, kernel_init=nn.initializers.normal(0.02), dtype=jnp.bfloat16)(x)
        x = nn.silu(x)
        x = nn.Dense(self.hidden_size, kernel_init=nn.initializers.normal(0.02), dtype=jnp.bfloat16)(x)
        return x

    # t is between [0, max_period]. It's the INTEGER timestep, not the fractional (0,1).;
    def timestep_embedding(self, t, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        t = jax.lax.convert_element_type(t, jnp.float32)
        dim = self.frequency_embedding_size
        half = dim // 2
        freqs = jnp.exp( -math.log(max_period) * jnp.arange(start=0, stop=half, dtype=jnp.float32) / half)
        args = t[:, None] * freqs[None]
        embedding = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)
        return embedding
    
class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    dropout_prob: float
    num_classes: int
    hidden_size: int

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            rng = self.make_rng('label_dropout')
            drop_ids = jax.random.bernoulli(rng, self.dropout_prob, (labels.shape[0],))
        else:
            drop_ids = force_drop_ids == 1
        labels = jnp.where(drop_ids, self.num_classes, labels)
        return labels
    
    @nn.compact
    def __call__(self, labels, train, force_drop_ids=None):
        embedding_table = nn.Embed(self.num_classes + 1, self.hidden_size, embedding_init=nn.initializers.normal(0.02))
        embeddings = embedding_table(labels)
        return embeddings

class MlpBlock(nn.Module):
    """Transformer MLP / feed-forward block."""

    mlp_dim: int
    dtype: Dtype = jnp.bfloat16
    out_dim: Optional[int] = None
    dropout_rate: float = None
    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.xavier_uniform()
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.normal(stddev=1e-6)

    @nn.compact
    def __call__(self, inputs):
        """It's just an MLP, so the input shape is (batch, len, emb)."""
        actual_out_dim = inputs.shape[-1] if self.out_dim is None else self.out_dim
        x = nn.Dense(
                features=self.mlp_dim,
                dtype=self.dtype,
                kernel_init=self.kernel_init,
                bias_init=self.bias_init,
                )(inputs)
        x = nn.gelu(x)
        # x = nn.Dropout(rate=self.dropout_rate)(x)
        output = nn.Dense(
                features=actual_out_dim,
                dtype=self.dtype,
                kernel_init=self.kernel_init,
                bias_init=self.bias_init, 
                )(x)
        # output = nn.Dropout(rate=self.dropout_rate)(output)
        return output
    

def modulate(x, shift, scale):
    return x * (1 + scale[:]) + shift[:]


def create_causal_temporal_mask(
    context_length: int,
    num_patches: int,
    num_heads: int,
) -> jnp.ndarray:
    """
    Creates a causal temporal mask that allows attention within the same temporal frame
    and to all previous frames, while disallowing attention to future frames.

    Args:
        context_length (int): Number of temporal frames (T).
        num_patches (int): Number of patches per frame (N).
        num_heads (int): Number of attention heads.

    Returns:
        jnp.ndarray: (1, num_heads, T*N, T*N) mask tensor.
                    0.0 where attention is allowed,
                    -1e9 where attention is masked.
    """
    T = context_length
    N = num_patches
    total_tokens = T * N

    # Create frame indices for each token
    frame_indices = jnp.repeat(jnp.arange(T), N)  # Shape: (T*N,)

    # Create a mask matrix where tokens in current or previous frames are allowed
    # mask_matrix[i, j] = True if frame[j] <= frame[i], else False
    mask_matrix = frame_indices[None, :] <= frame_indices[:, None]  # Shape: (T*N, T*N)

    # Convert boolean mask to numerical mask: 0.0 for allowed, -1e9 for masked
    float_mask = jnp.where(mask_matrix, True, False)#.astype(jnp.float32)  # Shape: (T*N, T*N)

    # Expand dimensions to include attention heads
    # Shape: (1, 1, T*N, T*N)
    float_mask = float_mask[None, None, :, :]

    # Broadcast the mask across all attention heads
    # Shape: (1, num_heads, T*N, T*N)
    float_mask = jnp.broadcast_to(float_mask, (1, num_heads, T*N, T*N))

    return float_mask


import functools
import inspect
from typing import overload

class MultiHeadDotProductAttention(nn.MultiHeadDotProductAttention):
  @overload
  def __call__(
    self,
    inputs_q: Array,
    inputs_k: Optional[Array] = None,
    inputs_v: Optional[Array] = None,
    *,
    mask: Optional[Array] = None,
    deterministic: Optional[bool] = None,
    dropout_rng: Optional[PRNGKey] = None,
    sow_weights: bool = False,
  ):
    ...

  @overload
  def __call__(
    self,
    inputs_q: Array,
    *,
    inputs_kv: Optional[Array] = None,
    mask: Optional[Array] = None,
    deterministic: Optional[bool] = None,
    dropout_rng: Optional[PRNGKey] = None,
    sow_weights: bool = False,
  ):
    ...

  @nn.compact
  def __call__(
    self,
    inputs_q: Array,
    inputs_k: Optional[Array] = None,
    inputs_v: Optional[Array] = None,
    *,
    inputs_kv: Optional[Array] = None,
    mask: Optional[Array] = None,
    deterministic: Optional[bool] = None,
    dropout_rng: Optional[PRNGKey] = None,
    sow_weights: bool = False,
    kv_cache: Any = None
  ):

    if inputs_kv is not None:
      if inputs_k is not None or inputs_v is not None:
        raise ValueError("")
      inputs_k = inputs_v = inputs_kv
    else:
      if inputs_k is None:
        if inputs_v is not None:
          raise ValueError("")
        inputs_k = inputs_q
      if inputs_v is None:
        inputs_v = inputs_k


    features = self.out_features or inputs_q.shape[-1]
    qkv_features = self.qkv_features or inputs_q.shape[-1]
    assert qkv_features % self.num_heads == 0, (
      f'Memory dimension ({qkv_features}) must be divisible by number of'
      f' heads ({self.num_heads}).'
    )
    head_dim = qkv_features // self.num_heads

    dense = functools.partial(
      nn.DenseGeneral,
      axis=-1,
      dtype=self.dtype,
      param_dtype=self.param_dtype,
      features=(self.num_heads, head_dim),
      kernel_init=self.kernel_init,
      bias_init=self.bias_init,
      use_bias=self.use_bias,
      precision=self.precision,
      dot_general=self.qkv_dot_general,
      dot_general_cls=self.qkv_dot_general_cls,
    )

    query, key, value = (
      dense(name='query')(inputs_q),
      dense(name='key')(inputs_k),
      dense(name='value')(inputs_v),
    )

    if kv_cache is not None:
        key_cache, value_cache = kv_cache[0], kv_cache[1] # same properties
        key = jnp.concatenate([key_cache, key], axis=1)
        value = jnp.concatenate([value_cache, value], axis=1)


    if (
      self.dropout_rate > 0.0
    ):  # Require `deterministic` only if using dropout.

      m_deterministic = nn.merge_param(
        'deterministic', self.deterministic, deterministic
      )
      if not m_deterministic and dropout_rng is None:
        dropout_rng = self.make_rng('dropout')
    else:
      m_deterministic = True

    # apply attention
    attn_args = (query, key, value)
    # This kwargs list match the default nn.dot_product_attention.
    # For custom `attention_fn`s, invalid kwargs will be filtered.
    attn_kwargs = dict(
      mask=mask,
      dropout_rng=dropout_rng,
      dropout_rate=self.dropout_rate,
      broadcast_dropout=self.broadcast_dropout,
      deterministic=m_deterministic,
      dtype=self.dtype,
      precision=self.precision,
      force_fp32_for_softmax=self.force_fp32_for_softmax,
    )
    attn_kwargs = {
        k: v
        for k, v in attn_kwargs.items()
        if k in inspect.signature(self.attention_fn).parameters
    }
    if sow_weights:
      x = self.attention_fn(*attn_args, **attn_kwargs, module=self)
    else:
      x = self.attention_fn(*attn_args, **attn_kwargs)
    # back to the original inputs dimensions
    out = nn.DenseGeneral(
      features=features,
      axis=(-2, -1),
      kernel_init=self.out_kernel_init or self.kernel_init,
      bias_init=self.out_bias_init or self.bias_init,
      use_bias=self.use_bias,
      dtype=self.dtype,
      param_dtype=self.param_dtype,
      precision=self.precision,
      dot_general=self.out_dot_general,
      dot_general_cls=self.out_dot_general_cls,
      name='out',  # type: ignore[call-arg]
    )(x)
    return out, (key, value)




class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    hidden_size: int
    num_heads: int
    mlp_ratio: float = 4.0
    token_per_frame: int = None
    sequence_length: int = None

    @nn.compact
    def __call__(self, x, c, inputs_kv):
        
        # Calculate adaLn modulation parameters.
        c = nn.silu(c) # (B, T, E)
        c = nn.Dense(6 * self.hidden_size, kernel_init=nn.initializers.constant(0.), dtype=jnp.bfloat16)(c) # (B, T*H*W, 6*E)
        c = c[:, :, None, :] # (B, T, 1, 6*E)
        c = jnp.repeat(c, self.token_per_frame, axis=2) # (B, T, N, 6*E)
        c = rearrange(c, 'B T N C -> B (T N) C') # (B, T*N, 6*E)

        # c = jnp.repeat(c, self.token_per_frame, axis=1) # temporray
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = jnp.split(c, 6, axis=-1)
        
        # Attention Residual.
        x_norm = nn.LayerNorm(use_bias=False, use_scale=False, dtype=jnp.bfloat16)(x)
        x_modulated = modulate(x_norm, shift_msa, scale_msa)

        mask = create_causal_temporal_mask(self.sequence_length, self.token_per_frame, self.num_heads)
        attn_x, (key_past, value_past) = MultiHeadDotProductAttention(kernel_init=nn.initializers.xavier_uniform(),
            num_heads=self.num_heads,
            dtype=jnp.bfloat16,
            )(x_modulated, x_modulated, kv_cache=inputs_kv, mask=mask if inputs_kv is None else None)
        next_kv_cache = jnp.stack([key_past, value_past], axis=0)
        x = x + (gate_msa[:, :] * attn_x)

        # MLP Residual.
        x_norm2 = nn.LayerNorm(use_bias=False, use_scale=False, dtype=jnp.bfloat16)(x)
        x_modulated2 = modulate(x_norm2, shift_mlp, scale_mlp)
        mlp_x = MlpBlock(mlp_dim=int(self.hidden_size * self.mlp_ratio))(x_modulated2)
        x = x + (gate_mlp[:, :] * mlp_x)


        return x, next_kv_cache
    
class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    patch_size: int
    out_channels: int
    hidden_size: int
    context_length: int = 64
    token_per_frame: int = None
    height: int = 3
    width: int = 5

    @nn.compact
    def __call__(self, x, c):
        B, N, E = x.shape
        c = nn.silu(c)
        c = nn.Dense(2 * self.hidden_size, kernel_init=nn.initializers.constant(0), dtype=jnp.bfloat16)(c)
        c = c[:, :, None, :] # (B, T, 1, 6*E)
        c = jnp.repeat(c, self.token_per_frame, axis=2) # (B, T, N, 6*E)
        c = rearrange(c, 'B T N C -> B (T N) C') # (B, T*N, 6*E)

        shift, scale = jnp.split(c, 2, axis=-1)
        x = modulate(nn.LayerNorm(use_bias=False, use_scale=False, dtype=jnp.bfloat16)(x), shift, scale)
        x = nn.Dense(256, 
                     kernel_init=nn.initializers.constant(0), dtype=jnp.bfloat16)(x)
        x = rearrange(x, 'B (T H W) C -> B T H W C', T=self.context_length, H=self.height, W=self.width, C=256)
        return x

class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding """
    patch_size: int
    embed_dim: int
    bias: bool = True

    @nn.compact
    def __call__(self, x):
        B, T, H, W, C = x.shape
        x = nn.Dense(self.embed_dim, kernel_init=nn.initializers.xavier_uniform(), use_bias=self.bias, dtype=jnp.bfloat16)(x) # (B, T, H, W, E)
        return x#, T*H*W

from jax import tree_util

def log_param_details(params):
    def size_and_bytes(x):
        return x.size, x.nbytes

    sizes_and_bytes = tree_util.tree_map(size_and_bytes, params)
    total_params = sum(tree_util.tree_leaves(tree_util.tree_map(lambda x: x[0], sizes_and_bytes)))
    total_bytes = sum(tree_util.tree_leaves(tree_util.tree_map(lambda x: x[1], sizes_and_bytes)))

    print(f"Total parameters: {total_params:,}")
    print(f"Total memory: {total_bytes / (1024 * 1024):.2f} MB")
    print("\nParameter details:")
    
    def print_details(path, value):
        size, bytes = value
        print(f"  {'/'.join(path)}:")
        print(f"    Shape: {params[path].shape}")
        print(f"    Size: {size:,}")
        print(f"    Memory: {bytes / 1024:.2f} KB")

    tree_util.tree_map_with_path(print_details, sizes_and_bytes)


# JAX implementation
def get_emb_jax(sin_inp):
    """
    Gets a base embedding for one dimension with sin and cos intertwined
    """
    emb = jnp.stack((jnp.sin(sin_inp), jnp.cos(sin_inp)), axis=-1)
    return jnp.reshape(emb, (*emb.shape[:-2], -1))

class PositionalEncoding3D(nn.Module):
    in_channels: int

    def setup(self):
        self.org_channels = self.in_channels
        channels = 684
        # channels = (jnp.ceil(self.in_channels / 6) * 2).astype(int)
        # channels += jnp.mod(channels, 2)
        # channels = jax.lax.cond(
        #     channels % 2 == 1,
        #     lambda x: x + 1,
        #     lambda x: x,
        #     channels
        # )
        self.channels = channels
        inv_freq = 1.0 / (10000 ** (jnp.arange(0, channels, 2) / channels))
        self.inv_freq = inv_freq

    @nn.compact
    def __call__(self, tensor, time_offset=0):
        """
        :param tensor: A 5d tensor of size (batch_size, x, y, z, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, y, z, ch)
        """
        if len(tensor.shape) != 5:
            raise RuntimeError("The input tensor has to be 5d!")

        batch_size, x, y, z, orig_ch = tensor.shape
        pos_x = jnp.arange(x, dtype=tensor.dtype) + time_offset
        pos_y = jnp.arange(y, dtype=tensor.dtype)
        pos_z = jnp.arange(z, dtype=tensor.dtype)

        sin_inp_x = jnp.einsum("i,j->ij", pos_x, self.inv_freq)
        sin_inp_y = jnp.einsum("i,j->ij", pos_y, self.inv_freq)
        sin_inp_z = jnp.einsum("i,j->ij", pos_z, self.inv_freq)

        emb_x = jnp.expand_dims(jnp.expand_dims(get_emb_jax(sin_inp_x), 1), 1)
        emb_y = jnp.expand_dims(get_emb_jax(sin_inp_y), 1)
        emb_z = get_emb_jax(sin_inp_z)

        emb = jnp.zeros((x, y, z, self.channels * 3), dtype=tensor.dtype)
        emb = emb.at[:, :, :, :self.channels].set(emb_x)
        emb = emb.at[:, :, :, self.channels:2*self.channels].set(emb_y)
        emb = emb.at[:, :, :, 2*self.channels:].set(emb_z)

        return jnp.repeat(emb[None, :, :, :, :orig_ch], batch_size, axis=0)

def discretize(value):
    bins = jnp.array([0.0, 0.2, 0.4, 0.6, 0.8])
    return jnp.digitize(value, bins) - 1
import numpy
@jax.jit
def label_embed(actions):
    weight = numpy.load("/home/ubuntu/jax_version/down_emb.npy") # (32, 11)
    # convert actions, which is (B, 11) to (B, 32)
    actions = jnp.dot(actions, weight.T)
    return actions

class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    patch_size: int
    hidden_size: int
    depth: int
    num_heads: int
    mlp_ratio: float
    ctx_dropout_prob: float
    height: int = 3
    width: int = 5
    @nn.compact
    def __call__(self, x, t, act, train=False, inputs_kv=None, context_length=64):
        in_channels = x.shape[-1]   
        
        x = PatchEmbed(self.patch_size, self.hidden_size)(x)
        token_per_frame = x.shape[2] * x.shape[3]  
        x= x + PositionalEncoding3D(self.hidden_size)(x, time_offset= 0 if inputs_kv is None else ((inputs_kv.shape[3]//token_per_frame)))
        x = rearrange(x, 'b t h w e -> b (t h w) e') 
   
        t = TimestepEmbedder(1024)(t[:, :]) 
        act = nn.Dense(1024, name="act_embed_d9", dtype=jnp.bfloat16)(act) 
        
        c = t + act 
        kvs = []
        for i in range(self.depth):
            x, kv = DiTBlock(self.hidden_size, self.num_heads, self.mlp_ratio, token_per_frame=token_per_frame, sequence_length=context_length)(x, c, inputs_kv=inputs_kv[i] if inputs_kv is not None else None)
            kvs.append(kv)
        x = FinalLayer(self.patch_size, in_channels, self.hidden_size, context_length=context_length, token_per_frame=token_per_frame, height=self.height, width=self.width)(x, c) # (B, num_patches, p*p*c)
        kvs = jnp.stack(kvs, axis=0)

        return x, kvs



def test_dit():
    rng = jax.random.PRNGKey(0)
    x = jax.random.normal(rng, (1, 16, 3, 5, 256))
    t = jax.random.randint(rng, (1, 16), 0, 1000)
    a = jax.random.normal(rng, (1, 16, 11))

    model = DiT(patch_size=1, hidden_size=2048, depth=16, num_heads=16, mlp_ratio=2.0, ctx_dropout_prob=0.1)

    params = model.init(rng, x, t, a, train=True, context_length=16)
    params = jax.tree_map(lambda x: x.astype(jnp.bfloat16), params)
    number_of_params = sum(p.size for p in jax.tree_flatten(params)[0])
    print ("Number of params", number_of_params)

    jitted_apply = jax.jit(model.apply, static_argnames=("context_length",))
    x = x.astype(jnp.bfloat16)
    t = t.astype(jnp.bfloat16)
    a = a.astype(jnp.bfloat16)
    y = jitted_apply(params, x, t, a, train=True, context_length=16)

    from diffusers import FlaxDDIMScheduler, FlaxDPMSolverMultistepScheduler
    scheduler = FlaxDDIMScheduler(prediction_type="v_prediction", clip_sample=False)
    scheduler_state = scheduler.create_state()
    scheduler_state = scheduler.set_timesteps(scheduler_state, 50)

    @jax.jit
    def time_step(x, N, scheduler_state):
        latents = x
        first_loop = True
        kv_cache = None
        for t in (scheduler_state.timesteps):
            _t = jnp.repeat(t[None, None,...], x.shape[0], axis=0)
            _t = jnp.repeat(_t, x.shape[1], axis=1)
            if first_loop:
                pred, kv_cache = jitted_apply(params, latents[:, :], _t[:, :], N[:, :], train=True, context_length=16)
                first_loop = False
            else:
                pkv= rearrange(kv_cache, 'l j b (s h w) e d -> l j b s h w e d', h=3, w=5)
                pkv = pkv[..., :-1, :, :, :, :]
                pkv = rearrange(pkv, 'l j b s h w e d -> l j b (s h w) e d')
                pred, _ = jitted_apply(params, latents[:, -1:], _t[:, -1:], N[:, -1:], train=True, context_length=1, inputs_kv=pkv)

            latents, scheduler_state = scheduler.step(scheduler_state, pred, t, latents).to_tuple()
        return latents


    for _ in tqdm.trange(100000):
        N = jax.random.normal(rng, (1, 16, 11))
        _ = time_step(x, N, scheduler_state)

if __name__ == "__main__":
    test_dit()
