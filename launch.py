# bsaically best generate function ever, load ema weights, nice
from .models.dit import DiT
import jax.numpy as jnp
import click
import jax
from functools import partial
import cv2
import numpy as np  
from tqdm import tqdm
import glob
import pickle
from .utils.train_state import TrainState
from einops import rearrange
from moviepy.editor import ImageSequenceClip
import warnings
from .server import start_demo_thingy
import ml_collections
import flax
# supress logs warning
warnings.filterwarnings("ignore")

def interpolate(data, noise, t):
    t = t.reshape((*t.shape, *(1 for _ in range(len(data.shape) - len(t.shape)))))
    return (1 - t) * data + t * noise

from math import log10


VAE_SCALE = 0.7
TOTAL_OPTIMIZATION_STEPS = 2_000_000
ACTION_LABELS = ['VERTICAL', 'HORIZONTAL', 'JUMP_OR_SNEAK', 'SPRINT', 'DROP', 'YAW', 'PITCH', 'ATTACK', 'USE', "NOISY_FRAME", *([f"HOTBAR{i}" for i in range(9)])]
VAE_HEIGHT, VAE_WIDTH = 3, 5
LATENT_DIM = 256
CONTEXT_LENGTH = 20
SAMPLING_STEPS = 20
PAST_CONTEXT_NOISE_STEPS = 0.12
def create_model(patch_size, hidden_size, depth, num_heads, mlp_ratio, ctx_dropout_prob):
    return DiT(
        patch_size=patch_size, hidden_size=hidden_size, depth=depth,
        num_heads=num_heads, mlp_ratio=mlp_ratio, ctx_dropout_prob=ctx_dropout_prob,
        height=VAE_HEIGHT
    )


def load_ema():
    import pickle
    with open('./ema_params_numpy.tmp', 'rb') as f:
        model_chk = pickle.load(f)
        return model_chk

vae_model_config =  ml_collections.ConfigDict({
    'filters': 64,
    'num_res_blocks': 2,
    'channel_multipliers': (1, 2, 4, 4, 4, 8, 8, 8),
    'embedding_dim': 256,
    'norm_type': 'GN',
    'weight_decay': 0.05,
    'clip_gradient': 1.0,
    'l2_loss_weight': 1.0,
    'eps_update_rate': 0.9999,
    # Quantizer
    'quantizer_type': 'kl', # or 'fsq', 'kl'
    # Quantizer (VQ)
    'quantizer_loss_ratio': 1,
    'codebook_size': 1024,
    'entropy_loss_ratio': 0.1,
    'entropy_loss_type': 'softmax',
    'entropy_temperature': 0.01,
    'commitment_cost': 0.25,
    # Quantizer (FSQ)
    'fsq_levels': 5, # Bins per dimension.
    # Quantizer (KL)
    'kl_weight': 1e-3,
    'image_channels': 3,
    'image_size': 640
})


def setup_vae(is_encode=False):
    from .models.vqvae import VQVAE
    import tensorflow as tf
    import pickle
    with tf.io.gfile.GFile("./vae_params_numpy.tmp", 'rb') as f:
        vae_params = pickle.loads(f.read())
    vae_model = VQVAE(vae_model_config, True)
    if is_encode:
        return vae_model, vae_params
        
    @partial(jax.jit, backend="cuda")
    def decode_latents(latents):
        pixels = vae_model.apply({"params":vae_params}, latents, method=vae_model.decode)
        pixels = jnp.clip(pixels, 0, 1)
        return pixels
    return decode_latents

def quantize_to_bins(value, bin_size=0.2):
    if abs(value) < 0.01:
        return 0
    value += bin_size if value > 0 else -bin_size
    quantized_value = round(value / bin_size) * bin_size
    return quantized_value


def load_validation_pickle(file_path):
    with open(file_path, "rb") as f:
        latent_frames, actions = pickle.load(f)
    actions_embeddings = np.zeros((len(actions), len(ACTION_LABELS)))
    for i, action_dict in enumerate(actions):
        action_dict["YAW"] = quantize_to_bins(action_dict["YAW"])
        action_dict["PITCH"] = quantize_to_bins(action_dict["PITCH"])
        for actkey, actv in action_dict.items():
            if (actkey not in ACTION_LABELS) and (actkey != "HOTBAR"):
                continue

            if actkey == "HOTBAR":
                hot_bar = np.zeros(9)
                hot_bar[actv] = 1
                for _i in range(9):
                    actions_embeddings[i, ACTION_LABELS.index(f"HOTBAR{_i}")] = hot_bar[_i]
            else:
                act_idx = ACTION_LABELS.index(actkey)
                actions_embeddings[i, act_idx] = actv
    return latent_frames.astype(np.float32), actions_embeddings.astype(np.float32)

def get_validation_samples(MAP_ID):
    files = [f"./maps/{MAP_ID}.pickle"]
    latents, actions = [], []
    for file in files:
        latent_frames, action_frames = load_validation_pickle(file)
        latents.append(latent_frames)
        actions.append(action_frames)
    latents = np.stack(latents)
    actions = np.stack(actions)
    return latents, actions


@partial(jax.jit, static_argnums=(1, ))
def sample_step(model, steps, latents, noised_context, context_time_step, actions, rng):
    ds = 1.0 / steps
    first_loop=True
    kv_cache = None
    for i in range(steps, 0, -1):
        t = jnp.array([i / steps])
        rng, step_rng = jax.random.split(rng)
        
        model_input = jnp.concatenate([noised_context, latents], axis=1) # (B, T, H, W, D)
        
        _t = jnp.concatenate([context_time_step, jnp.ones((context_time_step.shape[0], 1)) * t], axis=1)

        if first_loop:
            model_output, kv_cache = model(model_input, _t, actions, train=False, rngs={'dropout': step_rng}, params=model.params, context_length=CONTEXT_LENGTH)
            first_loop = False
            kv_cache = rearrange(kv_cache, 'l j b (s h w) e d -> l j b s h w e d', h=3, w=5)
            kv_cache = kv_cache[..., :-1, :, :, :, :]
            kv_cache = rearrange(kv_cache, 'l j b s h w e d -> l j b (s h w) e d')
        else:
            model_output, _ = model(model_input[:, -1:], _t[:, -1:], actions[:, -1:], train=False, rngs={'dropout': step_rng}, params=model.params, context_length=1, inputs_kv=kv_cache)

        latents = latents - (model_output[:, -1:] * ds)
    return rng, latents



def diffusion_handle(decode_latents, SAMPLING_STEPS, PAST_CONTEXT_NOISE_STEPS, MAP_ID):
    context_frames, _actions = get_validation_samples(MAP_ID)
    context_frames, actions = context_frames[0, :CONTEXT_LENGTH], _actions[0, :CONTEXT_LENGTH]

    context_frames = jnp.array(context_frames)
    past_actions = jnp.array(actions)

    means, logvars = jnp.split(context_frames, 2, axis=-1)
    rng = jax.random.PRNGKey(0)
    eps = jax.random.normal(rng, means.shape)
    context_frames = (means + jnp.exp(logvars * 0.5) * eps) / VAE_SCALE
    
    prgbar = tqdm(desc=f"Sampling frames steps={SAMPLING_STEPS}, noise std: {PAST_CONTEXT_NOISE_STEPS}", position=1)

    def step(model,  action_took: np.ndarray):
        nonlocal context_frames, past_actions, rng
        
        latent_shape = (1, 1, VAE_HEIGHT, VAE_WIDTH, LATENT_DIM)
        latents = jax.random.normal(rng, latent_shape)
        
        context_time_step = jnp.ones((1, CONTEXT_LENGTH - 1)) * PAST_CONTEXT_NOISE_STEPS
        context_noise = jax.random.normal(rng, context_frames[1:].shape)
        noised_context = interpolate(context_frames[1:], context_noise, context_time_step[0])[None, ...]
        
        actions = jnp.concatenate([past_actions[1:], action_took[None, ...]], axis=0)[None, ...] # (B, T, 5)
        
        rng, latents = sample_step(model, SAMPLING_STEPS, latents, noised_context, context_time_step, actions, rng)
        
        new_frame_latent = latents[0, 0] * VAE_SCALE
        context_frames = jnp.concatenate([context_frames[1:], latents[0, :]], axis=0)
        past_actions = actions[0]

        pixels = decode_latents(new_frame_latent[None, ...])
        pixels = jnp.clip(pixels, 0, 1)
        pixels = (pixels[0] * 255)[:, :, ::-1].astype(jnp.uint8)
        
        prgbar.update(1)        

        pixels_np = np.array(pixels)
        pixels_np = pixels_np[:-24, :, :]
        # del, mem free
        del pixels
        del new_frame_latent
        del latents
        return pixels_np
    
    return step



@click.command()
@click.option("--patch_size", type=int, default=1)
@click.option("--hidden_size", type=int, default=2048)
@click.option("--depth", type=int, default=16)
@click.option("--num_heads", type=int, default=16)
@click.option("--mlp_ratio", type=float, default=4.0)
@click.option("--ctx_dropout_prob", type=float, default=0.1)
@click.option("--port", type=int, default=3000)
def main(patch_size, hidden_size, depth, num_heads, mlp_ratio, ctx_dropout_prob, port):
    model = create_model(patch_size, hidden_size, depth, num_heads, mlp_ratio, ctx_dropout_prob)
    ema_params = load_ema()
    state = TrainState.create(model, ema_params)
    state = jax.device_put(state)
    decode_latents = setup_vae()

    def diffusion_handler(_, MAP_ID):
        frames = []
        internal_handler = diffusion_handle(decode_latents, SAMPLING_STEPS, PAST_CONTEXT_NOISE_STEPS, MAP_ID)
        def end_save_video():
            nonlocal frames
            print("session ended")
        def step_handler(actions):
            try:
                frame= internal_handler(state, actions)
                frames.append(frame)
                return frame
            except Exception as e:
                print("Error in diffusion stream", e)
                import traceback
                traceback.print_exc()
                return get_fake_frame()
        return step_handler, end_save_video

    print("Hey there! welcome to the demo, wait for a bit, ignore the coming sampling messages as it's just a warmup")
    strwfl = start_demo_thingy(diffusion_handler, port)
    while True:
        strwfl.step(lambda *x: x)

if __name__ == "__main__":
    main()
