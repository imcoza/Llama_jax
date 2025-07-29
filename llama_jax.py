# Configure JAX to use GPU and prevent memory preallocation
import os 
import jax
import jax.numpy as jnp
import math
from jax import random, vmap
import tiktoken
from functools import partial
import os
import jax.lax as lax
import pickle


from numpy import dtype

def rms_norm(x, weight, eps = 1e-5):
    variance = jnp.mean(jnp.square(x), axis = -1, keepdims = True)
    return x * weight * jnp.reciprocal(jnp.sqrt(variance + eps))

def precompute_freq_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (jnp.arange(0, dim // 2, dtype=jnp.float32) / dim))
    t = jnp.arange(end, dtype=jnp.float32)  # positions
    freqs = jnp.outer(t, freqs)  # shape: [seq_len, dim // 2]
    return jnp.complex64(jnp.exp(1j * freqs))  # shape: [seq_len, dim // 2]

def apply_rotary_emb(xq, xk, freqs_cis):
    xq_r, xk_r = jnp.reshape(xq, (*xq.shape[:-1], -1, 2)), jnp.reshape(xk, (*xk.shape[:-1], -1, 2))
    xq_complex = jnp.complex64(xq_r[..., 0] + 1j * xq_r[..., 1])
    xk_complex = jnp.complex64(xk_r[..., 0] + 1j * xk_r[..., 1])
    freqs_cis = jnp.reshape(freqs_cis, (1, freqs_cis.shape[0], 1, freqs_cis.shape[1]))
    xq_out = xq_complex * freqs_cis
    xk_out = xk_complex * freqs_cis
    xq = jnp.stack([jnp.real(xq_out), jnp.imag(xq_out)], axis=-1).reshape(xq.shape)
    xk = jnp.stack([jnp.real(xk_out), jnp.imag(xk_out)], axis=-1).reshape(xk.shape)
    return xq, xk

def repeat_kv(x,n_reps):
    return x if n_reps == 1 else jnp.repeat(x, n_reps, axis = 2)

#model weight initialization
def init_weight(key, shape, scale= None):
    scale = 1.0 / math.sqrt(shape[0]) if scale is None else scale
    return jax. random.normal(key, shape) *  scale

def init_attention_weights(key, dims,n_heads,n_kv_heads):
    keys = jax.random.split(key, 4)
    head_dim = dims // n_heads
    return{
        'wq': init_weight(keys[0], (dims, n_heads, head_dim)),
        'wk':init_weight(keys[1], (dims, n_kv_heads, head_dim)),
        'wv':init_weight(keys[2], (n_heads, dims, head_dim)),
        'wo':init_weight(keys[3], (n_heads, head_dim, dims)),
    }

# FFN with 3 trainable parameters
def init_ffn_weighst(key, dim):
    keys = jax.random.split(key, 3)
    return{
        'w1': init_weight(keys[0], (dim, 4 * dim)),
        'w2': init_weight(keys[1], (4 * dim, dim)),
        'w3': init_weight(keys[2], (dim, 4 * dim)),
    }
# Combining weights to transformer block
def init_transformer_block(key, dims, n_heads, n_kv_heads):
    keys = jax.random.split(key, 4)
    return{
        'attn': init_attention_weights(keys[0], dims, n_heads, n_kv_heads),
        'ffn': init_ffn_weighst(keys[1], dims),
        'attention_norm': init_weight(keys[2], (dims,), scale = 1.0),
        'ffn_norm': init_weight(keys[3], (dims,), scale = 1.0)
    }

def init_model_params(key, vocab_size,dim, n_heads,n_layers, n_kv_heads):
    key = jax.random.split(key,4)
    params = {
        'token_embedding': init_weight(key[0], (vocab_size, dim)),
        'norm_f': init_weight(key[1], (dim,), scale = 1.0),
        'output': init_weight(key[2], (dim, vocab_size)),
    }
    block_keys = jax.random.split(key[3], n_layers)
    params['blocks']= [init_transformer_block(block_key, dim, n_heads, n_kv_heads) for block_key in block_keys]
    return params

def attention(params,x, mask, freqs_cis,n_heads,n_kv_heads,cache = None,position =0):
    B,T,C = x.shape
    head_dim = C // n_heads
    
    # Reshape weights for proper matrix multiplication
    wq = params['wq'].reshape(C, n_heads * head_dim)
    wk = params['wk'].reshape(C, n_kv_heads * head_dim)
    wo = params['wo'].reshape(n_heads * head_dim, C)
    
    q = jnp.dot(x, wq).reshape(B,T,n_heads,head_dim)
    k = jnp.dot(x, wk).reshape(B,T,n_kv_heads,head_dim)
    
    # Handle wv differently - it's (n_heads, C, head_dim)
    wv = params['wv']  # Shape: (n_heads, C, head_dim)
    v = jnp.einsum('btd,ndh->btnh', x, wv)  # Shape: (B, T, n_heads, head_dim)
    
    q,k = apply_rotary_emb(q,k,freqs_cis[position:position+T])
    if cache is not None:
        k = jnp.concatenate([cache[0],k],axis = 1)
        v = jnp.concatenate([cache[1],v],axis = 1)
    new_cache = (k,v)
    k = repeat_kv(k,n_heads // n_kv_heads)
    v = repeat_kv(v,n_heads // n_kv_heads)
    q,k,v = map(lambda x: x.transpose(0,2,1,3), (q,k,v))
    scores = jnp.matmul(q,k.transpose(0,1,3,2)) /math.sqrt(head_dim)
    if mask is not None:
        scores = scores +mask[:,:,:T,:T]
    scores = jax.nn.softmax(scores, axis = -1)
    output = jnp.matmul(scores,v)
    output = output.transpose(0,2,1,3).reshape(B,T,-1)
    return jnp.dot(output, wo) , new_cache

def feed_forward(params,x):
    return jnp.dot(jax.nn.silu(jnp.dot(x, params['w3'])) * jnp.dot(x,params['w1']), params['w2'])

def transformer_block(params, x, mask, freqs_cis, n_heads,n_kv_heads,cache = None,position =0,training = False, dropout_rate = 0.0,key = None):
    attn_output, new_cache = attention(params['attn'],rms_norm(x,params['attention_norm']),mask,freqs_cis,n_heads,n_kv_heads,cache,position)
    if training:
        dropout_key , key = jax. random.split(key)
        attn_output = jax.random.bernoulli(dropout_key, 1-dropout_rate, shape = attn_output.shape) * attn_output / (1- dropout_rate)
    h = attn_output + x
    ff_output = feed_forward(params['ffn'],rms_norm(h,params['ffn_norm']))
    if training:
        dropout_key , key = jax. random.split(key)
        ff_output = jax.random.bernoulli(dropout_key, 1-dropout_rate, shape = ff_output.shape) * ff_output / (1- dropout_rate)
    out = ff_output + h
    return out, new_cache

def model_forward(params, inputs, config, cache=None, position=0):
    B, T = inputs.shape
    h = params['token_embedding'][inputs]
    freqs_cis = precompute_freq_cis(config.dim // config.n_heads, config.max_seq_len)
    mask = jnp.tril(jnp.ones((config.max_seq_len, config.max_seq_len)))
    mask = jnp.where(mask == 0, -1e9, 0.0)
    mask = mask.astype(h.dtype)
    mask = mask[None, None, :, :]
    new_caches = []
    for i, block in enumerate(params['blocks']):
        layer_cache = cache[i] if cache is not None else None
        h, layer_cache = transformer_block(block, h, mask, freqs_cis, config.n_heads, config.n_kv_heads, layer_cache, position, training=False, dropout_rate=config.dropout_rate)
        new_caches.append(layer_cache)
    h = rms_norm(h, params['norm_f'])
    logits = jnp.dot(h, params['output'])
    return logits, new_caches

os.environ['JAX_PLATFORM_NAME'] = 'gpu'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
print("JAX devices:", jax.devices())


# Initialize tokenizer and load data
enc = tiktoken.get_encoding("gpt2")
with open('shakespeare.txt', 'r') as f:
    text = f.read()
tokens = enc.encode(text)
data = jnp.array(tokens)

# Model configuration
class ModelConfig:
    vocab_size = enc.n_vocab
    dim = 256
    n_layers = 6
    n_heads = 8
    n_kv_heads = 4
    max_seq_len = 512
    batch_size = 32
    learning_rate = 3e-4
    dropout_rate = 0.0

config = ModelConfig()

# Initialize model
key = random.PRNGKey(0)
params = init_model_params(
    key=key,
    vocab_size=config.vocab_size,
    dim=config.dim,
    n_layers=config.n_layers,
    n_heads=config.n_heads,
    n_kv_heads=config.n_kv_heads
)
def save_params(params, filepath):
    numpy_params = jax.tree.map(lambda x: x.copy(), params)
    with open(filepath, 'wb') as f:
        pickle.dump(numpy_params, f)

def load_params(filepath):
    with open(filepath, 'rb') as f:
        numpy_params = pickle.load(f)
    # convert back to JAX arrays
    params = jax.tree.map(lambda x: jnp.array(x), numpy_params)
    return params

def get_batch(key, data, batch_size, seq_len):
    # Generate random starting indices
    ix = random.randint(key, (batch_size,), 0, len(data) - seq_len)

    # Vectorized operation to get input and target sequences
    x = vmap(lambda i: lax.dynamic_slice(data, (i,), (seq_len,)))(ix)
    y = vmap(lambda i: lax.dynamic_slice(data, (i + 1,), (seq_len,)))(ix)

    return x, y


def generate(params, prompt_tokens, max_new_tokens, config):
    x = jnp.array(prompt_tokens)
    for _ in range(max_new_tokens):
        x_crop = x[-config.max_seq_len:]
        logits, _ = model_forward(params, x_crop[None, :], config)
        logits = logits[0, -1, :]  # take the last logit
        next_token = random.categorical(random.PRNGKey(0), logits, shape=(1,))[0]
        x = jnp.concatenate([x, jnp.array([next_token])])
    return x.tolist()

def compute_loss(params, batch):
    inputs, targets = batch
    logits, _ = model_forward(params, inputs, config)
    logits = logits.reshape(-1, config.vocab_size)
    targets = targets.reshape(-1)
    loss = -jnp.mean(
        jnp.take_along_axis(
            jax.nn.log_softmax(logits),
            targets[:, None],
            axis=1
        )
    )
    return loss

@jax.jit
def update_step(params, batch):
    loss, grads = jax.value_and_grad(compute_loss)(params, batch)
    params = jax.tree.map(
        lambda p, g: p - config.learning_rate * g,
        params,
        grads
    )
    return params, loss

def train(num_epochs=30, steps_per_epoch=1000):
    key = random.PRNGKey(0)
    params_state = params  # copying

    epoch_losses = []

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 50)

        epoch_loss = 0.0
        for step in range(steps_per_epoch):

            key, batch_key = random.split(key)

            # Get batch
            batch = get_batch(batch_key, data, config.batch_size, config.max_seq_len)

            # Update model
            params_state, loss = update_step(params_state, batch)
            epoch_loss += loss


            if step % 100 == 0:
                print(f"epoch {epoch + 1}, step {step}/{steps_per_epoch}: loss = {loss:.4f}")


        avg_epoch_loss = epoch_loss / steps_per_epoch
        epoch_losses.append(avg_epoch_loss)

        print(f"\nepoch {epoch + 1} | average loss: {avg_epoch_loss:.4f}")


        if (epoch + 1) % 5 == 0:
            save_params(params_state, f'model_checkpoint_epoch_{epoch+1}.pkl')


    print("Loss by epoch:")
    for epoch, loss in enumerate(epoch_losses, 1):
        print(f"Epoch {epoch}: {loss:.4f}")

    # Save final model
    save_params(params_state, 'model_final.pkl')
    return params_state

# Train the model
trained_params = train()