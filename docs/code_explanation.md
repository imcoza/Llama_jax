# Llama JAX Implementation - Code Block Explanation

## Overview
This implementation is a complete Llama transformer model using JAX. It includes all the key components: RMS normalization, rotary positional embeddings, multi-head attention, and feed-forward networks.

---

## 1. Imports and Configuration

```python
import os 
import jax
import jax.numpy as jnp
import math
from jax import random, vmap
import tiktoken
from functools import partial
import jax.lax as lax
import pickle
```

**Purpose**: Import necessary libraries for JAX operations, tokenization, and mathematical functions.

**Example Usage**:
```python
# JAX arrays work like NumPy but are immutable and support automatic differentiation
x = jnp.array([1, 2, 3, 4])
y = jnp.sin(x)  # Element-wise sine
```

---

## 2. RMS Normalization

```python
def rms_norm(x, weight, eps = 1e-5):
    variance = jnp.mean(jnp.square(x), axis = -1, keepdims = True)
    return x * weight * jnp.reciprocal(jnp.sqrt(variance + eps))
```

**Purpose**: Implements RMS (Root Mean Square) normalization, which is more stable than LayerNorm.

**How it works**:
1. Calculate variance across the last dimension
2. Normalize by the reciprocal of the square root of variance
3. Scale by learnable weight parameter

**Example**:
```python
# Input: (batch_size, seq_len, hidden_dim)
x = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # Shape: (2, 3)
weight = jnp.array([1.0, 1.0, 1.0])  # Shape: (3,)

# Calculate variance: mean of squares
variance = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
# variance = [[4.67], [20.67]]

# Normalize
normalized = x * weight * jnp.reciprocal(jnp.sqrt(variance + 1e-5))
```

---

## 3. Rotary Positional Embeddings (RoPE)

### Precompute Frequencies
```python
def precompute_freq_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (jnp.arange(0, dim // 2, dtype=jnp.float32) / dim))
    t = jnp.arange(end, dtype=jnp.float32)  # positions
    freqs = jnp.outer(t, freqs)  # shape: [seq_len, dim // 2]
    return jnp.complex64(jnp.exp(1j * freqs))  # shape: [seq_len, dim // 2]
```

**Purpose**: Precompute complex frequency embeddings for rotary positional encoding.

**Example**:
```python
dim = 8  # hidden dimension
end = 10  # sequence length
freqs_cis = precompute_freq_cis(dim, end)
# Shape: (10, 4) - complex numbers for each position and frequency
```

### Apply Rotary Embeddings
```python
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
```

**Purpose**: Apply rotary positional embeddings to query and key tensors.

**Example**:
```python
# Input tensors: (batch, seq_len, n_heads, head_dim)
xq = jnp.random.normal(jax.random.PRNGKey(0), (2, 5, 4, 8))  # 2 batches, 5 seq, 4 heads, 8 dim
xk = jnp.random.normal(jax.random.PRNGKey(1), (2, 5, 4, 8))
freqs_cis = precompute_freq_cis(8, 5)

# Apply rotary embeddings
xq_rot, xk_rot = apply_rotary_emb(xq, xk, freqs_cis)
# Output: Same shape as input but with positional information encoded
```

---

## 4. Utility Functions

### Repeat KV Heads
```python
def repeat_kv(x, n_reps):
    return x if n_reps == 1 else jnp.repeat(x, n_reps, axis=2)
```

**Purpose**: Repeat key/value heads when using grouped-query attention (GQA).

**Example**:
```python
# If n_heads=8, n_kv_heads=2, then n_reps=4
x = jnp.array([[[1, 2], [3, 4]]])  # Shape: (1, 1, 2, 2)
repeated = repeat_kv(x, 4)  # Shape: (1, 1, 8, 2) - repeated 4 times
```

---

## 5. Weight Initialization

### Basic Weight Initialization
```python
def init_weight(key, shape, scale=None):
    scale = 1.0 / math.sqrt(shape[0]) if scale is None else scale
    return jax.random.normal(key, shape) * scale
```

**Purpose**: Initialize weights with proper scaling (Xavier initialization).

**Example**:
```python
key = jax.random.PRNGKey(42)
weight = init_weight(key, (512, 256))
# Shape: (512, 256), scaled by 1/sqrt(512)
```

### Attention Weights
```python
def init_attention_weights(key, dims, n_heads, n_kv_heads):
    keys = jax.random.split(key, 4)
    head_dim = dims // n_heads
    return {
        'wq': init_weight(keys[0], (dims, n_heads, head_dim)),
        'wk': init_weight(keys[1], (dims, n_kv_heads, head_dim)),
        'wv': init_weight(keys[2], (n_heads, dims, head_dim)),
        'wo': init_weight(keys[3], (n_heads, head_dim, dims)),
    }
```

**Purpose**: Initialize attention weights for query, key, value, and output projections.

**Example**:
```python
key = jax.random.PRNGKey(42)
attn_weights = init_attention_weights(key, dims=512, n_heads=8, n_kv_heads=8)
# Returns dict with:
# wq: (512, 8, 64) - query projection
# wk: (512, 8, 64) - key projection  
# wv: (8, 512, 64) - value projection
# wo: (8, 64, 512) - output projection
```

### Feed-Forward Weights
```python
def init_ffn_weighst(key, dim):
    keys = jax.random.split(key, 3)
    return {
        'w1': init_weight(keys[0], (dim, 4 * dim)),
        'w2': init_weight(keys[1], (4 * dim, dim)),
        'w3': init_weight(keys[2], (dim, 4 * dim)),
    }
```

**Purpose**: Initialize SwiGLU feed-forward network weights.

**Example**:
```python
key = jax.random.PRNGKey(42)
ffn_weights = init_ffn_weighst(key, dim=512)
# Returns dict with:
# w1: (512, 2048) - first linear layer
# w2: (2048, 512) - second linear layer  
# w3: (512, 2048) - gate linear layer (for SwiGLU)
```

---

## 6. Model Architecture

### Transformer Block
```python
def init_transformer_block(key, dims, n_heads, n_kv_heads):
    keys = jax.random.split(key, 4)
    return {
        'attn': init_attention_weights(keys[0], dims, n_heads, n_kv_heads),
        'ffn': init_ffn_weighst(keys[1], dims),
        'attention_norm': init_weight(keys[2], (dims,), scale=1.0),
        'ffn_norm': init_weight(keys[3], (dims,), scale=1.0)
    }
```

**Purpose**: Initialize a complete transformer block with attention and feed-forward components.

**Example**:
```python
key = jax.random.PRNGKey(42)
block = init_transformer_block(key, dims=512, n_heads=8, n_kv_heads=8)
# Returns dict with attention weights, FFN weights, and normalization weights
```

### Full Model Parameters
```python
def init_model_params(key, vocab_size, dim, n_heads, n_layers, n_kv_heads):
    key = jax.random.split(key, 4)
    params = {
        'token_embedding': init_weight(key[0], (vocab_size, dim)),
        'norm_f': init_weight(key[1], (dim,), scale=1.0),
        'output': init_weight(key[2], (dim, vocab_size)),
    }
    block_keys = jax.random.split(key[3], n_layers)
    params['blocks'] = [init_transformer_block(block_key, dim, n_heads, n_kv_heads) 
                       for block_key in block_keys]
    return params
```

**Purpose**: Initialize the complete model with embedding, transformer blocks, and output layer.

**Example**:
```python
key = jax.random.PRNGKey(42)
params = init_model_params(key, vocab_size=32000, dim=512, n_heads=8, n_layers=6, n_kv_heads=8)
# Returns dict with:
# token_embedding: (32000, 512)
# norm_f: (512,) - final normalization
# output: (512, 32000) - output projection
# blocks: list of 6 transformer blocks
```

---

## 7. Core Functions

### Multi-Head Attention
```python
def attention(params, x, mask, freqs_cis, n_heads, n_kv_heads, cache=None, position=0):
    B, T, C = x.shape
    head_dim = C // n_heads
    
    # Reshape weights for proper matrix multiplication
    wq = params['wq'].reshape(C, n_heads * head_dim)
    wk = params['wk'].reshape(C, n_kv_heads * head_dim)
    wo = params['wo'].reshape(n_heads * head_dim, C)
    
    q = jnp.dot(x, wq).reshape(B, T, n_heads, head_dim)
    k = jnp.dot(x, wk).reshape(B, T, n_kv_heads, head_dim)
    
    # Handle wv differently - it's (n_heads, C, head_dim)
    wv = params['wv']  # Shape: (n_heads, C, head_dim)
    v = jnp.einsum('btd,ndh->btnh', x, wv)  # Shape: (B, T, n_heads, head_dim)
    
    q, k = apply_rotary_emb(q, k, freqs_cis[position:position+T])
    if cache is not None:
        k = jnp.concatenate([cache[0], k], axis=1)
        v = jnp.concatenate([cache[1], v], axis=1)
    new_cache = (k, v)
    k = repeat_kv(k, n_heads // n_kv_heads)
    v = repeat_kv(v, n_heads // n_kv_heads)
    q, k, v = map(lambda x: x.transpose(0, 2, 1, 3), (q, k, v))
    scores = jnp.matmul(q, k.transpose(0, 1, 3, 2)) / math.sqrt(head_dim)
    if mask is not None:
        scores = scores + mask[:, :, :T, :T]
    scores = jax.nn.softmax(scores, axis=-1)
    output = jnp.matmul(scores, v)
    output = output.transpose(0, 2, 1, 3).reshape(B, T, -1)
    return jnp.dot(output, wo), new_cache
```

**Purpose**: Implement multi-head attention with rotary embeddings and caching.

**Step-by-step process**:
1. **Project to Q, K, V**: Transform input to query, key, value
2. **Apply rotary embeddings**: Add positional information
3. **Handle caching**: For autoregressive generation
4. **Compute attention scores**: Q @ K^T / sqrt(head_dim)
5. **Apply mask**: For causal attention
6. **Softmax**: Normalize attention weights
7. **Apply attention**: Weighted sum of values
8. **Project output**: Final linear transformation

**Example**:
```python
# Input: (batch_size, seq_len, hidden_dim)
x = jnp.random.normal(jax.random.PRNGKey(0), (2, 10, 512))
mask = jnp.tril(jnp.ones((10, 10)))  # Causal mask
freqs_cis = precompute_freq_cis(64, 10)  # 64 = 512/8 heads

output, cache = attention(attn_params, x, mask, freqs_cis, n_heads=8, n_kv_heads=8)
# Output: (2, 10, 512), cache: (k, v) for next iteration
```

### Feed-Forward Network (SwiGLU)
```python
def feed_forward(params, x):
    return jnp.dot(jax.nn.silu(jnp.dot(x, params['w3'])) * jnp.dot(x, params['w1']), params['w2'])
```

**Purpose**: Implement SwiGLU activation: `FFN(x) = (SiLU(xW3) ⊙ xW1)W2`

**Example**:
```python
x = jnp.random.normal(jax.random.PRNGKey(0), (2, 10, 512))
ffn_params = init_ffn_weighst(jax.random.PRNGKey(1), 512)
output = feed_forward(ffn_params, x)  # Shape: (2, 10, 512)
```

### Transformer Block
```python
def transformer_block(params, x, mask, freqs_cis, n_heads, n_kv_heads, cache=None, position=0, training=False, dropout_rate=0.0, key=None):
    attn_output, new_cache = attention(params['attn'], rms_norm(x, params['attention_norm']), mask, freqs_cis, n_heads, n_kv_heads, cache, position)
    if training:
        dropout_key, key = jax.random.split(key)
        attn_output = jax.random.bernoulli(dropout_key, 1-dropout_rate, shape=attn_output.shape) * attn_output / (1-dropout_rate)
    h = attn_output + x  # Residual connection
    ff_output = feed_forward(params['ffn'], rms_norm(h, params['ffn_norm']))
    if training:
        dropout_key, key = jax.random.split(key)
        ff_output = jax.random.bernoulli(dropout_key, 1-dropout_rate, shape=ff_output.shape) * ff_output / (1-dropout_rate)
    out = ff_output + h  # Residual connection
    return out, new_cache
```

**Purpose**: Complete transformer block with attention, feed-forward, normalization, and residual connections.

**Architecture**:
```
Input → RMS Norm → Attention → Dropout → Residual → RMS Norm → FFN → Dropout → Residual → Output
```

**Example**:
```python
x = jnp.random.normal(jax.random.PRNGKey(0), (2, 10, 512))
block_params = init_transformer_block(jax.random.PRNGKey(1), 512, 8, 8)
mask = jnp.tril(jnp.ones((10, 10)))
freqs_cis = precompute_freq_cis(64, 10)

output, cache = transformer_block(block_params, x, mask, freqs_cis, 8, 8)
# Output: (2, 10, 512), cache for next iteration
```

---

## 8. Complete Model Forward Pass

```python
def model_forward(params, inputs, config, cache=None, position=0):
    B, T = inputs.shape
    h = params['token_embedding'][inputs]  # Token embedding
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
    h = rms_norm(h, params['norm_f'])  # Final normalization
    logits = jnp.dot(h, params['output'])  # Output projection
    return logits, new_caches
```

**Purpose**: Complete forward pass through the entire model.

**Process**:
1. **Token embedding**: Convert token IDs to vectors
2. **Precompute**: Frequencies and causal mask
3. **Transformer blocks**: Process through all layers
4. **Final norm**: RMS normalization
5. **Output projection**: Convert to logits

**Example**:
```python
# Initialize model
key = jax.random.PRNGKey(42)
config = type('Config', (), {
    'vocab_size': 32000,
    'dim': 512,
    'n_heads': 8,
    'n_kv_heads': 8,
    'n_layers': 6,
    'max_seq_len': 2048,
    'dropout_rate': 0.1
})()
params = init_model_params(key, config.vocab_size, config.dim, config.n_heads, config.n_layers, config.n_kv_heads)

# Forward pass
inputs = jnp.array([[1, 2, 3, 4, 5]])  # Token IDs
logits, caches = model_forward(params, inputs, config)
# logits: (1, 5, 32000) - probability distribution over vocabulary
# caches: list of (k, v) for each layer for next iteration
```

---

## 9. Configuration and Setup

```python
os.environ['JAX_PLATFORM_NAME'] = 'gpu'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
print("JAX devices:", jax.devices())
```

**Purpose**: Configure JAX to use GPU and prevent memory preallocation.

**Example Output**:
```
JAX devices: [CpuDevice(id=0)]  # or [GpuDevice(id=0)] if GPU available
```

---

## Summary

This implementation provides a complete Llama transformer model with:

✅ **Modern Architecture**: RMS normalization, rotary embeddings, SwiGLU activation
✅ **Efficient Attention**: Multi-head attention with caching for autoregressive generation
✅ **Proper Initialization**: Xavier initialization for stable training
✅ **Flexible Configuration**: Support for different model sizes and attention variants
✅ **JAX Integration**: Immutable arrays, automatic differentiation ready

The model is ready for both inference and training (with gradient computation added). 