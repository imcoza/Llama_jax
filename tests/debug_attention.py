import jax
import jax.numpy as jnp
from llama_jax_fixed import *

# Test with small dimensions
key = jax.random.PRNGKey(42)
dims, n_heads, n_kv_heads = 64, 4, 4
attn_params = init_attention_weights(key, dims, n_heads, n_kv_heads)

print("Attention parameter shapes:")
for k, v in attn_params.items():
    print(f"  {k}: {v.shape}")

# Create sample input
batch_size, seq_len = 1, 5
x = jax.random.normal(key, (batch_size, seq_len, dims))
print(f"\nInput shape: {x.shape}")

# Create mask
mask = jnp.tril(jnp.ones((seq_len, seq_len)))
mask = jnp.where(mask == 0, -1e9, 0.0)
mask = mask[None, None, :, :]

# Precompute frequencies
head_dim = dims // n_heads
freqs_cis = precompute_freq_cis(head_dim, seq_len)

print(f"\nHead dimension: {head_dim}")
print(f"Frequencies shape: {freqs_cis.shape}")

# Test attention step by step
B, T, C = x.shape

# Reshape weights
wq = attn_params['wq'].reshape(C, n_heads * head_dim)
wk = attn_params['wk'].reshape(C, n_kv_heads * head_dim)
wo = attn_params['wo'].reshape(n_heads * head_dim, C)

print(f"\nReshaped weights:")
print(f"wq: {wq.shape}")
print(f"wk: {wk.shape}")
print(f"wo: {wo.shape}")

# Compute Q, K, V
q = jnp.dot(x, wq).reshape(B, T, n_heads, head_dim)
k = jnp.dot(x, wk).reshape(B, T, n_kv_heads, head_dim)

print(f"\nQ, K shapes:")
print(f"q: {q.shape}")
print(f"k: {k.shape}")

# Handle wv
wv = attn_params['wv']
v = jnp.einsum('btd,ndh->btnh', x, wv)
print(f"v: {v.shape}")

# Apply rotary embeddings
q, k = apply_rotary_emb(q, k, freqs_cis)
print(f"\nAfter rotary embeddings:")
print(f"q: {q.shape}")
print(f"k: {k.shape}")

# Repeat KV heads
k = repeat_kv(k, n_heads // n_kv_heads)
v = repeat_kv(v, n_heads // n_kv_heads)
print(f"\nAfter repeating KV heads:")
print(f"k: {k.shape}")
print(f"v: {v.shape}")

# Transpose for attention
q, k, v = map(lambda x: x.transpose(0, 2, 1, 3), (q, k, v))
print(f"\nAfter transpose:")
print(f"q: {q.shape}")
print(f"k: {k.shape}")
print(f"v: {v.shape}")

# Compute attention scores
scores = jnp.matmul(q, k.transpose(0, 1, 3, 2)) / math.sqrt(head_dim)
print(f"\nAttention scores shape: {scores.shape}")

# Apply mask
if mask is not None:
    scores = scores + mask[:, :, :T, :T]
print(f"After mask: {scores.shape}")

# Softmax
scores = jax.nn.softmax(scores, axis=-1)
print(f"After softmax: {scores.shape}")

# Apply attention to values
output = jnp.matmul(scores, v)
print(f"After attention: {output.shape}")

# Transpose back and reshape
output = output.transpose(0, 2, 1, 3).reshape(B, T, n_heads * head_dim)
print(f"After reshape: {output.shape}")

# Final projection
final_output = jnp.dot(output, wo)
print(f"Final output: {final_output.shape}") 