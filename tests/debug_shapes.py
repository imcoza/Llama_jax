import jax
import jax.numpy as jnp
from llama_jax import *

# Initialize a small model to debug shapes
key = jax.random.PRNGKey(42)
config = type('Config', (), {
    'vocab_size': 1000,
    'dim': 64,
    'n_heads': 4,
    'n_kv_heads': 4,
    'n_layers': 1,
    'max_seq_len': 128,
    'dropout_rate': 0.1
})()

params = init_model_params(key, config.vocab_size, config.dim, 
                         config.n_heads, config.n_layers, config.n_kv_heads)

print("Model parameters structure:")
print(f"Number of blocks: {len(params['blocks'])}")
print(f"Token embedding shape: {params['token_embedding'].shape}")
print(f"Output shape: {params['output'].shape}")

block = params['blocks'][0]
print(f"\nBlock structure:")
print(f"Attention weights: {list(block['attn'].keys())}")
for k, v in block['attn'].items():
    print(f"  {k}: {v.shape}")
print(f"FFN weights: {list(block['ffn'].keys())}")
for k, v in block['ffn'].items():
    print(f"  {k}: {v.shape}")
print(f"Attention norm: {block['attention_norm'].shape}")
print(f"FFN norm: {block['ffn_norm'].shape}")

# Test with small input
x = jax.random.normal(key, (1, 5, config.dim))
print(f"\nInput shape: {x.shape}")

# Test attention weights
attn_params = block['attn']
print(f"\nAttention parameter shapes:")
print(f"wq: {attn_params['wq'].shape}")
print(f"wk: {attn_params['wk'].shape}")
print(f"wv: {attn_params['wv'].shape}")
print(f"wo: {attn_params['wo'].shape}")

# Test matrix multiplication
head_dim = config.dim // config.n_heads
print(f"\nHead dimension: {head_dim}")

# Reshape for proper multiplication
wq_reshaped = attn_params['wq'].reshape(config.dim, config.n_heads * head_dim)
wk_reshaped = attn_params['wk'].reshape(config.dim, config.n_kv_heads * head_dim)
wv_reshaped = attn_params['wv'].reshape(config.n_heads, config.dim, head_dim)
wo_reshaped = attn_params['wo'].reshape(config.n_heads * head_dim, config.dim)

print(f"Reshaped shapes:")
print(f"wq_reshaped: {wq_reshaped.shape}")
print(f"wk_reshaped: {wk_reshaped.shape}")
print(f"wv_reshaped: {wv_reshaped.shape}")
print(f"wo_reshaped: {wo_reshaped.shape}")

# Test matrix multiplications using the attention function
from llama_jax import attention

# Create dummy parameters for testing
test_params = {
    'wq': attn_params['wq'],
    'wk': attn_params['wk'],
    'wv': attn_params['wv'],
    'wo': attn_params['wo']
}

# Test attention function
freqs_cis = precompute_freq_cis(head_dim, config.max_seq_len)
output, cache = attention(test_params, x, None, freqs_cis, config.n_heads, config.n_kv_heads)

print(f"\nAttention output shape: {output.shape}")
print(f"Cache shapes: {[c.shape for c in cache]}") 