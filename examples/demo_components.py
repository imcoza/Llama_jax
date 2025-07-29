#!/usr/bin/env python3
"""
Demonstration of Llama JAX Implementation Components
This script shows how each component works with concrete examples.
"""

import jax
import jax.numpy as jnp
from llama_jax import *

def demo_rms_norm():
    print("=== RMS Normalization Demo ===")
    
    # Create sample input
    x = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # Shape: (2, 3)
    weight = jnp.array([1.0, 1.0, 1.0])  # Shape: (3,)
    
    print(f"Input shape: {x.shape}")
    print(f"Input: {x}")
    print(f"Weight: {weight}")
    
    # Apply RMS normalization
    normalized = rms_norm(x, weight)
    print(f"Normalized shape: {normalized.shape}")
    print(f"Normalized: {normalized}")
    print()

def demo_rotary_embeddings():
    print("=== Rotary Embeddings Demo ===")
    
    # Create sample query and key tensors
    batch_size, seq_len, n_heads, head_dim = 1, 4, 2, 8
    xq = jnp.random.normal(jax.random.PRNGKey(0), (batch_size, seq_len, n_heads, head_dim))
    xk = jnp.random.normal(jax.random.PRNGKey(1), (batch_size, seq_len, n_heads, head_dim))
    
    print(f"Query shape: {xq.shape}")
    print(f"Key shape: {xk.shape}")
    
    # Precompute frequencies
    freqs_cis = precompute_freq_cis(head_dim, seq_len)
    print(f"Frequencies shape: {freqs_cis.shape}")
    
    # Apply rotary embeddings
    xq_rot, xk_rot = apply_rotary_emb(xq, xk, freqs_cis)
    print(f"Rotated query shape: {xq_rot.shape}")
    print(f"Rotated key shape: {xk_rot.shape}")
    print()

def demo_attention():
    print("=== Multi-Head Attention Demo ===")
    
    # Initialize attention parameters
    key = jax.random.PRNGKey(42)
    dims, n_heads, n_kv_heads = 64, 4, 4
    attn_params = init_attention_weights(key, dims, n_heads, n_kv_heads)
    
    print("Attention parameter shapes:")
    for k, v in attn_params.items():
        print(f"  {k}: {v.shape}")
    
    # Create sample input
    batch_size, seq_len = 2, 5
    x = jnp.random.normal(key, (batch_size, seq_len, dims))
    print(f"\nInput shape: {x.shape}")
    
    # Create causal mask
    mask = jnp.tril(jnp.ones((seq_len, seq_len)))
    mask = jnp.where(mask == 0, -1e9, 0.0)
    mask = mask[None, None, :, :]
    
    # Precompute frequencies
    head_dim = dims // n_heads
    freqs_cis = precompute_freq_cis(head_dim, seq_len)
    
    # Apply attention
    output, cache = attention(attn_params, x, mask, freqs_cis, n_heads, n_kv_heads)
    print(f"Output shape: {output.shape}")
    print(f"Cache shapes: {[c.shape for c in cache]}")
    print()

def demo_feed_forward():
    print("=== Feed-Forward Network (SwiGLU) Demo ===")
    
    # Initialize FFN parameters
    key = jax.random.PRNGKey(42)
    dim = 64
    ffn_params = init_ffn_weighst(key, dim)
    
    print("FFN parameter shapes:")
    for k, v in ffn_params.items():
        print(f"  {k}: {v.shape}")
    
    # Create sample input
    batch_size, seq_len = 2, 5
    x = jnp.random.normal(key, (batch_size, seq_len, dim))
    print(f"\nInput shape: {x.shape}")
    
    # Apply feed-forward
    output = feed_forward(ffn_params, x)
    print(f"Output shape: {output.shape}")
    print()

def demo_transformer_block():
    print("=== Transformer Block Demo ===")
    
    # Initialize transformer block
    key = jax.random.PRNGKey(42)
    dims, n_heads, n_kv_heads = 64, 4, 4
    block_params = init_transformer_block(key, dims, n_heads, n_kv_heads)
    
    print("Block parameter structure:")
    for k, v in block_params.items():
        if isinstance(v, dict):
            print(f"  {k}:")
            for sub_k, sub_v in v.items():
                print(f"    {sub_k}: {sub_v.shape}")
        else:
            print(f"  {k}: {v.shape}")
    
    # Create sample input
    batch_size, seq_len = 2, 5
    x = jnp.random.normal(key, (batch_size, seq_len, dims))
    print(f"\nInput shape: {x.shape}")
    
    # Create causal mask
    mask = jnp.tril(jnp.ones((seq_len, seq_len)))
    mask = jnp.where(mask == 0, -1e9, 0.0)
    mask = mask[None, None, :, :]
    
    # Precompute frequencies
    head_dim = dims // n_heads
    freqs_cis = precompute_freq_cis(head_dim, seq_len)
    
    # Apply transformer block
    output, cache = transformer_block(block_params, x, mask, freqs_cis, n_heads, n_kv_heads)
    print(f"Output shape: {output.shape}")
    print(f"Cache shapes: {[c.shape for c in cache]}")
    print()

def demo_full_model():
    print("=== Full Model Demo ===")
    
    # Initialize model
    key = jax.random.PRNGKey(42)
    config = type('Config', (), {
        'vocab_size': 1000,
        'dim': 64,
        'n_heads': 4,
        'n_kv_heads': 4,
        'n_layers': 2,
        'max_seq_len': 128,
        'dropout_rate': 0.1
    })()
    
    params = init_model_params(key, config.vocab_size, config.dim, 
                             config.n_heads, config.n_layers, config.n_kv_heads)
    
    print("Model structure:")
    print(f"  Token embedding: {params['token_embedding'].shape}")
    print(f"  Final norm: {params['norm_f'].shape}")
    print(f"  Output: {params['output'].shape}")
    print(f"  Number of blocks: {len(params['blocks'])}")
    
    # Create sample input (token IDs)
    inputs = jnp.array([[1, 2, 3, 4, 5]])  # Shape: (1, 5)
    print(f"\nInput tokens: {inputs}")
    print(f"Input shape: {inputs.shape}")
    
    # Forward pass
    logits, caches = model_forward(params, inputs, config)
    print(f"Output logits shape: {logits.shape}")
    print(f"Number of caches: {len(caches)}")
    
    # Show probability distribution for first token
    probs = jax.nn.softmax(logits[0, 0, :])
    top_5_indices = jnp.argsort(probs)[-5:][::-1]
    print(f"\nTop 5 predicted tokens for first position:")
    for i, idx in enumerate(top_5_indices):
        print(f"  {i+1}. Token {idx}: {probs[idx]:.4f}")
    print()

def main():
    print("Llama JAX Implementation - Component Demonstrations\n")
    
    demo_rms_norm()
    demo_rotary_embeddings()
    demo_attention()
    demo_feed_forward()
    demo_transformer_block()
    demo_full_model()
    
    print("All demonstrations completed successfully! ✅")

if __name__ == "__main__":
    main() 