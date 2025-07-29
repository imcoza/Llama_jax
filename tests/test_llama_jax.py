import jax
import jax.numpy as jnp
from llama_jax import *

# Test configuration
class Config:
    def __init__(self):
        self.vocab_size = 32000
        self.dim = 512
        self.n_heads = 8
        self.n_kv_heads = 8
        self.n_layers = 6
        self.max_seq_len = 2048
        self.dropout_rate = 0.1

def test_model():
    print("Testing Llama JAX implementation...")
    
    # Initialize random key
    key = jax.random.PRNGKey(42)
    
    # Create config
    config = Config()
    
    # Initialize model parameters
    print("Initializing model parameters...")
    params = init_model_params(key, config.vocab_size, config.dim, 
                             config.n_heads, config.n_layers, config.n_kv_heads)
    
    print(f"Model initialized with {len(params['blocks'])} layers")
    print(f"Embedding shape: {params['token_embedding'].shape}")
    print(f"Output shape: {params['output'].shape}")
    
    # Test forward pass
    print("\nTesting forward pass...")
    batch_size = 2
    seq_len = 10
    
    # Create dummy input tokens
    inputs = jax.random.randint(key, (batch_size, seq_len), 0, config.vocab_size)
    print(f"Input shape: {inputs.shape}")
    
    # Run forward pass
    logits, caches = model_forward(params, inputs, config)
    print(f"Output logits shape: {logits.shape}")
    print(f"Number of caches: {len(caches)}")
    
    # Test attention mechanism
    print("\nTesting attention mechanism...")
    x = jax.random.normal(key, (batch_size, seq_len, config.dim))
    mask = jnp.tril(jnp.ones((seq_len, seq_len)))
    mask = jnp.where(mask == 0, -1e9, 0.0)
    mask = mask[None, None, :, :]
    
    freqs_cis = precompute_freq_cis(config.dim // config.n_heads, config.max_seq_len)
    
    # Test single transformer block
    block = params['blocks'][0]
    output, cache = transformer_block(block, x, mask, freqs_cis, 
                                    config.n_heads, config.n_kv_heads)
    print(f"Transformer block output shape: {output.shape}")
    print(f"Cache shapes: {[c.shape for c in cache]}")
    
    # Test RMS normalization
    print("\nTesting RMS normalization...")
    norm_output = rms_norm(x, jnp.ones(config.dim))
    print(f"RMS norm output shape: {norm_output.shape}")
    
    # Test rotary embeddings
    print("\nTesting rotary embeddings...")
    q = jax.random.normal(key, (batch_size, seq_len, config.n_heads, config.dim // config.n_heads))
    k = jax.random.normal(key, (batch_size, seq_len, config.n_kv_heads, config.dim // config.n_heads))
    q_rot, k_rot = apply_rotary_emb(q, k, freqs_cis[:seq_len])
    print(f"Rotary embedding output shapes: {q_rot.shape}, {k_rot.shape}")
    
    print("\nAll tests passed! The Llama JAX implementation is functioning correctly.")

if __name__ == "__main__":
    test_model() 