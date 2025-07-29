import jax
import jax.numpy as jnp
from llama_jax_fixed import *

# Test with simple configuration
key = jax.random.PRNGKey(42)

# Simple config for testing
class SimpleConfig:
    vocab_size = 1000
    dim = 64
    n_layers = 2
    n_heads = 4
    n_kv_heads = 4  # Same as n_heads to avoid GQA complexity
    max_seq_len = 128
    batch_size = 2
    learning_rate = 3e-4
    dropout_rate = 0.0

config = SimpleConfig()

# Initialize model
params = init_model_params(
    key=key,
    vocab_size=config.vocab_size,
    dim=config.dim,
    n_layers=config.n_layers,
    n_heads=config.n_heads,
    n_kv_heads=config.n_kv_heads
)

print("Model initialized successfully!")
print(f"Number of blocks: {len(params['blocks'])}")
print(f"Embedding shape: {params['token_embedding'].shape}")

# Test forward pass
test_input = jnp.array([[1, 2, 3, 4, 5]])
print(f"\nTest input shape: {test_input.shape}")

try:
    logits, caches = model_forward(params, test_input, config)
    print(f"Forward pass successful! Logits shape: {logits.shape}")
    print(f"Number of caches: {len(caches)}")
    
    # Test generation
    prompt = [1, 2, 3]
    generated = generate(params, prompt, 5, config)
    print(f"Generation successful! Generated tokens: {generated}")
    
    print("\n✅ All tests passed!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc() 