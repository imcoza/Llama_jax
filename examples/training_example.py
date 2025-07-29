#!/usr/bin/env python3
"""
Training Example for Llama JAX Implementation

This script demonstrates how to train the Llama model on custom text data.
"""

import jax
import jax.numpy as jnp
import os
import sys

# Add parent directory to path to import llama_jax
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llama_jax import *

def create_sample_data():
    """Create sample training data."""
    sample_text = """
    The quick brown fox jumps over the lazy dog.
    Machine learning is a subset of artificial intelligence.
    Deep learning models can learn complex patterns from data.
    Transformers have revolutionized natural language processing.
    JAX provides automatic differentiation and GPU acceleration.
    """
    return sample_text

def main():
    print("🚀 Llama JAX Training Example")
    print("=" * 50)
    
    # Create sample data
    text = create_sample_data()
    
    # Initialize tokenizer
    enc = tiktoken.get_encoding("gpt2")
    tokens = enc.encode(text)
    data = jnp.array(tokens)
    
    print(f"📊 Dataset: {len(tokens)} tokens")
    
    # Model configuration
    class TrainingConfig:
        vocab_size = enc.n_vocab
        dim = 128  # Smaller for faster training
        n_layers = 4
        n_heads = 4
        n_kv_heads = 4
        max_seq_len = 64
        batch_size = 8
        learning_rate = 3e-4
        dropout_rate = 0.1
    
    config = TrainingConfig()
    
    print(f"🧠 Model: {config.dim} dim, {config.n_layers} layers, {config.n_heads} heads")
    
    # Initialize model
    key = jax.random.PRNGKey(42)
    params = init_model_params(
        key=key,
        vocab_size=config.vocab_size,
        dim=config.dim,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        n_kv_heads=config.n_kv_heads
    )
    
    print("✅ Model initialized successfully!")
    
    # Test forward pass
    test_input = jnp.array([[1, 2, 3, 4, 5]])
    logits, _ = model_forward(params, test_input, config)
    print(f"✅ Forward pass: {logits.shape}")
    
    # Train the model
    print("\n🎯 Starting training...")
    print("-" * 30)
    
    trained_params = train(
        num_epochs=3,
        steps_per_epoch=50
    )
    
    print("\n✅ Training completed!")
    
    # Test generation
    print("\n🎭 Testing text generation...")
    prompt = [1, 2, 3]  # Start tokens
    generated = generate(trained_params, prompt, max_new_tokens=10, config=config)
    
    print(f"Generated tokens: {generated}")
    
    # Save model
    save_params(trained_params, 'trained_model_example.pkl')
    print("💾 Model saved as 'trained_model_example.pkl'")
    
    print("\n🎉 Example completed successfully!")

if __name__ == "__main__":
    main() 