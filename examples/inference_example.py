#!/usr/bin/env python3
"""
Inference Example for Llama JAX Implementation

This script demonstrates how to use a trained Llama model for text generation.
"""

import jax
import jax.numpy as jnp
import os
import sys

# Add parent directory to path to import llama_jax
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llama_jax import *

def main():
    print("🎭 Llama JAX Inference Example")
    print("=" * 50)
    
    # Initialize tokenizer
    enc = tiktoken.get_encoding("gpt2")
    
    # Model configuration (must match training config)
    class InferenceConfig:
        vocab_size = enc.n_vocab
        dim = 128
        n_layers = 4
        n_heads = 4
        n_kv_heads = 4
        max_seq_len = 64
        batch_size = 8
        learning_rate = 3e-4
        dropout_rate = 0.0  # No dropout during inference
    
    config = InferenceConfig()
    
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
    
    print("✅ Model initialized!")
    
    # Try to load trained model if available
    model_path = 'trained_model_example.pkl'
    if os.path.exists(model_path):
        print(f"📂 Loading trained model from {model_path}")
        params = load_params(model_path)
        print("✅ Trained model loaded!")
    else:
        print("⚠️  No trained model found. Using initialized model.")
    
    # Example prompts
    prompts = [
        "The quick brown fox",
        "Machine learning is",
        "Deep learning models",
        "Transformers have",
        "JAX provides"
    ]
    
    print(f"\n🎯 Generating text for {len(prompts)} prompts...")
    print("-" * 40)
    
    for i, prompt_text in enumerate(prompts, 1):
        print(f"\n{i}. Prompt: '{prompt_text}'")
        
        # Tokenize prompt
        prompt_tokens = enc.encode(prompt_text)
        print(f"   Tokens: {prompt_tokens}")
        
        # Generate text
        generated_tokens = generate(params, prompt_tokens, max_new_tokens=15, config=config)
        
        # Decode generated text
        generated_text = enc.decode(generated_tokens)
        print(f"   Generated: '{generated_text}'")
        
        # Show token probabilities for first few tokens
        if len(generated_tokens) > len(prompt_tokens):
            new_tokens = generated_tokens[len(prompt_tokens):len(prompt_tokens)+3]
            print(f"   Next tokens: {new_tokens}")
    
    print(f"\n🎉 Inference completed!")
    
    # Interactive generation
    print(f"\n💬 Interactive Generation (type 'quit' to exit)")
    print("-" * 40)
    
    while True:
        user_input = input("\nEnter prompt: ").strip()
        
        if user_input.lower() == 'quit':
            break
        
        if not user_input:
            continue
        
        try:
            # Tokenize user input
            prompt_tokens = enc.encode(user_input)
            
            # Generate text
            generated_tokens = generate(params, prompt_tokens, max_new_tokens=20, config=config)
            
            # Decode and display
            generated_text = enc.decode(generated_tokens)
            print(f"Generated: {generated_text}")
            
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main() 