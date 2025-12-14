import torch
from gpt import GPTModel  

def test_gpt_model():
    # Define a mock configuration
    cfg = {
        'vocab_size': 1000,
        'emb_dim': 64,
        'context_length': 16,
        'drop_rate': 0.1,
        'n_layers': 4,
        'num_heads': 8
    }

    # Create dummy input: batch of token indices
    batch_size = 2
    seq_len = cfg['context_length']
    dummy_input = torch.randint(0, cfg['vocab_size'], (batch_size, seq_len))

    # Initialize model
    model = GPTModel(cfg)

    # Forward pass
    logits = model(dummy_input)

    # Output shapes
    print("Input shape: ", dummy_input.shape)     # (b, seq_len)
    print("Logits shape:", logits.shape)          # (b, seq_len, vocab_size)

if __name__ == "__main__":
    test_gpt_model()
