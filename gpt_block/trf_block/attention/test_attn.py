import torch
from masked_mha import MultiHeadAttention

def test_multihead_attention():
    # Hyperparameters
    batch_size = 2
    seq_len = 5
    d_in = 32
    d_out = 64
    context_length = 10
    dropout = 0.1
    num_heads = 8

    # Create dummy input tensor (b, seq_len, d_in)
    x = torch.randn(batch_size, seq_len, d_in)

    # Initialize MultiHeadAttention
    mha = MultiHeadAttention(d_in=d_in, d_out=d_out, context_length=context_length, dropout=dropout, num_heads=num_heads)

    # Forward pass
    output = mha(x)

    print("Input shape: ", x.shape)          # Expect (b, seq_len, d_in)
    print("Output shape:", output.shape)     # Expect (b, seq_len, d_out)

if __name__ == "__main__":
    test_multihead_attention()
