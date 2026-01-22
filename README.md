# GPT-2 From Scratch

This repository presents a from-scratch PyTorch implementation of GPT-2, covering everything from raw text tokenization to autoregressive generation and model loading from pretrained weights.
It is designed as a deep learning systems project that bridges theory, implementation, and real-world LLM applications.

## 🚀 Project Highlights

🔨 Built GPT-2 entirely from scratch in PyTorch (no Hugging Face models)

📐 Supports GPT-2 Small (124M) scale

🧠 Covers all core transformer internals

🧪 Includes model loading and weight transfer from pretrained GPT-2

🗣️ Enables text generation with various decoding strategies

📊 Tracks training and generation capabilities

## 📌 Why This Project Matters

Deep LLM mastery
Understand how embeddings, attention, residuals, and normalization interact to generate coherent language.

Scalable & practical
The codebase provides a foundation for understanding and extending GPT-2 architectures.

Production-ready pipelines
Load pretrained weights and generate text with advanced decoding techniques.

This project serves as a hands-on bridge between foundational transformer theory and real-world LLM engineering.

## 🧠 GPT-2 From Scratch on Raw Text
### 🎯 Objective

Recreate GPT-2 from the ground up and train it on raw text (the-verdict.txt) to build a fully functional autoregressive language model, with capabilities to load and utilize pretrained GPT-2 weights.

### 🧩 Implementation Details

#### 1️⃣ Tokenization & Embeddings

Built a custom tokenizer pipeline using GPT-2 BPE tokenizer via tiktoken

Handles out-of-vocabulary tokens efficiently

Converts raw text into input–target pairs using PyTorch Dataset and DataLoader

Combined:
- Token embeddings
- Absolute positional embeddings

to produce structured input representations

#### 2️⃣ Transformer Block Implementation

Implemented all components from scratch:

- Layer Normalization
- GELU activation
- Feed-Forward Networks (MLP)
- Residual connections

Architecture:
- 12 stacked transformer blocks (GPT-2 Small / 124M)
- Weight tying between input embeddings and output projection
- Strict adherence to GPT-2 architectural design

#### 3️⃣ Attention Mechanisms

Implemented:
- Self-Attention
- Causal (masked) attention
- Multi-Head Attention

Verified tensor shapes:
- (Batch, Sequence, d_model)
- (Heads, d_k, d_v)

Scaled dot-product attention using √d_k for numerical stability

Generated context vectors as query-conditioned weighted sums of values

#### 4️⃣ Autoregressive Text Generation

Implemented token-by-token generation with advanced decoding strategies:

🔹 Decoding Techniques
- Top-k Sampling
- Temperature Scaling
- Top-k + Temperature combination

Balances coherence and diversity for higher-quality generations

#### 5️⃣ Model Loading & Weight Transfer

Load pretrained GPT-2 weights (124M) into the from-scratch model

Transfer weights from TensorFlow checkpoints to PyTorch model

Verify model functionality with pretrained weights

#### 6️⃣ Training Loop & Metrics

Optimizer: AdamW

Learning rate scheduling

Gradient clipping for stability

Loss function: Cross-Entropy

Tracked:
- Training loss
- Validation loss
- Perplexity

## 📁 Project Structure

- `config/`: Configuration files for model parameters
- `data/`: Data preparation and dataset handling
- `gpt_block/`: Core GPT model implementation
- `trf_block/`: Transformer block components
- `load_pretrained_wt/`: Scripts for loading pretrained weights
- `post_training/`: Text generation and post-training utilities
- `pretrained_gpt2/`: Pretrained GPT-2 model files (124M)

## 🚀 Getting Started

1. Set up the Python environment (virtual environment in `env/`)
2. Prepare data using `data/prepare_data.py`
3. Train the model or load pretrained weights
4. Generate text using `post_training/generate.py`

## 📊 Results

The implementation successfully recreates GPT-2 Small (124M) from scratch and demonstrates text generation capabilities with various decoding strategies.