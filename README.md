# tiny-gpt-from-scratch

A minimal, educational implementation of a **decoder-only Transformer (GPT-style)** language model built **from scratch** using PyTorch.  
This project focuses on understanding the *core mechanics* of modern large language models by implementing every major component explicitly rather than relying on high-level libraries.

> This repository is **inspired by Andrej Karpathy’s “GPT from scratch” lecture**, but the code is being written, structured, and extended independently as a personal learning and experimentation project.

---

## Motivation

Large Language Models (LLMs) often appear opaque due to their scale and abstraction layers.  
The goal of this project is to **demystify GPT-style models** by:

- Building a Transformer decoder **step-by-step**
- Understanding how tokenization, embeddings, attention, and training interact
- Creating a compact model that can be trained and experimented with on a laptop

This repository is intended for **learning, experimentation, and extension**, not for production use.

---

## What This Project Implements

- Character-level language modeling
- Decoder-only Transformer architecture
- Causal self-attention (masked attention)
- Multi-head attention
- Feed-forward networks
- Residual connections + LayerNorm
- Autoregressive text generation
- Training loop with cross-entropy loss

All components are implemented explicitly to preserve clarity.

---

## Model Architecture (High-Level)
```
Input text
  |
  v
Tokenization (character-level)
  |
  v
Token Embedding + Positional Embedding
  |
  v
[ Repeated N times ]
  |
  +--> Masked Multi-Head Self Attention
  |       |
  |       +--> Residual Connection
  |       +--> LayerNorm
  |
  +--> Feed Forward Network
  |       |
  |       +--> Residual Connection
  |       +--> LayerNorm
  |
  v
Linear projection to vocabulary
  |
  v
Softmax  -->  Next-token prediction

```
---

## Dataset

- **Tiny Shakespeare** dataset  
  (commonly used for educational language modeling tasks)

The model learns to predict the next character given a context window.

---

## Repository Structure
```
tiny-gpt-from-scratch/
├── tinygpt.ipynb   # Main notebook with full implementation
├── README.md      # Project documentation
└── LICENSE        # MIT License
```

---

## How to Run
```
1. Clone the repository:
   ```bash
   git clone https://github.com/prateek-g-bit/tiny-gpt-from-scratch.git
   cd tiny-gpt-from-scratch
2.Open notebook:
  jupyter notebook tinygpt.ipynb
3.Run cells sequentially to:
  Load data
  Train the model
  Generate text samples
```
## Example Output
After training, the model can generate Shakespeare-like text at the character level, demonstrating that it has learned syntax, structure, and basic semantics.
(Note: Output quality depends on training time and hyperparameters.)

## Learning Outcomes
```
Through this project, the following concepts were explored in depth:
Why GPT is decoder-only
How causal masking enables autoregressive generation
Role of embeddings and positional encodings
Attention as a similarity-based information routing mechanism
Training dynamics of language models
```
## Possible Extensions
This repository is intentionally minimal and designed to be extended:
```
Byte-level or word-level tokenization
Training on network logs / system logs instead of text
Scaling to larger context windows
Adding dropout, weight tying, or RMSNorm
Converting to a log-sequence anomaly detection model
Using learned representations for downstream tasks
```

## Acknowledgements

Inspired by Andrej Karpathy’s educational lectures
Dataset: Tiny Shakespeare

This repository is not a fork and does not copy code verbatim.
It is a learning-driven reimplementation with personal modifications and experimentation.

## Author

Prateek Gupta
IIT Kharagpur

## Interested in:
```
Transformers
Representation learning
Security & anomaly detection
Applied ML systems
```
