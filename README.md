# Transformer Language Model From Scratch

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A complete implementation of a GPT-style decoder-only Transformer language model built entirely from first principles in PyTorch.

This project recreates the core architectural components introduced in the Transformer paper and adapts them for autoregressive language modeling using causal self-attention. The model is trained on the Tiny Shakespeare dataset and generates character-level text in Shakespearean style.

---

## Overview

Modern large language models are built upon the Transformer architecture. This project implements the foundational components of these systems without relying on high-level transformer libraries.

Key objectives include:

* Building self-attention from scratch
* Implementing multi-head attention and residual connections
* Applying causal masking for autoregressive generation
* Training a decoder-only language model
* Generating coherent Shakespeare-style text

The implementation serves as an educational and engineering-focused exploration of how GPT-style language models function internally.

---

## Project Highlights

* Implemented a complete decoder-only Transformer architecture from scratch
* Built causal self-attention and multi-head attention modules manually
* Developed token and positional embedding layers
* Trained a 3.2M parameter language model on Tiny Shakespeare
* Added configurable text generation with temperature sampling
* Optimized training and inference for Apple Silicon and CUDA devices

---

## Architecture

### Decoder-Only Transformer

The model follows a GPT-style decoder architecture composed of:

1. Token Embeddings
2. Positional Embeddings
3. Transformer Blocks
4. Layer Normalization
5. Language Modeling Head

Unlike encoder-based architectures, the model uses causal masking to ensure that tokens can only attend to previous positions during training and inference.

---

### Multi-Head Self-Attention

The attention mechanism projects input representations into:

* Query (Q)
* Key (K)
* Value (V)

Multiple attention heads operate in parallel before being concatenated and projected back into the model dimension.

This allows the model to capture different contextual relationships simultaneously.

---

### Causal Attention Masking

A lower-triangular mask prevents information leakage from future tokens.

```text
Token 1 → attends to Token 1
Token 2 → attends to Tokens 1–2
Token 3 → attends to Tokens 1–3
...
```

This transforms the architecture into an autoregressive language model suitable for text generation.

---

### Position-Wise Feed Forward Network

Each Transformer block contains a feed-forward network:

```text
256 → 512 → 256
```

This enables nonlinear feature transformation after attention computation.

---

### Residual Connections & Layer Normalization

Each sublayer uses:

* Residual skip connections
* Layer Normalization
* Dropout regularization

These components improve optimization stability and gradient flow during training.

---

## Model Configuration

| Parameter              | Value     |
| ---------------------- | --------- |
| Vocabulary Size        | 65        |
| Context Length         | 128       |
| Embedding Dimension    | 256       |
| Attention Heads        | 8         |
| Transformer Layers     | 6         |
| Feed Forward Dimension | 512       |
| Dropout                | 0.1       |
| Total Parameters       | 3,195,969 |

---

## Parameter Breakdown

### Embeddings

| Component             | Parameters |
| --------------------- | ---------- |
| Token Embeddings      | 16,640     |
| Positional Embeddings | 1,280,000  |

### Transformer Stack

Approximately 1.88 million parameters across:

* Multi-head attention layers
* Feed-forward networks
* Layer normalization layers

### Output Projection

```text
256 × 65 + 65 = 16,705 parameters
```

---

## Repository Structure

| File                             | Description                                                            |
| -------------------------------- | ---------------------------------------------------------------------- |
| `transformer.py`                 | Core Transformer implementation including attention and encoder blocks |
| `train_lm.py`                    | Training pipeline and language model wrapper                           |
| `generate.py`                    | Text generation utility with CLI support                               |
| `shakespeare_model_epoch_10.pth` | Saved model checkpoint                                                 |
| `overnight_training_log.txt`     | Training logs and metrics                                              |

---

## Installation

### Create a Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### Install Dependencies

```bash
pip install torch
```

---

## Dataset

The model is trained on the Tiny Shakespeare dataset, a widely used benchmark for character-level language modeling.

The dataset is downloaded automatically during training:

```text
shakespeare.txt
```

---

## Training

Train the model from scratch using:

```bash
python train_lm.py
```

### Training Workflow

1. Download and preprocess Tiny Shakespeare
2. Build vocabulary mappings
3. Create input-target training sequences
4. Train the Transformer language model
5. Save checkpoints periodically

Generated checkpoints:

```text
checkpoints/
├── model_epoch_10.pth
├── model_epoch_20.pth
└── ...
```

---

## Text Generation

Generate text using a trained checkpoint:

```bash
python generate.py \
  --prompt "ROMEO:\n" \
  --length 250 \
  --temperature 0.8
```

### Interactive Mode

```bash
python generate.py \
  --length interactive \
  --temperature 0.85
```

Users can continuously provide prompts and receive generated text.

Type:

```text
exit
```

to terminate the session.

---

## Generation Parameters

| Argument        | Description                      |
| --------------- | -------------------------------- |
| `--checkpoint`  | Path to model checkpoint         |
| `--prompt`      | Initial text prompt              |
| `--length`      | Number of characters to generate |
| `--temperature` | Sampling temperature             |

### Temperature Effects

| Temperature | Behavior                     |
| ----------- | ---------------------------- |
| 0.5         | Conservative and predictable |
| 0.8         | Balanced generation          |
| 1.0+        | More diverse and creative    |

---

## Sample Output

Prompt:

```text
ROMEO:
```

Generated text:

```text
ROMEO:
I am a sorry that are a foul grace,
That stood so far from me to make a merrial kind of
Thy son present poor into his master.

First Senator:
Have you done the third to try him upon him;
This desires of wras spirit.
```

The model learns:

* Character-level syntax
* Dialogue structure
* Speaker formatting
* Shakespearean vocabulary patterns

without any explicit grammar rules.

---

## Technical Components

### Implemented From Scratch

* Scaled Dot-Product Attention
* Multi-Head Self-Attention
* Positional Embeddings
* Layer Normalization
* Residual Connections
* Feed Forward Networks
* Causal Attention Masks
* Autoregressive Sampling

No transformer libraries were used for model construction.

---

## Hardware Support

Automatic device selection:

```python
mps      # Apple Silicon
cuda     # NVIDIA GPU
cpu      # CPU fallback
```

This enables efficient training on:

* MacBook Air/Pro (M-Series)
* Mac Mini
* NVIDIA GPU systems
* Standard CPU environments

---

## Future Improvements

Potential extensions include:

* Byte Pair Encoding (BPE) tokenization
* Rotary Positional Embeddings (RoPE)
* Flash Attention
* Mixed Precision Training
* KV Cache Optimization
* Larger context windows
* GPT-2 style architecture
* Transformer decoder blocks instead of adapted encoder blocks

---

## Technology Stack

* Python
* PyTorch
* Transformer Architecture
* Self-Attention
* Deep Learning
* Natural Language Processing
* Generative AI

---

## Learning Outcomes

This project demonstrates understanding of:

* Transformer internals
* Attention mechanisms
* Language model training
* Sequence modeling
* Neural network optimization
* Text generation systems

---

## License

This project is released under the MIT License.

See the `LICENSE` file for additional information.

---

## References

* Attention Is All You Need (Vaswani et al., 2017)
* GPT: Improving Language Understanding by Generative Pre-Training
* PyTorch Documentation
* Tiny Shakespeare Dataset
