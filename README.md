# Transformer From Scratch

A raw, educational implementation of a Generative Pre-trained Transformer (GPT) style decoder-only language model built completely from scratch using PyTorch. The model is trained on the Tiny Shakespeare dataset to generate text in the style of William Shakespeare.

It leverages Apple Silicon GPU acceleration (`mps`) where available, allowing fast training and inference on modern Macs (like Mac Mini M4).

---

## Features

- **Custom Transformer Architecture**: Implemented self-attention, multi-head attention, layer normalization, positional embedding, and causal masking from first principles.
- **Decoder-Only via Causal Masking**: Adapts an Encoder architecture to act as a autoregressive decoder (GPT) using a lower-triangular causal attention mask.
- **Mac-Optimized**: Automatically detects and runs on Apple Silicon GPU (`mps`) or CUDA.
- **Interactive Generation CLI**: An easy-to-use CLI supporting custom starting prompts, length options, temperature controls, and an interactive prompt loop.
- **Graceful Input Handling**: Automatically filters out characters not present in the Tiny Shakespeare training vocabulary.

---

## File Structure

- `transformer.py` - Core implementation of the transformer components:
  - Multi-Head Self Attention
  - Position-Wise Feed-Forward Networks
  - Positional Encoding
  - Transformer Encoder Layer & Encoder Stack
- `train_lm.py` - Script for downloading the Tiny Shakespeare dataset, defining the `LanguageModel` wrapper, and running the training loop.
- `generate.py` - Interactive and CLI-based text generator utility.
- `shakespeare_model_epoch_10.pth` - Saved model checkpoint at epoch 10.
- `overnight_training_log.txt` - Output log from the overnight training session.

---

## Installation & Setup

1. **Activate Virtual Environment**:
   ```bash
   source .venv/bin/activate
   ```
   *(Or install requirements manually in your preferred environment: `pip install torch`)*

2. **Download Dataset (Automatic)**:
   The first time you run training, it will automatically download the Tiny Shakespeare text file (`shakespeare.txt`).

---

## Usage

### 1. Generating Text (Inference)

You can run text generation using the trained checkpoint (`shakespeare_model_epoch_10.pth`).

#### Run with a specific prompt:
```bash
python generate.py --prompt "ROMEO:\n" --length 250 --temperature 0.8
```

#### Run in interactive mode:
```bash
python generate.py --length interactive --temperature 0.85
```
*In interactive mode, type your prompt and press Enter to generate. Type `exit` to quit.*

#### Generation Parameters:
- `--checkpoint`: Path to the `.pth` model file (default: `shakespeare_model_epoch_10.pth`).
- `--prompt`: Starting text (default: `"ROMEO:\n"`).
- `--length`: Number of characters to generate (or `"interactive"`).
- `--temperature`: Creativity multiplier. Higher = more creative/random, lower = more deterministic.

---

### 2. Training the Model

To train the model from scratch or resume training:
```bash
python train_lm.py
```
This will train the language model on the Tiny Shakespeare text and save model checkpoints every 10 epochs.
