import torch
import torch.nn as nn
import os
import argparse
from transformer import Encoder

# --- 1. Load Data & Build Vocabulary ---
DATA_FILE = "shakespeare.txt"
if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(f"Could not find {DATA_FILE}. Please run training or download it first.")

with open(DATA_FILE, 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
char2idx = {ch: i for i, ch in enumerate(chars)}
idx2char = {i: ch for i, ch in enumerate(chars)}

# --- 2. Model Architecture ---
class LanguageModel(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff, dropout, max_len=5000):
        super(LanguageModel, self).__init__()
        self.stack = Encoder(vocab_size, d_model, num_layers, num_heads, d_ff, dropout, max_len)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def create_causal_mask(self, seq_len, device):
        mask = torch.tril(torch.ones((seq_len, seq_len), device=device)).bool()
        mask = mask.unsqueeze(0).unsqueeze(0)
        return mask

    def forward(self, x):
        seq_len = x.size(1)
        mask = self.create_causal_mask(seq_len, x.device)
        out = self.stack(x, mask)
        logits = self.fc_out(out)
        return logits

# --- 3. Text Generation Function ---
def generate_text(model, start_str, generate_len=500, temperature=1.0, device='cpu'):
    model.eval()
    tokens = [char2idx[ch] for ch in start_str if ch in char2idx]
    if not tokens:
        tokens = [char2idx.get(' ', 0)]
    x = torch.tensor(tokens).unsqueeze(0).to(device)
    
    print("\n--- GENERATED TEXT ---")
    print(start_str, end="")
    
    with torch.no_grad():
        for _ in range(generate_len):
            # Truncate input if it exceeds seq_len (128)
            x_cond = x[:, -128:]
            
            logits = model(x_cond)
            # Get the logits for the very last character
            next_token_logits = logits[:, -1, :] / temperature
            
            # Apply softmax to get probabilities
            probs = torch.softmax(next_token_logits, dim=-1)
            
            # Sample from the distribution
            next_token = torch.multinomial(probs, num_samples=1)
            
            print(idx2char[next_token.item()], end="", flush=True)
            
            # Append to our sequence
            x = torch.cat((x, next_token), dim=1)
            
    print("\n----------------------\n")

def main():
    parser = argparse.ArgumentParser(description="Generate text using the trained Transformer model.")
    parser.add_argument("--checkpoint", type=str, default="shakespeare_model_epoch_10.pth", help="Path to the model checkpoint (.pth file)")
    parser.add_argument("--prompt", type=str, default="ROMEO:\n", help="Starting text prompt for generation")
    parser.add_argument("--length", type=str, default="500", help="Number of characters to generate (or 'interactive')")
    parser.add_argument("--temperature", type=float, default=0.8, help="Temperature for sampling (higher = more creative/random, lower = more deterministic)")
    args = parser.parse_args()

    # Determine device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Model Hyperparameters matching training
    d_model = 256
    num_layers = 6
    num_heads = 8
    d_ff = 512
    dropout = 0.1

    # Initialize model
    model = LanguageModel(
        vocab_size=vocab_size,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        dropout=dropout
    ).to(device)

    # Load weights
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint '{args.checkpoint}' not found.")
        print("Please check the filename or run training to generate a checkpoint.")
        return

    print(f"Loading checkpoint from {args.checkpoint}...")
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    print("Model loaded successfully!")

    # Fix escape characters in prompt
    prompt_str = args.prompt.replace("\\n", "\n").replace("\\t", "\t")

    # Perform generation
    if args.length == "interactive":
        print("\nEntering Interactive Mode! Type 'exit' or Ctrl+C to quit.")
        while True:
            try:
                prompt = input("\nEnter prompt (or press Enter for default 'ROMEO:\\n'): ")
                if prompt.strip().lower() == "exit":
                    break
                if not prompt:
                    prompt = "ROMEO:\n"
                else:
                    prompt = prompt.replace("\\n", "\n").replace("\\t", "\t")
                length = input("Enter length to generate (default 300): ")
                try:
                    length = int(length) if length.strip() else 300
                except ValueError:
                    length = 300
                
                generate_text(model, start_str=prompt, generate_len=length, temperature=args.temperature, device=device)
            except KeyboardInterrupt:
                break
    else:
        try:
            length = int(args.length)
        except ValueError:
            length = 500
        generate_text(model, start_str=prompt_str, generate_len=length, temperature=args.temperature, device=device)

if __name__ == "__main__":
    main()
