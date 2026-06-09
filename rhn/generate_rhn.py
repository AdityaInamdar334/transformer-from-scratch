import torch
import torch.nn as nn
import os
import argparse
from rhn import RecurrentHighwayNetwork

# --- 1. Load Data & Build Vocabulary ---
DATA_FILE = "../shakespeare.txt"
if not os.path.exists(DATA_FILE):
    DATA_FILE = "shakespeare.txt"

if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(f"Could not find {DATA_FILE}. Please run training or download it first.")

with open(DATA_FILE, 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
char2idx = {ch: i for i, ch in enumerate(chars)}
idx2char = {i: ch for i, ch in enumerate(chars)}

# --- 2. RHN Language Model Wrapper ---
class RHNLanguageModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, recurrence_depth):
        super(RHNLanguageModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rhn = RecurrentHighwayNetwork(embedding_dim, hidden_size, recurrence_depth)
        self.fc_out = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, h_prev=None):
        embedded = self.embedding(x)
        out, h_next = self.rhn(embedded, h_prev)
        logits = self.fc_out(out)
        return logits, h_next

# --- 3. Text Generation Function ---
def generate_text(model, start_str, generate_len=500, temperature=1.0, device='cpu'):
    model.eval()
    tokens = [char2idx[ch] for ch in start_str if ch in char2idx]
    if not tokens:
        tokens = [char2idx.get(' ', 0)]
    
    x = torch.tensor(tokens).unsqueeze(0).to(device)
    
    print("\n--- GENERATED TEXT ---")
    print(start_str, end="")
    
    h = None
    
    with torch.no_grad():
        logits, h = model(x, h)
        next_token_logits = logits[:, -1, :] / temperature
        probs = torch.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        char = idx2char[next_token.item()]
        print(char, end="", flush=True)
        
        current_token = next_token
        for _ in range(generate_len - 1):
            logits, h = model(current_token, h)
            next_token_logits = logits[:, -1, :] / temperature
            probs = torch.softmax(next_token_logits, dim=-1)
            current_token = torch.multinomial(probs, num_samples=1)
            print(idx2char[current_token.item()], end="", flush=True)
            
    print("\n----------------------\n")

def main():
    parser = argparse.ArgumentParser(description="Generate text using the trained Recurrent Highway Network.")
    parser.add_argument("--checkpoint", type=str, default="rhn_model_epoch_1.pth", help="Path to the model checkpoint (.pth file)")
    parser.add_argument("--prompt", type=str, default="ROMEO:\n", help="Starting text prompt for generation")
    parser.add_argument("--length", type=str, default="500", help="Number of characters to generate (or 'interactive')")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    args = parser.parse_args()

    # Determine device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Model Hyperparameters
    embedding_dim = 128
    hidden_size = 256
    recurrence_depth = 4

    # Initialize model
    model = RHNLanguageModel(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_size=hidden_size,
        recurrence_depth=recurrence_depth
    ).to(device)

    # Load weights
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint '{args.checkpoint}' not found.")
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
