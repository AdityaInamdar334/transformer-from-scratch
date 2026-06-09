import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import urllib.request
import os
import time
from transformer import Encoder

# --- 1. Dataset Preparation ---
# Download Tiny Shakespeare dataset
DATA_FILE = "shakespeare.txt"
if not os.path.exists(DATA_FILE):
    print("Downloading Tiny Shakespeare dataset...")
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    urllib.request.urlretrieve(url, DATA_FILE)

with open(DATA_FILE, 'r', encoding='utf-8') as f:
    text = f.read()

# Create Vocabulary
chars = sorted(list(set(text)))
vocab_size = len(chars)
char2idx = {ch: i for i, ch in enumerate(chars)}
idx2char = {i: ch for i, ch in enumerate(chars)}

print(f"Dataset size: {len(text)} characters")
print(f"Vocabulary size: {vocab_size} unique characters")

class ShakespeareDataset(Dataset):
    def __init__(self, text, seq_len):
        self.seq_len = seq_len
        # We need seq_len + 1 characters to form X (input) and Y (target)
        self.num_samples = len(text) - seq_len
        
        # Convert entire text to tensor once for efficiency
        self.data = torch.tensor([char2idx[ch] for ch in text], dtype=torch.long)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        chunk = self.data[idx:idx + self.seq_len + 1]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y

# --- 2. Model Architecture ---
# A Generative Pre-trained Transformer (GPT) is essentially a Decoder-only model.
# Since we don't have an Encoder to attend to, we can actually just use our 
# 'Encoder' module from transformer.py but apply a CAUSAL MASK to the self-attention
# so it can't look ahead at future tokens. This makes it function identically to a GPT!
class LanguageModel(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff, dropout, max_len=5000):
        super(LanguageModel, self).__init__()
        # We use the Encoder module as our causal stack
        self.stack = Encoder(vocab_size, d_model, num_layers, num_heads, d_ff, dropout, max_len)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def create_causal_mask(self, seq_len, device):
        # Creates a lower triangular matrix of Trues
        # This prevents token 'i' from attending to any token > 'i'
        mask = torch.tril(torch.ones((seq_len, seq_len), device=device)).bool()
        # Shape needs to be broadcastable: (batch_size=1, num_heads=1, seq_len, seq_len)
        mask = mask.unsqueeze(0).unsqueeze(0)
        return mask

    def forward(self, x):
        seq_len = x.size(1)
        mask = self.create_causal_mask(seq_len, x.device)
        
        # Pass through the transformer stack
        out = self.stack(x, mask)
        
        # Output vocabulary probabilities
        logits = self.fc_out(out)
        return logits

# --- 3. Text Generation Function ---
def generate_text(model, start_str, generate_len=200, device='cpu'):
    model.eval()
    tokens = [char2idx[ch] for ch in start_str]
    x = torch.tensor(tokens).unsqueeze(0).to(device)
    
    print("\n--- GENERATED TEXT ---")
    print(start_str, end="")
    
    with torch.no_grad():
        for _ in range(generate_len):
            # Truncate input if it exceeds seq_len to avoid confusing the model
            x_cond = x[:, -128:] # We'll match the seq_len we train with
            
            logits = model(x_cond)
            # Get the logits for the very last character
            next_token_logits = logits[:, -1, :]
            
            # Apply softmax to get probabilities
            probs = torch.softmax(next_token_logits, dim=-1)
            
            # Sample from the distribution (makes generation more creative/natural)
            next_token = torch.multinomial(probs, num_samples=1)
            
            print(idx2char[next_token.item()], end="", flush=True)
            
            # Append to our sequence
            x = torch.cat((x, next_token), dim=1)
            
    print("\n----------------------\n")
    model.train()

# --- 4. Training Setup ---
def train():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Metal Performance Shaders) on Mac Mini M4!")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # Hyperparameters for an OVERNIGHT RUN
    # Increased sequence length so it remembers longer context for better grammar
    seq_len = 128
    batch_size = 64
    epochs = 100  # Will take hours, perfect for overnight
    d_model = 256 # Doubled the brain size
    num_layers = 6 # Deeper network
    num_heads = 8
    d_ff = 512
    dropout = 0.1
    learning_rate = 0.001

    # Train on the ENTIRE book
    dataset = ShakespeareDataset(text, seq_len)
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = LanguageModel(
        vocab_size=vocab_size,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        dropout=dropout
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print("Starting generation before training (Expect Gibberish):")
    generate_text(model, start_str="ROMEO:\n", generate_len=100, device=device)

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        start_time = time.time()
        
        for batch_idx, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            logits = model(x)
            
            # Reshape for CrossEntropyLoss which expects (batch_size * seq_len, vocab_size)
            logits = logits.view(-1, vocab_size)
            y = y.view(-1)
            
            loss = criterion(logits, y)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
            
            # Print intermediate generation
            if batch_idx > 0 and batch_idx % 200 == 0:
                print(f"\n[Epoch {epoch+1}, Batch {batch_idx}] Intermediate generation:")
                generate_text(model, start_str="\nROMEO:\n", generate_len=100, device=device)
            
        end_time = time.time()
        avg_loss = epoch_loss / len(dataloader)
        print(f"\nEpoch: {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Time: {end_time - start_time:.2f}s")
        
        # Save a checkpoint every 10 epochs just in case!
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"shakespeare_model_epoch_{epoch+1}.pth")

    print("\nTraining complete! Final generation:")
    generate_text(model, start_str="\nROMEO:\n", generate_len=500, device=device)
    
    # Save the final model
    torch.save(model.state_dict(), "shakespeare_model_final.pth")
    print("Model saved to shakespeare_model_final.pth!")
    
    return model, device

if __name__ == "__main__":
    train()
