import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import time
from rhn import RecurrentHighwayNetwork

# --- 1. Dataset Preparation ---
DATA_FILE = "../shakespeare.txt"
if not os.path.exists(DATA_FILE):
    DATA_FILE = "shakespeare.txt"

if not os.path.exists(DATA_FILE):
    raise FileNotFoundError("Could not find shakespeare.txt. Please ensure it is present in the workspace.")

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
        self.num_samples = len(text) - seq_len
        self.data = torch.tensor([char2idx[ch] for ch in text], dtype=torch.long)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        chunk = self.data[idx:idx + self.seq_len + 1]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y

# --- 2. RHN Language Model Wrapper ---
class RHNLanguageModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, recurrence_depth):
        super(RHNLanguageModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rhn = RecurrentHighwayNetwork(embedding_dim, hidden_size, recurrence_depth)
        self.fc_out = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, h_prev=None):
        # x shape: (batch_size, seq_len)
        embedded = self.embedding(x)  # shape: (batch_size, seq_len, embedding_dim)
        out, h_next = self.rhn(embedded, h_prev)
        logits = self.fc_out(out)     # shape: (batch_size, seq_len, vocab_size)
        return logits, h_next

# --- 3. Text Generation Function ---
def generate_text(model, start_str, generate_len=200, temperature=1.0, device='cpu'):
    model.eval()
    tokens = [char2idx[ch] for ch in start_str if ch in char2idx]
    if not tokens:
        tokens = [char2idx.get(' ', 0)]
    
    x = torch.tensor(tokens).unsqueeze(0).to(device)
    
    print("\n--- GENERATED TEXT ---")
    print(start_str, end="")
    
    h = None
    
    with torch.no_grad():
        # First process the start prompt to build the hidden state
        # For sequence inputs, model returns outputs and the final hidden state
        logits, h = model(x, h)
        next_token_logits = logits[:, -1, :] / temperature
        probs = torch.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        char = idx2char[next_token.item()]
        print(char, end="", flush=True)
        
        # Now feed tokens step-by-step
        current_token = next_token
        for _ in range(generate_len - 1):
            logits, h = model(current_token, h)
            next_token_logits = logits[:, -1, :] / temperature
            probs = torch.softmax(next_token_logits, dim=-1)
            current_token = torch.multinomial(probs, num_samples=1)
            print(idx2char[current_token.item()], end="", flush=True)
            
    print("\n----------------------\n")
    model.train()

# --- 4. Training Loop ---
def train():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS GPU acceleration!")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # Hyperparameters
    seq_len = 128
    batch_size = 64
    epochs = 10
    embedding_dim = 128
    hidden_size = 256
    recurrence_depth = 4
    learning_rate = 0.001

    dataset = ShakespeareDataset(text, seq_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = RHNLanguageModel(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_size=hidden_size,
        recurrence_depth=recurrence_depth
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print("Initial generation (Expect gibberish):")
    generate_text(model, start_str="ROMEO:\n", generate_len=100, device=device)

    for epoch in range(epochs):
        epoch_loss = 0
        start_time = time.time()
        
        for batch_idx, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            logits, _ = model(x)
            
            # Reshape for CrossEntropyLoss
            logits = logits.view(-1, vocab_size)
            y = y.view(-1)
            
            loss = criterion(logits, y)
            loss.backward()
            
            # Gradient clipping is crucial for recurrent networks to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            
            optimizer.step()
            epoch_loss += loss.item()
            
            if batch_idx > 0 and batch_idx % 500 == 0:
                print(f"\n[Epoch {epoch+1}, Batch {batch_idx}] Intermediate generation:")
                generate_text(model, start_str="\nROMEO:\n", generate_len=100, device=device)
            
        end_time = time.time()
        avg_loss = epoch_loss / len(dataloader)
        print(f"\nEpoch: {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Time: {end_time - start_time:.2f}s")
        
        # Save checkpoint
        torch.save(model.state_dict(), f"rhn_model_epoch_{epoch+1}.pth")

    print("\nTraining complete! Final generation:")
    generate_text(model, start_str="\nROMEO:\n", generate_len=400, device=device)
    
    torch.save(model.state_dict(), "rhn_model_final.pth")
    print("Model saved to rhn_model_final.pth!")

if __name__ == "__main__":
    train()
