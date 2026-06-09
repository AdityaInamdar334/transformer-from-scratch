import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random
import time
from transformer import Transformer

# --- 1. Dataset Preparation ---
# We create a simple synthetic task: 2-digit addition
# Example: Input "15+27=" -> Output "42"

VOCAB = ['<pad>', '<sos>', '<eos>', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '=']
char2idx = {ch: i for i, ch in enumerate(VOCAB)}
idx2char = {i: ch for i, ch in enumerate(VOCAB)}

PAD_IDX = char2idx['<pad>']
SOS_IDX = char2idx['<sos>']
EOS_IDX = char2idx['<eos>']

class AdditionDataset(Dataset):
    def __init__(self, num_samples):
        self.num_samples = num_samples
        self.data = []
        for _ in range(num_samples):
            a = random.randint(0, 99)
            b = random.randint(0, 99)
            input_str = f"{a}+{b}="
            output_str = f"{a+b}"
            self.data.append((input_str, output_str))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        input_str, output_str = self.data[idx]
        
        # Tokenize input and add SOS/EOS
        input_tokens = [SOS_IDX] + [char2idx[ch] for ch in input_str] + [EOS_IDX]
        # Tokenize output and add SOS/EOS
        output_tokens = [SOS_IDX] + [char2idx[ch] for ch in output_str] + [EOS_IDX]
        
        return torch.tensor(input_tokens), torch.tensor(output_tokens)

def collate_fn(batch):
    # Padding sequences in a batch
    src_batch, tgt_batch = zip(*batch)
    
    src_batch = nn.utils.rnn.pad_sequence(src_batch, padding_value=PAD_IDX, batch_first=True)
    tgt_batch = nn.utils.rnn.pad_sequence(tgt_batch, padding_value=PAD_IDX, batch_first=True)
    
    return src_batch, tgt_batch

# --- 2. Training Setup ---

def train():
    # Check for Mac M4 MPS support, otherwise use CPU/CUDA
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Metal Performance Shaders) on Mac Mini M4!")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # Hyperparameters
    num_samples = 10000
    batch_size = 64
    epochs = 15
    d_model = 128
    num_layers = 2
    num_heads = 4
    d_ff = 256
    dropout = 0.1
    learning_rate = 0.001

    dataset = AdditionDataset(num_samples)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    model = Transformer(
        src_vocab_size=len(VOCAB),
        tgt_vocab_size=len(VOCAB),
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        dropout=dropout,
        pad_idx=PAD_IDX
    ).to(device)

    # Ignore padding index when calculating loss
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # --- 3. Training Loop ---
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        start_time = time.time()
        
        for batch_idx, (src, tgt) in enumerate(dataloader):
            src, tgt = src.to(device), tgt.to(device)
            
            # The target for the model input is the target sequence without the last token
            # The target for the loss is the target sequence without the first token (<sos>)
            tgt_input = tgt[:, :-1]
            tgt_expected = tgt[:, 1:]
            
            optimizer.zero_grad()
            
            # Forward pass
            output = model(src, tgt_input)
            
            # output shape: (batch_size, seq_len, vocab_size)
            # tgt_expected shape: (batch_size, seq_len)
            # Reshape for CrossEntropyLoss which expects (batch_size * seq_len, vocab_size)
            output = output.reshape(-1, output.shape[-1])
            tgt_expected = tgt_expected.reshape(-1)
            
            loss = criterion(output, tgt_expected)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            epoch_loss += loss.item()
            
        end_time = time.time()
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch: {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Time: {end_time - start_time:.2f}s")

    print("Training complete!")
    return model, device

# --- 4. Evaluation ---
def evaluate(model, device):
    model.eval()
    test_cases = ["15+27=", "99+1=", "0+0=", "45+55=", "12+34="]
    print("\n--- Evaluation ---")
    
    with torch.no_grad():
        for test_str in test_cases:
            # Tokenize
            tokens = [SOS_IDX] + [char2idx[ch] for ch in test_str] + [EOS_IDX]
            src_tensor = torch.tensor(tokens).unsqueeze(0).to(device)
            
            # Start with SOS token for decoder
            tgt_indices = [SOS_IDX]
            
            for i in range(10): # max length
                tgt_tensor = torch.tensor(tgt_indices).unsqueeze(0).to(device)
                
                output = model(src_tensor, tgt_tensor)
                
                # Get the highest probability token for the last position
                next_token = output.argmax(dim=-1)[:, -1].item()
                tgt_indices.append(next_token)
                
                if next_token == EOS_IDX:
                    break
                    
            # Decode
            result = "".join([idx2char[idx] for idx in tgt_indices if idx not in [SOS_IDX, EOS_IDX]])
            print(f"Input: {test_str}  ->  Predicted Output: {result}")

if __name__ == "__main__":
    model, device = train()
    evaluate(model, device)
