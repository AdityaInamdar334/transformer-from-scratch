import torch
import torch.nn as nn

class RHNCell(nn.Module):
    def __init__(self, input_size, hidden_size, recurrence_depth=2):
        super(RHNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.recurrence_depth = recurrence_depth

        # Layer 1: processes both input x_t and recurrent state h_{t-1}
        self.w_layer1 = nn.Linear(input_size, 2 * hidden_size)  # Projects x_t to H and T gates
        self.r_layer1 = nn.Linear(hidden_size, 2 * hidden_size)  # Projects h_{t-1}^L to H and T gates

        # Layers 2 to L: only process intermediate hidden state h_t^{l-1}
        self.r_layers = nn.ModuleList([
            nn.Linear(hidden_size, 2 * hidden_size) for _ in range(recurrence_depth - 1)
        ])

    def forward(self, x, h_prev):
        # x shape: (batch_size, input_size)
        # h_prev shape: (batch_size, hidden_size)
        
        # --- Layer 1 ---
        w_proj = self.w_layer1(x)
        r_proj = self.r_layer1(h_prev)
        
        h_proj, t_proj = torch.chunk(w_proj + r_proj, 2, dim=-1)
        
        h_tilde = torch.tanh(h_proj)
        t_gate = torch.sigmoid(t_proj)
        
        # Carry gate C is typically 1 - T
        c_gate = 1.0 - t_gate
        
        h = h_prev * c_gate + h_tilde * t_gate

        # --- Layers 2 to L ---
        for l in range(self.recurrence_depth - 1):
            r_proj_l = self.r_layers[l](h)
            h_proj_l, t_proj_l = torch.chunk(r_proj_l, 2, dim=-1)
            
            h_tilde_l = torch.tanh(h_proj_l)
            t_gate_l = torch.sigmoid(t_proj_l)
            c_gate_l = 1.0 - t_gate_l
            
            h = h * c_gate_l + h_tilde_l * t_gate_l
            
        return h

class RecurrentHighwayNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, recurrence_depth=2):
        super(RecurrentHighwayNetwork, self).__init__()
        self.hidden_size = hidden_size
        self.cell = RHNCell(input_size, hidden_size, recurrence_depth)

    def forward(self, x, h_init=None):
        # x shape: (batch_size, seq_len, input_size)
        batch_size, seq_len, _ = x.size()
        
        if h_init is None:
            h_init = torch.zeros(batch_size, self.hidden_size, device=x.device)
            
        h = h_init
        outputs = []
        
        for t in range(seq_len):
            x_t = x[:, t, :]
            h = self.cell(x_t, h)
            outputs.append(h.unsqueeze(1))
            
        # outputs shape: (batch_size, seq_len, hidden_size)
        outputs = torch.cat(outputs, dim=1)
        return outputs, h
