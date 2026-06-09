import torch
from rhn import RHNCell, RecurrentHighwayNetwork

def test_rhn():
    print("Testing RHN modules...")
    batch_size = 4
    seq_len = 10
    input_size = 16
    hidden_size = 32
    recurrence_depth = 3
    
    # 1. Test RHNCell
    cell = RHNCell(input_size, hidden_size, recurrence_depth)
    x = torch.randn(batch_size, input_size)
    h_prev = torch.randn(batch_size, hidden_size)
    h_next = cell(x, h_prev)
    print(f"RHNCell output shape: {h_next.shape}")
    assert h_next.shape == (batch_size, hidden_size)

    # 2. Test RecurrentHighwayNetwork
    rhn = RecurrentHighwayNetwork(input_size, hidden_size, recurrence_depth)
    seq_x = torch.randn(batch_size, seq_len, input_size)
    outputs, final_h = rhn(seq_x)
    print(f"RecurrentHighwayNetwork outputs shape: {outputs.shape}")
    print(f"RecurrentHighwayNetwork final hidden shape: {final_h.shape}")
    assert outputs.shape == (batch_size, seq_len, hidden_size)
    assert final_h.shape == (batch_size, hidden_size)
    
    print("All tests passed successfully!")

if __name__ == "__main__":
    test_rhn()
