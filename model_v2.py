import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
from infra_2d import TransformerDecoder2D
from config import ACTION_SIZE, SEQ_LEN, D_MODEL, NUM_LAYERS, NUM_HEADS, D_FF, DROPOUT, OUTPUT_SIZE, MAX_DISTANCE

_DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "model_epoch_7.pth")

def load_base_model(model_path=None):
    if model_path is None:
        model_path = _DEFAULT_MODEL_PATH
    """
    Load the base TransformerDecoder2D model
    """
    # Model architecture parameters (from shared config)
    action_size = ACTION_SIZE
    seq_len     = SEQ_LEN
    d_model     = D_MODEL
    num_layers  = NUM_LAYERS
    num_heads   = NUM_HEADS
    d_ff        = D_FF
    dropout     = DROPOUT
    output_size = OUTPUT_SIZE
    
    # Set device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Metal GPU")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA GPU")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
    # Create model
    model = TransformerDecoder2D(
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        dropout=dropout,
        action_size=action_size,            
        seq_len=seq_len,
        output_size=output_size,
        max_distance=MAX_DISTANCE,
        use_causal_mask=False    
    ).to(device)
    
    # Load weights
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    print(f"Successfully loaded model from {model_path}")
    
    return model, device

if __name__ == "__main__":
    model, device = load_base_model()
    print("Model successfully loaded and ready for use")
