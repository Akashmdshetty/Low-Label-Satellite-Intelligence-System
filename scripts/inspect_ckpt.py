import torch
import os

def inspect_checkpoint(path):
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return
    
    print(f"--- Inspecting {path} ---")
    checkpoint = torch.load(path, map_location='cpu')
    print(f"Keys: {list(checkpoint.keys())}")
    
    if 'backbone' in checkpoint:
        print("Found 'backbone' key.")
        print(f"Backbone keys sample: {list(checkpoint['backbone'].keys())[:5]}")
    elif 'model_state_dict' in checkpoint:
        print("Found 'model_state_dict' key.")
        print(f"Model state dict keys sample: {list(checkpoint['model_state_dict'].keys())[:5]}")
    else:
        print("No standard keys found. Printing top-level keys sample:")
        print(f"Sample keys: {list(checkpoint.keys())[:10]}")

if __name__ == "__main__":
    inspect_checkpoint('last_checkpoint.pth')
    inspect_checkpoint('finetune_model.pth')
