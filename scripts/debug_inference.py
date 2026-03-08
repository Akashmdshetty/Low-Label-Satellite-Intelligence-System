import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms, datasets
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.backbone import get_backbone

def test_inference(image_path, model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Model Setup (Parity with app.py and training)
    backbone, feature_dim = get_backbone("resnet18")
    classifier = nn.Sequential(
        nn.Flatten(),
        nn.Linear(feature_dim, 10)
    )
    
    # Load checkpoint
    if not os.path.exists(model_path):
        print(f"Error: Checkpoint {model_path} not found.")
        return

    checkpoint = torch.load(model_path, map_location=device)
    backbone.load_state_dict(checkpoint['backbone'], strict=True)
    classifier.load_state_dict(checkpoint['classifier'], strict=True)
    
    model = nn.Sequential(backbone, classifier).to(device)
    model.eval()

    # 2. Preprocessing (Parity with training)
    inference_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 3. Load Image
    image = Image.open(image_path).convert("RGB")
    input_tensor = inference_transform(image).unsqueeze(0).to(device)

    print(f"Input tensor shape: {input_tensor.shape}")
    print(f"Input tensor min: {input_tensor.min().item():.4f}")
    print(f"Input tensor max: {input_tensor.max().item():.4f}")

    # 4. Inference
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)

    # 5. Get Class Names
    data_path = r"C:\Users\aakas\Downloads\EuroSAT\2750" # From user's path structure
    class_names = datasets.ImageFolder(data_path).classes
    
    print("\nClass Probabilities:")
    for i, prob in enumerate(probs[0]):
        print(f"{class_names[i]:<20}: {prob.item():.4f}")

    pred_idx = torch.argmax(probs, dim=1).item()
    print(f"\nPredicted Class: {class_names[pred_idx]} (Index: {pred_idx})")

if __name__ == "__main__":
    img_path = r"C:\Users\aakas\Downloads\EuroSAT\2750\Industrial\Industrial_393.jpg"
    model_path = "finetune_model.pth"
    test_inference(img_path, model_path)
