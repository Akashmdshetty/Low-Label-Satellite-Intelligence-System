# SSL BYOL for EuroSAT

Research-level implementation of Bootstrap Your Own Latent (BYOL) for low-label satellite image classification on the EuroSAT dataset.

## Installation
```bash
pip install -r requirements.txt
```

## Structure
- `configs/`: Hyperparameters and paths.
- `data/`: Dataset loading, BYOL augmentations, and stratified splitting.
- `models/`: ResNet18 backbone, Projection, and Prediction MLPs.
- `trainers/`: Pretraining (SSL), Fine-tuning (Low-label), and Linear Probing.
- `evaluation/`: k-NN monitoring and t-SNE visualization.

## Usage

### 1. Pretraining
Train the ResNet18 backbone using self-supervised learning on 100% unlabeled EuroSAT data.
```bash
python scripts/run_pretrain.py --config configs/config.yaml
```

### 2. Fine-tuning (1%, 5%, 10%)
Evaluate the representation by fine-tuning on labeled subsets.
```bash
python scripts/run_finetune.py --config configs/config.yaml --ckpt checkpoint_epoch_90.pth
```

## Self-Supervised Learning Details

### EMA Update Logic
The target network parameters $\xi$ are updated as an exponential moving average (EMA) of the online network parameters $\theta$:
$$\xi \leftarrow \tau \xi + (1 - \tau) \theta$$
The decay rate $\tau$ follows a cosine schedule, starting at $\tau_{base}$ and increasing to 1.

### Loss Computation
BYOL minimizes the mean squared error between the normalized prediction and target representation, which is equivalent to the negative cosine similarity:
$$L = 2 - 2 \cdot \frac{\langle p, z \rangle}{\|p\| \cdot \|z\|}$$

### Why Stop-Gradient?
The target network provides regression targets. If we backpropagated through the target network, the online and target networks would rapidly converge to a constant representation (collapse). Stop-gradient ensures that the target network only evolves via EMA, creating a moving target that the online network must "chase".

### Preventing Collapse
Collapse is prevented by:
1. **Asymmetry**: The online network has an extra predictor head.
2. **Target Evolution**: The target network is updated via EMA, not SGD.
3. **Normalization**: L2 normalization and Batch Norm layers provide implicit contrastive-like effects.
