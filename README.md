# Vertical Federated Learning (VFL) Implementation

This repository contains implementations of Vertical Federated Learning (VFL), including security analysis components, inspired by the framework described in:  
**["Vertical Federated Learning: Challenges, Methodologies and Experiments"](https://arxiv.org/pdf/2202.04309)**  
*Wei et al., 2022*

## 📌 Key Features

### 1. Core Implementations

#### `final_vfl.py` (Base Implementation)
- Implements core VFL workflow from the paper:
  - Bottom models for each client
  - Top model for label owner
  - Federated training with backward propagation
- **Differences from paper**:
  - Simplified gradient flow (only label owner updates top model)
  - No PSI (Private Set Intersection) or differential privacy

#### `2_vfl_modified_debug.py` (Extended Implementation)
- Adds diagnostic tools:
  - Gradient norm tracking for both clients
  - Optional reconstruction loss for feature learning
- Closer to paper's methodology:
  - Both clients participate in backpropagation
  - Supports gradient analysis for imbalance detection

### 2. Security Analysis

#### `3_fia_vfl.py` (Feature Inference Attack)
- Implements state-of-the-art feature inference attacks against VFL:
  - **Dual-model architecture**: 
    - Discriminator network to detect genuine embeddings
    - Reconstructor network to recover private features
  - Comprehensive evaluation metrics:
    - AUC-ROC for membership inference
    - Reconstruction MSE for feature leakage
    - Full classification reports
  - Visualization tools:
    - Embedding space analysis (PCA plots)
    - Training progress tracking

## 🛠️ Setup & Usage

```bash
git clone https://github.com/your_username/vfl-implementation.git
cd FIA_VFL

# Run base implementation
python final_vfl.py

# Run extended version with gradient tracking
python 2_vfl_modified_debug.py

# Launch feature inference attack analysis
python 3_fia_vfl.py
