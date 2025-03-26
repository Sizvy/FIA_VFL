# Vertical Federated Learning (VFL) Implementation

This repository contains two implementations of Vertical Federated Learning (VFL), inspired by the framework described in:  
**["Vertical Federated Learning: Challenges, Methodologies and Experiments"](https://arxiv.org/pdf/2202.04309)**  
*Wei et al., 2022*

## 📌 Key Features

### 1. `final_vfl.py` (Base Implementation)
- Implements core VFL workflow from the paper:
  - Bottom models for each client
  - Top model for label owner
  - Federated training with backward propagation
- **Differences from paper**:
  - Simplified gradient flow (only label owner updates top model)
  - No PSI (Private Set Intersection) or differential privacy

### 2. `2_vfl_modified_debug.py` (Extended Implementation)
- Adds diagnostic tools:
  - Gradient norm tracking for both clients
  - Optional reconstruction loss for feature learning
- Closer to paper's methodology:
  - Both clients participate in backpropagation
  - Supports gradient analysis for imbalance detection

## 🛠️ Setup

```bash
git clone https://github.com/your_username/vfl-implementation.git
cd vfl-implementation
# Run base implementatio
python 2_vfl_modified.py
# Run extended version with gradient tracking
python 2_vfl_modified_debug.py
