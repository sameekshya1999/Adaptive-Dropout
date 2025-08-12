# ARB-Dropout: Adaptive Gradient-Based Dropout for Uncertainty Estimation

PyTorch implementation of ARB-Dropout, an adaptive dropout method that dynamically adjusts rates via gradient variance for efficient uncertainty estimation in deep neural networks.

## Key Features
- 🚀 **20-50x faster** than MC-Dropout at test time
- 📊 Adaptive dropout rates based on gradient variance
- 🔍 Combined epistemic + aleatoric uncertainty estimation
- 🏆 Benchmarked on CIFAR-10/100, SVHN, and STL-10

## Installation
```bash
git clone https://github.com/sameekshya1999/Adaptive-Dropout.git
cd Adaptive-Dropout
pip install -r requirements.txt
