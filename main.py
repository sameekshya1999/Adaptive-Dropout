

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import time
import numpy as np
from torch.utils.data import random_split

# Device, hyperparameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size, test_batch_size, max_epochs, lr, dropout_p, mc_samples, patience = 64, 1000, 15, 0.001, 0.5, 20, 5

# Data loaders
# CIFAR-10
transform_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
cifar10_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transform_cifar)
cifar10_test = datasets.CIFAR10('./data', train=False, transform=transform_cifar)
cifar10_train_size = int(0.9 * len(cifar10_dataset))
cifar10_val_size = len(cifar10_dataset) - cifar10_train_size
cifar10_train, cifar10_val = random_split(cifar10_dataset, [cifar10_train_size, cifar10_val_size])
cifar10_train_loader = torch.utils.data.DataLoader(cifar10_train, batch_size=batch_size, shuffle=True)
cifar10_val_loader = torch.utils.data.DataLoader(cifar10_val, batch_size=test_batch_size, shuffle=False)
cifar10_test_loader = torch.utils.data.DataLoader(cifar10_test, batch_size=test_batch_size, shuffle=False)

# CIFAR-100
cifar100_dataset = datasets.CIFAR100('./data', train=True, download=True, transform=transform_cifar)
cifar100_test = datasets.CIFAR100('./data', train=False, transform=transform_cifar)
cifar100_train_size = int(0.9 * len(cifar100_dataset))
cifar100_val_size = len(cifar100_dataset) - cifar100_train_size
cifar100_train, cifar100_val = random_split(cifar100_dataset, [cifar100_train_size, cifar100_val_size])
cifar100_train_loader = torch.utils.data.DataLoader(cifar100_train, batch_size=batch_size, shuffle=True)
cifar100_val_loader = torch.utils.data.DataLoader(cifar100_val, batch_size=test_batch_size, shuffle=False)
cifar100_test_loader = torch.utils.data.DataLoader(cifar100_test, batch_size=test_batch_size, shuffle=False)

# SVHN
transform_svhn = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
svhn_dataset = datasets.SVHN('./data', split='train', download=True, transform=transform_svhn)
svhn_test = datasets.SVHN('./data', split='test', download=True, transform=transform_svhn)
svhn_train_size = int(0.9 * len(svhn_dataset))
svhn_val_size = len(svhn_dataset) - svhn_train_size
svhn_train, svhn_val = random_split(svhn_dataset, [svhn_train_size, svhn_val_size])
svhn_train_loader = torch.utils.data.DataLoader(svhn_train, batch_size=batch_size, shuffle=True)
svhn_val_loader = torch.utils.data.DataLoader(svhn_val, batch_size=test_batch_size, shuffle=False)
svhn_test_loader = torch.utils.data.DataLoader(svhn_test, batch_size=test_batch_size, shuffle=False)

# STL-10
transform_stl = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
stl10_dataset = datasets.STL10('./data', split='train', download=True, transform=transform_stl)
stl10_test = datasets.STL10('./data', split='test', download=True, transform=transform_stl)
stl10_train_size = int(0.9 * len(stl10_dataset))
stl10_val_size = len(stl10_dataset) - stl10_train_size
stl10_train, stl10_val = random_split(stl10_dataset, [stl10_train_size, stl10_val_size])
stl10_train_loader = torch.utils.data.DataLoader(stl10_train, batch_size=batch_size, shuffle=True)
stl10_val_loader = torch.utils.data.DataLoader(stl10_val, batch_size=test_batch_size, shuffle=False)
stl10_test_loader = torch.utils.data.DataLoader(stl10_test, batch_size=test_batch_size, shuffle=False)

# Model
class Net(nn.Module):
    def __init__(self, num_classes=10):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.dropout = nn.Dropout(dropout_p)
        self.fc2 = nn.Linear(128, num_classes)
        self.noise_head = nn.Linear(128, 1)

    def forward(self, x, mc=False, adapt=False):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 8 * 8)
        h = F.relu(self.fc1(x))
        p_i = dropout_p
        if adapt:
            with torch.enable_grad():
                h_grad = h.clone().requires_grad_(True)
                logits_temp = self.fc2(h_grad)
                dummy_target = torch.zeros(logits_temp.size(0), dtype=torch.long, device=device)
                loss = F.cross_entropy(logits_temp, dummy_target, reduction='sum')
                grad = torch.autograd.grad(loss, h_grad, create_graph=False)[0]
                var_grad = grad.var(0).mean()
                p_i = torch.clamp(var_grad / (var_grad + 1e-6), 0, 0.5)
            h = h * (1 - p_i)
        elif mc:
            h = self.dropout(h)
        else:
            h = h * (1 - dropout_p)
        logits = self.fc2(h)
        sigma_a = F.softplus(self.noise_head(h))
        return logits, sigma_a, (h, p_i)

# Training with early stopping
def train(model, train_loader, val_loader, name, num_classes):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_val_loss = float('inf')
    patience_counter = 0
    for epoch in range(max_epochs):
        train_loss = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            logits, sigma_a, _ = model(data, mc=True)
            cls_loss = F.cross_entropy(logits, target)
            noise_loss = ((logits.argmax(1) - target.float())**2 / (sigma_a + 1e-6)).mean()
            loss = cls_loss + 0.1 * noise_loss
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # Validation
        val_loss = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                logits, sigma_a, _ = model(data)
                cls_loss = F.cross_entropy(logits, target)
                noise_loss = ((logits.argmax(1) - target.float())**2 / (sigma_a + 1e-6)).mean()
                val_loss += (cls_loss + 0.1 * noise_loss).item()
        val_loss /= len(val_loader)

        print(f"{name} Epoch {epoch+1}: Train Loss {train_loss:.4f}, Val Loss {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"{name} Early stopping at epoch {epoch+1}")
                break
    print(f"{name} trained.")

# MC Dropout test with normalized entropy
def mc_dropout_test(model, loader, name, num_classes):
    model.eval()
    correct = 0
    start_time = time.time()
    uncertainties = []
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            outputs = torch.stack([F.softmax(model(data, mc=True)[0], dim=1) for _ in range(mc_samples)], dim=0)
            mean_pred = outputs.mean(0)
            entropy = -torch.sum(mean_pred * torch.log(mean_pred + 1e-6), dim=1)
            norm_entropy = entropy / np.log(num_classes)  # Normalized to [0,1]
            uncertainties.append(norm_entropy.mean().item())
            pred = mean_pred.argmax(1)
            correct += pred.eq(target).sum().item()
    acc = correct / len(loader.dataset)
    time_taken = time.time() - start_time
    avg_unc = np.mean(uncertainties)
    print(f"MC Dropout on {name}: Acc {acc:.4f}, Norm Avg Unc {avg_unc:.4f}, Time {time_taken:.2f}s")

# ARB-Dropout test with normalized uncertainty (approximated as entropy + normalized variance)
def arb_dropout_test(model, loader, name, num_classes):
    model.eval()
    correct = 0
    start_time = time.time()
    uncertainties = []
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        logits, sigma_a, (h, p_i) = model(data, adapt=True)
        W2 = model.fc2.weight
        var_terms = (W2.unsqueeze(1) * h.unsqueeze(0))**2
        var_epistemic = p_i * (1 - p_i) * var_terms.sum(-1)
        mean_pred = F.softmax(logits, dim=1)
        entropy = -torch.sum(mean_pred * torch.log(mean_pred + 1e-6), dim=1)
        norm_entropy = entropy / np.log(num_classes)
        norm_var = (var_epistemic.mean(0) + sigma_a.squeeze()) / (num_classes * 10)  # Rough normalization to ~0-1 scale
        uncertainty = norm_entropy + norm_var  # Combined, starting near 1 for random
        uncertainties.append(uncertainty.mean().item())
        pred = mean_pred.argmax(1)
        correct += pred.eq(target).sum().item()
    acc = correct / len(loader.dataset)
    time_taken = time.time() - start_time
    avg_unc = np.mean(uncertainties)
    print(f"ARB-Dropout on {name}: Acc {acc:.4f}, Norm Avg Unc {avg_unc:.4f}, Time {time_taken:.2f}s")

# Function to test initial (random) model
def initial_test(model, test_loader, name, num_classes):
    print(f"Initial (random) for {name}:")
    mc_dropout_test(model, test_loader, name, num_classes)
    arb_dropout_test(model, test_loader, name, num_classes)

# Main
if __name__ == "__main__":
    # CIFAR-10
    cifar10_model = Net(num_classes=10).to(device)
    initial_test(cifar10_model, cifar10_test_loader, 'CIFAR10', 10)
    train(cifar10_model, cifar10_train_loader, cifar10_val_loader, 'CIFAR10', 10)
    mc_dropout_test(cifar10_model, cifar10_test_loader, 'CIFAR10', 10)
    arb_dropout_test(cifar10_model, cifar10_test_loader, 'CIFAR10', 10)

    # CIFAR-100
    cifar100_model = Net(num_classes=100).to(device)
    initial_test(cifar100_model, cifar100_test_loader, 'CIFAR100', 100)
    train(cifar100_model, cifar100_train_loader, cifar100_val_loader, 'CIFAR100', 100)
    mc_dropout_test(cifar100_model, cifar100_test_loader, 'CIFAR100', 100)
    arb_dropout_test(cifar100_model, cifar100_test_loader, 'CIFAR100', 100)

    # SVHN
    svhn_model = Net(num_classes=10).to(device)
    initial_test(svhn_model, svhn_test_loader, 'SVHN', 10)
    train(svhn_model, svhn_train_loader, svhn_val_loader, 'SVHN', 10)
    mc_dropout_test(svhn_model, svhn_test_loader, 'SVHN', 10)
    arb_dropout_test(svhn_model, svhn_test_loader, 'SVHN', 10)

    # STL-10
    stl10_model = Net(num_classes=10).to(device)
    initial_test(stl10_model, stl10_test_loader, 'STL10', 10)
    train(stl10_model, stl10_train_loader, stl10_val_loader, 'STL10', 10)
    mc_dropout_test(stl10_model, stl10_test_loader, 'STL10', 10)
    arb_dropout_test(stl10_model, stl10_test_loader, 'STL10', 10)

