import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import math
import time

class DualNeuronLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        std = math.sqrt(2.0 / in_features)
        w_init = torch.randn(out_features, in_features) * std
        self.register_buffer('w_init', w_init)
        
        # Orden de entrada de la neurona destino (Fan-In)
        self.delta_in_m = nn.Parameter(torch.ones(out_features, 1))
        self.delta_in_a = nn.Parameter(torch.zeros(out_features, 1))
        
        # Orden de salida de la neurona origen (Fan-Out)
        self.delta_out_m = nn.Parameter(torch.ones(1, in_features))
        self.delta_out_a = nn.Parameter(torch.zeros(1, in_features))
        
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        # Cruzamos las órdenes (Outer product / Broadcasting)
        # Cada cable recibe una combinación ÚNICA de sus dos neuronas
        # w_m shape: (out, in). Es una matriz de Rango 1!
        w_m = self.delta_in_m * self.delta_out_m
        w_a = self.delta_in_a + self.delta_out_a
        
        # El peso evoluciona de forma completamente independiente para cada cable
        w_evolved = self.w_init * w_m + w_a
        
        return torch.matmul(x, w_evolved.t()) + self.bias

class DualNeuronMLP(nn.Module):
    def __init__(self):
        super().__init__()
        # Misma arquitectura que V2 (784 -> 512 -> 10) para comparar justamente
        self.layer1 = DualNeuronLayer(784, 512)
        self.layer2 = DualNeuronLayer(512, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.layer1(x)
        x = torch.relu(x)
        x = self.layer2(x)
        return x

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Entrenando V6 (Cables Independientes / Dual Neuron) en: {device}")

    BATCH_SIZE = 256
    EPOCHS = 10
    LR = 0.01

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = DualNeuronMLP().to(device)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parámetros entrenables: {total_params} (Cables independientes mediante factorización de Rango 1)")

    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        t0 = time.time()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += criterion(output, target).item() * data.size(0)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        test_acc = correct / len(test_loader.dataset)
        t_epoch = time.time() - t0
        
        print(f"Epoch {epoch:2d} | Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f} | Time: {t_epoch:.1f}s")

if __name__ == "__main__":
    main()