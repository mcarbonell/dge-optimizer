import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import math
import time

class DualNeuronLayerRank2(nn.Module):
    def __init__(self, in_features, out_features, rank=2):
        super().__init__()
        std = math.sqrt(2.0 / in_features)
        w_init = torch.randn(out_features, in_features) * std
        self.register_buffer('w_init', w_init)
        
        self.rank = rank
        
        # Vectores de rango para multiplicador
        self.delta_in_m = nn.Parameter(torch.randn(out_features, rank) * 0.1 + 1.0)
        self.delta_out_m = nn.Parameter(torch.randn(rank, in_features) * 0.1 + 1.0)
        
        # Vectores de rango para aditivo
        self.delta_in_a = nn.Parameter(torch.randn(out_features, rank) * 0.1)
        self.delta_out_a = nn.Parameter(torch.randn(rank, in_features) * 0.1)
        
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        # Reconstrucción de la matriz de variaciones mediante multiplicación de matrices de bajo rango (Rank-2)
        # delta_in_m (out, rank) @ delta_out_m (rank, in) -> w_m (out, in)
        w_m = torch.matmul(self.delta_in_m, self.delta_out_m)
        w_a = torch.matmul(self.delta_in_a, self.delta_out_a)
        
        # Aplicamos la variación al peso congelado base
        w_evolved = self.w_init * w_m + w_a
        
        return torch.matmul(x, w_evolved.t()) + self.bias

class DualNeuronRank2MLP(nn.Module):
    def __init__(self):
        super().__init__()
        # Misma arquitectura que V2 y V6 (784 -> 512 -> 10) para comparar justamente
        self.layer1 = DualNeuronLayerRank2(784, 512, rank=2)
        self.layer2 = DualNeuronLayerRank2(512, 10, rank=2)

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.layer1(x)
        x = torch.relu(x)
        x = self.layer2(x)
        return x

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Entrenando V6b (Dual Neuron Rank-2) en: {device}")

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

    model = DualNeuronRank2MLP().to(device)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parámetros entrenables: {total_params} (Factorización de Rango 2)")

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