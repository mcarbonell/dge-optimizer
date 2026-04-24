import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import math
import time

class DualNeuronPhaseLayer(nn.Module):
    def __init__(self, in_features, out_features, rank=2):
        super().__init__()
        std = math.sqrt(2.0 / in_features)
        w_init = torch.randn(out_features, in_features) * std
        self.register_buffer('w_init', w_init)
        
        self.rank = rank
        
        self.delta_in_m = nn.Parameter(torch.randn(out_features, rank) * 0.1 + 1.0)
        self.delta_out_m = nn.Parameter(torch.randn(rank, in_features) * 0.1 + 1.0)
        
        self.delta_in_a = nn.Parameter(torch.randn(out_features, rank) * 0.1)
        self.delta_out_a = nn.Parameter(torch.randn(rank, in_features) * 0.1)
        
        self.theta_bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        w_m = torch.matmul(self.delta_in_m, self.delta_out_m)
        w_a = torch.matmul(self.delta_in_a, self.delta_out_a)
        
        w_evolved = self.w_init * w_m + w_a
        phase_bias = torch.sin(self.theta_bias)
        
        return torch.matmul(x, w_evolved.t()) + phase_bias

class CifarPhaseMLP(nn.Module):
    def __init__(self):
        super().__init__()
        # CIFAR-10: Imágenes de 3 canales de 32x32 = 3072 inputs
        # Usamos 1024 neuronas ocultas. 
        # Parámetros MLP estándar: 3072*1024 + 1024*10 = ~3.1 Millones
        self.layer1 = DualNeuronPhaseLayer(3072, 1024, rank=2)
        self.layer2 = DualNeuronPhaseLayer(1024, 10, rank=2)

    def forward(self, x):
        x = x.view(-1, 3072)
        x = self.layer1(x)
        x = torch.relu(x)
        x = self.layer2(x)
        return x

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Entrenando V8 (CIFAR-10: Rank-2 + Phase Bias) en: {device}")

    BATCH_SIZE = 256
    EPOCHS = 15 # Damos un poco más de tiempo porque CIFAR-10 es más duro
    LR = 0.005 # Reducimos un poco el LR para mayor estabilidad en imágenes complejas

    # Normalización estándar para CIFAR-10
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    train_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10('./data', train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = CifarPhaseMLP().to(device)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parámetros entrenables: {total_params:,} (vs ~3,100,000 de un MLP estándar)")

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