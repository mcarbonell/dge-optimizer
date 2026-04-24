import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import math
import time

# --- Capa Lineal Rank-2 de la V7 ---
class DualNeuronPhaseLayer(nn.Module):
    def __init__(self, in_features, out_features, rank=2):
        super().__init__()
        std = math.sqrt(2.0 / in_features)
        self.register_buffer('w_init', torch.randn(out_features, in_features) * std)
        
        self.delta_in_m = nn.Parameter(torch.randn(out_features, rank) * 0.1 + 1.0)
        self.delta_out_m = nn.Parameter(torch.randn(rank, in_features) * 0.1 + 1.0)
        self.delta_in_a = nn.Parameter(torch.randn(out_features, rank) * 0.1)
        self.delta_out_a = nn.Parameter(torch.randn(rank, in_features) * 0.1)
        self.theta_bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        w_m = torch.matmul(self.delta_in_m, self.delta_out_m)
        w_a = torch.matmul(self.delta_in_a, self.delta_out_a)
        w_evolved = self.w_init * w_m + w_a
        return torch.matmul(x, w_evolved.t()) + torch.sin(self.theta_bias)


# --- NUEVA: Capa Convolucional Rank-2 ---
class DualConvPhaseLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, rank=2):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        
        # Filtros espaciales aleatorios CONGELADOS
        # Shape: (out_channels, in_channels, kH, kW)
        fan_in = in_channels * kernel_size * kernel_size
        std = math.sqrt(2.0 / fan_in)
        self.register_buffer('w_init', torch.randn(out_channels, in_channels, kernel_size, kernel_size) * std)
        
        # Modulación Rank-2 entre canales de entrada y salida
        self.delta_in_m = nn.Parameter(torch.randn(out_channels, rank) * 0.1 + 1.0)
        self.delta_out_m = nn.Parameter(torch.randn(rank, in_channels) * 0.1 + 1.0)
        
        self.delta_in_a = nn.Parameter(torch.randn(out_channels, rank) * 0.1)
        self.delta_out_a = nn.Parameter(torch.randn(rank, in_channels) * 0.1)
        
        # Bias Angular por canal de salida
        self.theta_bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x):
        # w_m shape: (out_channels, in_channels)
        w_m = torch.matmul(self.delta_in_m, self.delta_out_m)
        w_a = torch.matmul(self.delta_in_a, self.delta_out_a)
        
        # Expandimos para broadcast sobre las dimensiones espaciales del kernel
        # (out_channels, in_channels, 1, 1)
        w_m_expanded = w_m.unsqueeze(-1).unsqueeze(-1)
        w_a_expanded = w_a.unsqueeze(-1).unsqueeze(-1)
        
        # Evolucionamos los filtros congelados
        w_evolved = self.w_init * w_m_expanded + w_a_expanded
        
        # Sesgo angular acotado [-1, 1]
        phase_bias = torch.sin(self.theta_bias)
        
        # Aplicamos convolución estándar
        out = F.conv2d(x, w_evolved, padding=self.padding)
        
        # Sumamos el bias (haciendo broadcast sobre batch, H y W)
        return out + phase_bias.view(1, -1, 1, 1)


class Rank2CNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Arquitectura CNN sencilla
        # Input CIFAR: 3 x 32 x 32
        self.conv1 = DualConvPhaseLayer(3, 32, kernel_size=3, padding=1, rank=2)
        # Después de pool: 32 x 16 x 16
        self.conv2 = DualConvPhaseLayer(32, 64, kernel_size=3, padding=1, rank=2)
        # Después de pool: 64 x 8 x 8
        
        # Clasificador Lineal Rank-2
        self.fc = DualNeuronPhaseLayer(64 * 8 * 8, 10, rank=2)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = F.max_pool2d(x, 2)
        
        x = self.conv2(x)
        x = torch.relu(x)
        x = F.max_pool2d(x, 2)
        
        x = x.view(x.size(0), -1) # Flatten (Batch, 64*8*8)
        x = self.fc(x)
        return x

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Entrenando V9 (Rank-2 CNN) en: {device}")

    BATCH_SIZE = 256
    EPOCHS = 15
    LR = 0.005

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    train_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10('./data', train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = Rank2CNN().to(device)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parámetros entrenables: {total_params:,} (CNN estándar: ~60,000 parámetros)")

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