import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import math
import time

class AttentionNeuronLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        # Pesos aleatorios fijos (el "cableado" inicial)
        # Usamos kaiming/he init para la varianza inicial
        std = math.sqrt(2.0 / in_features)
        w_init = torch.randn(out_features, in_features) * std
        self.register_buffer('w_init', w_init)
        
        # Variables entrenables por neurona (O(neuronas) en vez de O(pesos))
        # delta_mult escala la importancia de las conexiones existentes
        self.delta_mult = nn.Parameter(torch.ones(out_features, 1))
        # theta_bias actúa como el umbral de activación (fase)
        self.theta_bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        # 1. El peso evoluciona multiplicativamente
        w_evolved = self.w_init * self.delta_mult
        
        # 2. Normalización de Fan-In: L1 norm sobre la dimensión de entrada
        # Mantenemos los signos pero aseguramos que la suma de magnitudes sea 1.
        w_abs_sum = torch.sum(torch.abs(w_evolved), dim=1, keepdim=True) + 1e-8
        w_norm = w_evolved / w_abs_sum
        
        # 3. Bias Angular
        bias_threshold = torch.sin(self.theta_bias)
        
        # 4. Combinación lineal + bias (Gating)
        # x: (Batch, In), w_norm.t(): (In, Out)
        out = torch.matmul(x, w_norm.t()) - bias_threshold
        return out

class AttentionNeuronMLP(nn.Module):
    def __init__(self):
        super().__init__()
        # Arquitectura simple: 784 -> 128 -> 10
        self.layer1 = AttentionNeuronLayer(784, 128)
        self.layer2 = AttentionNeuronLayer(128, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.layer1(x)
        x = torch.relu(x)
        x = self.layer2(x)
        return x

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Entrenando en: {device}")

    # Hyperparámetros
    BATCH_SIZE = 256
    EPOCHS = 10
    LR = 0.01

    # Datos
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Modelo y Optimizador
    model = AttentionNeuronMLP().to(device)
    
    # Contar parámetros
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parámetros entrenables: {total_params} (vs {784*128 + 128 + 128*10 + 10} de un MLP estándar)")

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

        # Evaluación
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