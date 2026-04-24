import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import math
import time

class StochasticNeuronLayer(nn.Module):
    def __init__(self, in_features, out_features, mask_prob=0.5):
        super().__init__()
        # Pesos aleatorios fijos (cableado base)
        std = math.sqrt(2.0 / in_features)
        w_init = torch.randn(out_features, in_features) * std
        self.register_buffer('w_init', w_init)
        
        # O(neuronas) variables
        self.delta_mult = nn.Parameter(torch.ones(out_features, 1))
        self.delta_add = nn.Parameter(torch.zeros(out_features, 1))
        
        # Bias angular: theta evoluciona libremente, pero el sesgo real es sin(theta)
        self.theta_bias = nn.Parameter(torch.zeros(out_features))
        
        self.mask_prob = mask_prob

    def forward(self, x):
        if self.training:
            # Máscara aleatoria. 1 = el cable escucha a la neurona. 0 = el cable ignora a la neurona.
            mask = torch.bernoulli(torch.full(self.w_init.shape, self.mask_prob, device=self.w_init.device))
            
            # Evolución parcial
            w_evolved = torch.where(mask > 0, 
                                    self.w_init * self.delta_mult + self.delta_add, 
                                    self.w_init)
        else:
            # En inferencia, aplicamos el valor esperado matemático para mantener la escala.
            # E[W] = P(mask) * (w_init * delta_mult + delta_add) + (1 - P(mask)) * w_init
            expected_mult = 1.0 + self.mask_prob * (self.delta_mult - 1.0)
            expected_add = self.mask_prob * self.delta_add
            w_evolved = self.w_init * expected_mult + expected_add
            
        # Bias acotado [-1, 1] usando la función seno
        bias_actual = torch.sin(self.theta_bias)
        
        return torch.matmul(x, w_evolved.t()) + bias_actual

class StochasticNeuronMLP(nn.Module):
    def __init__(self):
        super().__init__()
        # Misma capacidad base que el v2, pero estocástico
        self.layer1 = StochasticNeuronLayer(784, 512, mask_prob=0.5)
        self.layer2 = StochasticNeuronLayer(512, 10, mask_prob=0.5)

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.layer1(x)
        x = torch.relu(x)
        x = self.layer2(x)
        return x

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Entrenando en: {device}")

    BATCH_SIZE = 256
    EPOCHS = 15 # Damos un poco más de tiempo porque la estocasticidad frena el aprendizaje inicial
    LR = 0.01

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = StochasticNeuronMLP().to(device)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parámetros entrenables: {total_params}")

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