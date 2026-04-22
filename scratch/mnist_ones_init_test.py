import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import time
import numpy as np

# Configuración
BATCH_SIZE = 128
EPOCHS = 5
LR = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Definir el Modelo con Inicialización Especial
class SpecialInitMLP(nn.Module):
    def __init__(self, init_ones=True):
        super(SpecialInitMLP, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        self.relu = nn.ReLU()
        
        if init_ones:
            # Forzar todos los pesos de la primera capa a 1.0
            nn.init.constant_(self.fc1.weight, 1.0)
            # El bias lo dejamos en 0 o default para que no sea todo idéntico?
            # El usuario dijo "los cables", así que solo pesos.
            nn.init.constant_(self.fc1.bias, 0.0)

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def train_and_eval(name, train_loader, test_loader, seed, init_ones):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    print(f"  Semilla {seed}...")
    model = SpecialInitMLP(init_ones=init_ones).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0
        for data, target in train_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        # Evaluar por época
        model.eval()
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(DEVICE), target.to(DEVICE)
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        
        acc = 100. * correct / len(test_loader.dataset)
        print(f"    Epoch {epoch}: Loss: {total_loss/len(train_loader):.4f}, Test Acc: {acc:.2f}%")
    
    return acc

if __name__ == "__main__":
    # Normalización [0, 1] (Solo positivos)
    transform_01 = transforms.Compose([transforms.ToTensor()])
    
    train_ds = datasets.MNIST('./data', train=True, download=True, transform=transform_01)
    test_ds = datasets.MNIST('./data', train=False, transform=transform_01)
    
    loader_tr = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    loader_te = torch.utils.data.DataLoader(test_ds, batch_size=BATCH_SIZE)
    
    seeds = [42, 123, 7, 99, 1000]
    results_ones = []
    
    print("\n" + "="*40)
    print("PROBANDO INICIALIZACIÓN A 1.0 (Capa 1) + Norm [0, 1]")
    print("="*40)
    
    t0_total = time.time()
    for s in seeds:
        acc = train_and_eval("Ones", loader_tr, loader_te, s, init_ones=True)
        results_ones.append(acc)
        print(f"    Acc: {acc:.2f}%")

    total_time = time.time() - t0_total
    
    print("\n" + "="*40)
    print(f"RESULTADO FINAL (5 Semillas)")
    print(f"Ones Init (Capa 1): {np.mean(results_ones):.2f}% ± {np.std(results_ones):.2f}")
    print(f"Tiempo total: {total_time:.2f}s")
    print("="*40)
    print("NOTA: Comparar con el 97.25% anterior de la inicialización aleatoria.")
