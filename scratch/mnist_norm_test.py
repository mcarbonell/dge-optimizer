import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import time

# Configuración
BATCH_SIZE = 128
EPOCHS = 5
LR = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Definir el Modelo
class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )
    def forward(self, x):
        return self.net(x)

def train_and_eval(name, train_loader, test_loader, seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    print(f"  Semilla {seed}...")
    model = SimpleMLP().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    
    t0 = time.time()
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
        
        # Evaluar
        model.eval()
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(DEVICE), target.to(DEVICE)
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        
        acc = 100. * correct / len(test_loader.dataset)
        print(f"Epoch {epoch}: Loss: {total_loss/len(train_loader):.4f}, Test Acc: {acc:.2f}%")
    
    total_time = time.time() - t0
    print(f"Tiempo total: {total_time:.2f}s")
    return acc

if __name__ == "__main__":
    # Experimento 1: Normalización Estándar (Media=0, Var=1)
    transform_std = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Experimento 2: Solo [0, 1] (Min-Max)
    transform_01 = transforms.Compose([
        transforms.ToTensor()
    ])
    
    train_ds_std = datasets.MNIST('./data', train=True, download=True, transform=transform_std)
    test_ds_std = datasets.MNIST('./data', train=False, transform=transform_std)
    
    train_ds_01 = datasets.MNIST('./data', train=True, download=True, transform=transform_01)
    test_ds_01 = datasets.MNIST('./data', train=False, transform=transform_01)
    
    loader_tr_std = torch.utils.data.DataLoader(train_ds_std, batch_size=BATCH_SIZE, shuffle=True)
    loader_te_std = torch.utils.data.DataLoader(test_ds_std, batch_size=BATCH_SIZE)
    
    loader_tr_01 = torch.utils.data.DataLoader(train_ds_01, batch_size=BATCH_SIZE, shuffle=True)
    loader_te_01 = torch.utils.data.DataLoader(test_ds_01, batch_size=BATCH_SIZE)
    
    seeds = [42, 123, 7, 99, 1000]
    results = {"std": [], "01": []}
    times = {"std": [], "01": []}

    print("\n" + "="*40)
    print("PROBANDO NORMALIZACIÓN ESTÁNDAR")
    print("="*40)
    for s in seeds:
        t0 = time.time()
        acc = train_and_eval("Std", loader_tr_std, loader_te_std, s)
        results["std"].append(acc)
        times["std"].append(time.time() - t0)

    print("\n" + "="*40)
    print("PROBANDO NORMALIZACIÓN [0, 1]")
    print("="*40)
    for s in seeds:
        t0 = time.time()
        acc = train_and_eval("01", loader_tr_01, loader_te_01, s)
        results["01"].append(acc)
        times["01"].append(time.time() - t0)

    import numpy as np
    
    print("\n" + "="*40)
    print(f"RESULTADO FINAL (5 Semillas)")
    print(f"Media 0: {np.mean(results['std']):.2f}% ± {np.std(results['std']):.2f} (Tiempo medio: {np.mean(times['std']):.2f}s)")
    print(f"Solo [0,1]: {np.mean(results['01']):.2f}% ± {np.std(results['01']):.2f} (Tiempo medio: {np.mean(times['01']):.2f}s)")
    print("="*40)
