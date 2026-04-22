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

# 1. Definir el Modelo con Inicialización por Neurona
class NeuronInitMLP(nn.Module):
    def __init__(self, mode="pos"):
        super(NeuronInitMLP, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        self.relu = nn.ReLU()
        
        with torch.no_grad():
            # Para cada neurona j de la capa 1 (128 neuronas)
            for j in range(128):
                if mode == "pos":
                    # Random [0, 1]
                    val = torch.rand(1).item()
                else:
                    # Random [-1, 1]
                    val = torch.rand(1).item() * 2.0 - 1.0
                
                # Asignar el mismo valor a todos los cables que entran a esa neurona
                self.fc1.weight[j, :] = val
            
            # Biases a cero
            self.fc1.bias.fill_(0.0)

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def train_and_eval(name, train_loader, test_loader, seed, mode):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    print(f"  [{name}] Semilla {seed}...")
    model = NeuronInitMLP(mode=mode).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    
    final_acc = 0
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
        
        final_acc = 100. * correct / len(test_loader.dataset)
        print(f"    Epoch {epoch}: Loss: {total_loss/len(train_loader):.4f}, Test Acc: {final_acc:.2f}%")
    
    return final_acc

if __name__ == "__main__":
    # Normalización [0, 1]
    transform_01 = transforms.Compose([transforms.ToTensor()])
    train_ds = datasets.MNIST('./data', train=True, download=True, transform=transform_01)
    test_ds = datasets.MNIST('./data', train=False, transform=transform_01)
    loader_tr = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    loader_te = torch.utils.data.DataLoader(test_ds, batch_size=BATCH_SIZE)
    
    seeds = [42, 123, 7, 99, 1000]
    
    for mode_name, mode in [("POSITIVOS [0, 1]", "pos"), ("MIXTO [-1, 1]", "mix")]:
        print("\n" + "="*50)
        print(f"MODO: {mode_name}")
        print("="*50)
        
        results = []
        t0 = time.time()
        for s in seeds:
            acc = train_and_eval(mode_name, loader_tr, loader_te, s, mode)
            results.append(acc)
        
        print(f"\nRESULTADO {mode_name}: {np.mean(results):.2f}% ± {np.std(results):.2f}")
        print(f"Tiempo total: {time.time() - t0:.2f}s")
