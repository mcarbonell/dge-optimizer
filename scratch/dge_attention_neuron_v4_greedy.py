import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import math
import time

class GreedyNeuronLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        # Aquí los pesos SÍ son permanentes y evolucionan (sin autograd)
        std = math.sqrt(2.0 / in_features)
        self.weights = nn.Parameter(torch.randn(out_features, in_features) * std, requires_grad=False)
        self.bias = nn.Parameter(torch.zeros(out_features), requires_grad=False)
        
        # Estado del optimizador local (Deltas adaptativos)
        self.delta_m = torch.ones(out_features, 1) * 0.01  # Delta multiplicativo inicial
        self.delta_a = torch.ones(out_features, 1) * 0.01  # Delta aditivo inicial

    def forward(self, x):
        return torch.matmul(x, self.weights.t()) + self.bias

class GreedyNeuronMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = GreedyNeuronLayer(784, 128)
        self.layer2 = GreedyNeuronLayer(128, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.layer1(x)
        x = torch.relu(x)
        x = self.layer2(x)
        return x

def evaluate_loss(model, x, y):
    logits = model(x)
    loss = nn.functional.cross_entropy(logits, y)
    return loss.item(), logits

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Entrenando Greedy en: {device}")

    # BATCH SIZE GIGANTE (como indicaste en el chat para estabilizar el greedy)
    BATCH_SIZE = 8192
    STEPS = 1000

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)

    # Cargar todo en memoria para velocidad extrema
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=10000, shuffle=False)
    
    X_test, Y_test = next(iter(test_loader))
    X_test, Y_test = X_test.to(device), Y_test.to(device)

    model = GreedyNeuronMLP().to(device)
    model.eval() # Siempre en eval (sin dropout tradicional)

    layers = [model.layer1, model.layer2]

    print("Iniciando Búsqueda Greedy por Neurona...")
    
    # Iterador infinito para lotes
    def get_batch():
        while True:
            for data, target in train_loader:
                yield data.to(device), target.to(device)
                
    batch_iter = get_batch()
    
    best_acc = 0.0
    
    # Usaremos el mismo lote para la evaluación base y la perturbada para tener una señal justa
    t0 = time.time()
    
    for step in range(1, STEPS + 1):
        X, Y = next(batch_iter)
        current_loss, _ = evaluate_loss(model, X, Y)
        
        # Elegir capa y neurona al azar
        l_idx = torch.randint(0, len(layers), (1,)).item()
        layer = layers[l_idx]
        n_idx = torch.randint(0, layer.weights.shape[0], (1,)).item()
        
        # === MUTACIÓN ESTRUCTURAL ===
        # Guardar estado original
        orig_weights = layer.weights[n_idx].clone()
        orig_bias = layer.bias[n_idx].clone()
        
        # Elegir tipo de mutación: 50% aditiva, 50% multiplicativa
        mut_type = torch.rand(1).item()
        
        # Crear máscara aleatoria (solo alterar 50% de los cables de esta neurona)
        mask = torch.bernoulli(torch.full_like(orig_weights, 0.5))
        
        # Aplicar perturbación
        signo = 1.0 if torch.rand(1).item() > 0.5 else -1.0
        
        if mut_type > 0.5:
            # Multiplicativa
            step_size = layer.delta_m[n_idx].item()
            factor = 1.0 + (signo * step_size)
            layer.weights[n_idx] = torch.where(mask > 0, orig_weights * factor, orig_weights)
        else:
            # Aditiva
            step_size = layer.delta_a[n_idx].item()
            adicion = signo * step_size
            layer.weights[n_idx] = torch.where(mask > 0, orig_weights + adicion, orig_weights)
            # También perturbamos el bias si es aditivo
            layer.bias[n_idx] += (signo * step_size * 0.1)

        # === EVALUACIÓN DEL LOT ===
        new_loss, _ = evaluate_loss(model, X, Y)
        
        # === REGLA DE ADAPTACIÓN (Rechenberg / Evolutiva) ===
        if new_loss < current_loss:
            # Éxito: Mantener cambios y aumentar confianza (acelerar delta)
            if mut_type > 0.5:
                layer.delta_m[n_idx] = min(layer.delta_m[n_idx].item() * 1.2, 0.5)
            else:
                layer.delta_a[n_idx] = min(layer.delta_a[n_idx].item() * 1.2, 0.5)
        else:
            # Fracaso: Revertir cambios y reducir confianza (afinar delta)
            layer.weights[n_idx] = orig_weights
            layer.bias[n_idx] = orig_bias
            if mut_type > 0.5:
                layer.delta_m[n_idx] = max(layer.delta_m[n_idx].item() * 0.95, 1e-5)
            else:
                layer.delta_a[n_idx] = max(layer.delta_a[n_idx].item() * 0.95, 1e-5)
                
        # Logs
        if step % 100 == 0:
            with torch.no_grad():
                logits = model(X_test)
                pred = logits.argmax(dim=1, keepdim=True)
                acc = pred.eq(Y_test.view_as(pred)).float().mean().item()
                best_acc = max(best_acc, acc)
                t_elapsed = time.time() - t0
                print(f"Step {step:4d} | Batch Loss: {current_loss:.4f} | Test Acc: {acc:.4f} | Time: {t_elapsed:.1f}s")

    print(f"Mejor Test Acc final: {best_acc:.4f}")

if __name__ == "__main__":
    main()