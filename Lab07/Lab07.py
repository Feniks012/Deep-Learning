import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader as TorchDataLoader
from torchvision import datasets, transforms

from torch_geometric.datasets import TUDataset, Actor
from torch_geometric.loader import DataLoader as GeoDataLoader
from torch_geometric.nn import GCNConv, global_mean_pool

import snntorch as snn
from snntorch import surrogate


# CONFIG
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

GNN_EPOCHS = 20
SNN_EPOCHS = 3
BATCH_SIZE = 64
LR = 0.001

print("DEVICE:", DEVICE)


# 1. GNN - GRAPH CLASSIFICATION - MUTAG


class GCNGraphClassifier(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes, depth):
        super().__init__()

        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))

        for _ in range(depth - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))

        self.classifier = nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)

        x = global_mean_pool(x, batch)
        return self.classifier(x)


def train_graph_model(model, train_loader, test_loader, epochs=20):
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for data in train_loader:
            data = data.to(DEVICE)

            optimizer.zero_grad()
            out = model(data.x, data.edge_index, data.batch)
            loss = criterion(out, data.y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data in test_loader:
            data = data.to(DEVICE)
            out = model(data.x, data.edge_index, data.batch)
            pred = out.argmax(dim=1)

            correct += (pred == data.y).sum().item()
            total += data.y.size(0)

    return correct / total


def run_mutag_experiment():

    print("GNN GRAPH CLASSIFICATION - MUTAG")


    dataset = TUDataset(root="./data/TUDataset", name="MUTAG")
    dataset = dataset.shuffle()

    train_size = int(0.8 * len(dataset))
    train_dataset = dataset[:train_size]
    test_dataset = dataset[train_size:]

    train_loader = GeoDataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = GeoDataLoader(test_dataset, batch_size=32, shuffle=False)

    results = {}

    for depth in range(1, 9):
        print(f"\nTraining GCN depth={depth}")

        model = GCNGraphClassifier(
            in_channels=dataset.num_node_features,
            hidden_channels=64,
            num_classes=dataset.num_classes,
            depth=depth
        )

        acc = train_graph_model(model, train_loader, test_loader, epochs=GNN_EPOCHS)
        results[depth] = acc

        print(f"MUTAG | GCN {depth} layers | Test Accuracy: {acc:.4f}")

    return results


# 2. GNN - NODE CLASSIFICATION - ACTOR


class GCNNodeClassifier(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes, depth):
        super().__init__()

        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))

        for _ in range(depth - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))

        self.classifier = nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)

        return self.classifier(x)


def train_actor_model(model, data, epochs=20):
    model.to(DEVICE)
    data = data.to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    train_mask = data.train_mask[:, 0] if data.train_mask.dim() == 2 else data.train_mask
    test_mask = data.test_mask[:, 0] if data.test_mask.dim() == 2 else data.test_mask

    for epoch in range(epochs):
        model.train()

        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out[train_mask], data.y[train_mask])
        loss.backward()
        optimizer.step()

    model.eval()

    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)

        correct = (pred[test_mask] == data.y[test_mask]).sum().item()
        total = int(test_mask.sum())

    return correct / total


def run_actor_experiment():

    print("GNN NODE CLASSIFICATION - ACTOR")


    dataset = Actor(root="./data/Actor")
    data = dataset[0]

    results = {}

    for depth in range(1, 9):
        print(f"\nTraining Actor GCN depth={depth}")

        model = GCNNodeClassifier(
            in_channels=dataset.num_node_features,
            hidden_channels=64,
            num_classes=dataset.num_classes,
            depth=depth
        )

        acc = train_actor_model(model, data, epochs=GNN_EPOCHS)
        results[depth] = acc

        print(f"Actor | GCN {depth} layers | Test Accuracy: {acc:.4f}")

    return results



# 3. SNN MODEL

class SimpleSNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_steps=25):
        super().__init__()

        self.num_steps = num_steps

        beta = 0.95
        spike_grad = surrogate.fast_sigmoid()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)

        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)

    def forward(self, x):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        spk2_rec = []

        x = x.view(x.size(0), -1)

        for step in range(self.num_steps):
            cur1 = self.fc1(x)
            spk1, mem1 = self.lif1(cur1, mem1)

            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)

            spk2_rec.append(spk2)

        return torch.stack(spk2_rec, dim=0)


def train_snn(model, train_loader, test_loader, epochs=3, name="SNN"):
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        running_loss = 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()

            spk_rec = model(imgs)
            output = spk_rec.sum(dim=0)

            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"[{name}] Epoch {epoch+1}/{epochs} | Loss: {running_loss / len(train_loader):.4f}")

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

            spk_rec = model(imgs)
            output = spk_rec.sum(dim=0)

            pred = output.argmax(dim=1)

            correct += (pred == labels).sum().item()
            total += labels.size(0)

    acc = correct / total
    print(f"{name} Test Accuracy: {acc:.4f}")

    return acc



# 4. SNN - FASHIONMNIST


def run_fashionmnist_snn():

    print("SNN - FASHIONMNIST")


    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_dataset = datasets.FashionMNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.FashionMNIST(
        root="./data",
        train=False,
        download=True,
        transform=transform
    )

    train_subset = torch.utils.data.Subset(train_dataset, range(10000))
    test_subset = torch.utils.data.Subset(test_dataset, range(2000))

    train_loader = TorchDataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = TorchDataLoader(test_subset, batch_size=BATCH_SIZE, shuffle=False)

    model = SimpleSNN(
        input_size=28 * 28,
        hidden_size=256,
        num_classes=10,
        num_steps=25
    )

    acc = train_snn(model, train_loader, test_loader, epochs=SNN_EPOCHS, name="FashionMNIST SNN")

    return acc



# 5. SNN - CIFAR10
def run_cifar10_snn():

    print("SNN - CIFAR10")


    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ])

    train_dataset = datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=transform
    )

    train_subset = torch.utils.data.Subset(train_dataset, range(10000))
    test_subset = torch.utils.data.Subset(test_dataset, range(2000))

    train_loader = TorchDataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = TorchDataLoader(test_subset, batch_size=BATCH_SIZE, shuffle=False)

    model = SimpleSNN(
        input_size=32 * 32,
        hidden_size=512,
        num_classes=10,
        num_steps=25
    )

    acc = train_snn(model, train_loader, test_loader, epochs=SNN_EPOCHS, name="CIFAR10 SNN")

    return acc



# 6. PLOTS

def plot_gnn_results(mutag_results, actor_results):
    depths = list(range(1, 9))

    mutag_acc = [mutag_results[d] for d in depths]
    actor_acc = [actor_results[d] for d in depths]

    plt.figure(figsize=(10, 5))
    plt.plot(depths, mutag_acc, marker="o", label="MUTAG - Graph Classification")
    plt.plot(depths, actor_acc, marker="o", label="Actor - Node Classification")
    plt.xlabel("Number of GCN layers")
    plt.ylabel("Accuracy")
    plt.title("GCN Depth vs Accuracy")
    plt.legend()
    plt.grid(True)
    plt.show()



# MAIN
if __name__ == "__main__":

    mutag_results = run_mutag_experiment()
    actor_results = run_actor_experiment()

    fashion_acc = run_fashionmnist_snn()
    cifar_acc = run_cifar10_snn()

    plot_gnn_results(mutag_results, actor_results)


    print("FINAL SUMMARY")
  

    print("\nMUTAG - Graph Classification:")
    for depth, acc in mutag_results.items():
        print(f"GCN {depth} layers: {acc:.4f}")

    print("\nActor - Node Classification:")
    for depth, acc in actor_results.items():
        print(f"GCN {depth} layers: {acc:.4f}")

    print("\nSNN:")
    print(f"FashionMNIST Accuracy: {fashion_acc:.4f}")
    print(f"CIFAR10 Accuracy:      {cifar_acc:.4f}")