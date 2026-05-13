import json
import random
import re
import requests
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import xgboost as xgb
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import spearmanr



# CONFIG

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OLLAMA_MODEL = "qwen2.5:7b"   
OLLAMA_URL = "http://localhost:11434/api/generate"

NUM_INITIAL_ARCHS = 70      
NUM_ITERATIONS = 5
EPOCHS_CNN = 2
BATCH_SIZE = 128
MAX_LAYERS_ENCODED = 12
NUM_CLASSES = 10
PARAM_LIMIT = 2_000_000

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)



# CIFAR-10

transform_train = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

transform_test = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])

train_full = datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=transform_train
)

test_full = datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=transform_test
)

# mniejsze podzbiory, żeby lab działał szybciej
train_indices = np.random.permutation(len(train_full))[:8000]
test_indices = np.random.permutation(len(test_full))[:2000]

train_set = Subset(train_full, train_indices)
test_set = Subset(test_full, test_indices)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

print("DEVICE:", DEVICE)
print("Train samples:", len(train_set))
print("Test samples:", len(test_set))



# OLLAMA — GENERATOR ARCHITEKTUR

def ask_ollama(prompt, model=OLLAMA_MODEL):
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": model,
            "prompt": prompt,
            "stream": False
        },
        timeout=120
    )
    response.raise_for_status()
    return response.json()["response"]


def extract_json(text):
    text = text.strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return json.loads(match.group(0))

    raise ValueError("Nie znaleziono poprawnego JSON-a w odpowiedzi LLM.")


def generate_architecture_with_ollama():
    prompt = """
Generate one small CNN architecture for CIFAR-10 in valid JSON only.

Allowed layer types:
- conv
- batchnorm
- relu
- maxpool
- dropout
- globalavgpool
- linear

Rules:
- output JSON only
- maximum 6 conv layers
- use small number of filters: 16, 32, 64, 128
- kernel can be 3 or 5
- dropout p can be 0.1, 0.2, 0.3, 0.4
- include globalavgpool before linear
- include one final linear hidden layer
- do not include softmax

Required format:
{
  "layers": [
    {"type":"conv","filters":32,"kernel":3},
    {"type":"batchnorm"},
    {"type":"relu"},
    {"type":"maxpool"},
    {"type":"conv","filters":64,"kernel":3},
    {"type":"batchnorm"},
    {"type":"relu"},
    {"type":"globalavgpool"},
    {"type":"linear","units":128}
  ]
}
"""
    text = ask_ollama(prompt)
    arch = extract_json(text)
    return sanitize_architecture(arch)


# fallback, gdy Ollama zwróci zły JSON
def generate_random_architecture():
    layers = []
    conv_count = random.randint(2, 5)
    filters_choices = [16, 32, 64, 128]

    for i in range(conv_count):
        filters = random.choice(filters_choices)
        kernel = random.choice([3, 5])
        layers.append({"type": "conv", "filters": filters, "kernel": kernel})
        layers.append({"type": "batchnorm"})
        layers.append({"type": "relu"})

        if i < 2 and random.random() < 0.7:
            layers.append({"type": "maxpool"})

        if random.random() < 0.4:
            layers.append({"type": "dropout", "p": random.choice([0.1, 0.2, 0.3])})

    layers.append({"type": "globalavgpool"})
    layers.append({"type": "linear", "units": random.choice([64, 128, 256])})

    return {"layers": layers}


# WALIDACJA / SANITYZACJA ARCHITEKTURY

def sanitize_architecture(arch):
    allowed = {
        "conv",
        "batchnorm",
        "relu",
        "maxpool",
        "dropout",
        "linear",
        "globalavgpool"
    }

    if not isinstance(arch, dict) or "layers" not in arch:
        return generate_random_architecture()

    clean_layers = []
    conv_count = 0
    has_gap = False
    has_linear = False

    for layer in arch["layers"]:
        if not isinstance(layer, dict):
            continue

        layer_type = str(layer.get("type", "")).lower()

        if layer_type not in allowed:
            continue

        if layer_type == "conv":
            if conv_count >= 6:
                continue

            filters = int(layer.get("filters", 32))
            kernel = int(layer.get("kernel", 3))

            filters = min(max(filters, 8), 128)
            if kernel not in [3, 5]:
                kernel = 3

            clean_layers.append({
                "type": "conv",
                "filters": filters,
                "kernel": kernel
            })
            conv_count += 1

        elif layer_type == "dropout":
            p = float(layer.get("p", 0.2))
            p = min(max(p, 0.0), 0.5)
            clean_layers.append({"type": "dropout", "p": p})

        elif layer_type == "linear":
            units = int(layer.get("units", 128))
            units = min(max(units, 16), 512)
            clean_layers.append({"type": "linear", "units": units})
            has_linear = True

        elif layer_type == "globalavgpool":
            clean_layers.append({"type": "globalavgpool"})
            has_gap = True

        else:
            clean_layers.append({"type": layer_type})

    if conv_count == 0:
        clean_layers.insert(0, {"type": "conv", "filters": 32, "kernel": 3})
        clean_layers.insert(1, {"type": "relu"})

    if not has_gap:
        clean_layers.append({"type": "globalavgpool"})

    if not has_linear:
        clean_layers.append({"type": "linear", "units": 128})

    return {"layers": clean_layers}



# BUDOWA CNN Z JSON-A

class CNNFromArchitecture(nn.Module):
    def __init__(self, arch, num_classes=10):
        super().__init__()

        layers = []
        in_channels = 3
        current_channels = 3
        classifier_units = 128

        for layer in arch["layers"]:
            layer_type = layer["type"]

            if layer_type == "conv":
                out_channels = int(layer["filters"])
                kernel = int(layer["kernel"])
                padding = kernel // 2

                layers.append(nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel,
                    padding=padding
                ))

                in_channels = out_channels
                current_channels = out_channels

            elif layer_type == "batchnorm":
                layers.append(nn.BatchNorm2d(current_channels))

            elif layer_type == "relu":
                layers.append(nn.ReLU())

            elif layer_type == "maxpool":
                layers.append(nn.MaxPool2d(2))

            elif layer_type == "dropout":
                layers.append(nn.Dropout2d(float(layer["p"])))

            elif layer_type == "globalavgpool":
                layers.append(nn.AdaptiveAvgPool2d((1, 1)))

            elif layer_type == "linear":
                classifier_units = int(layer["units"])

        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(current_channels, classifier_units),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(classifier_units, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def is_valid_architecture(arch):
    try:
        model = CNNFromArchitecture(arch).to(DEVICE)
        dummy = torch.randn(2, 3, 32, 32).to(DEVICE)
        out = model(dummy)
        params = count_parameters(model)

        if out.shape != (2, NUM_CLASSES):
            return False

        if params > PARAM_LIMIT:
            return False

        return True
    except Exception:
        return False


# TRENING I EWALUACJA CNN

def train_and_evaluate_architecture(arch, epochs=EPOCHS_CNN):
    model = CNNFromArchitecture(arch, num_classes=NUM_CLASSES).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        model.train()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    return accuracy



# ENCODING ARCHITEKTURY

LAYER_CODES = {
    "conv": 1,
    "batchnorm": 2,
    "relu": 3,
    "maxpool": 4,
    "dropout": 5,
    "linear": 6,
    "globalavgpool": 7
}


def encode_architecture(arch, max_layers=MAX_LAYERS_ENCODED):
    vector = []

    for layer in arch["layers"][:max_layers]:
        layer_type = layer["type"]
        code = LAYER_CODES.get(layer_type, 0)

        a = 0.0
        b = 0.0

        if layer_type == "conv":
            a = float(layer.get("filters", 0))
            b = float(layer.get("kernel", 0))

        elif layer_type == "dropout":
            a = float(layer.get("p", 0))
            b = 0.0

        elif layer_type == "linear":
            a = float(layer.get("units", 0))
            b = 0.0

        vector.extend([code, a, b])

    while len(vector) < max_layers * 3:
        vector.extend([0.0, 0.0, 0.0])

    return np.array(vector, dtype=np.float32)


# METRYKI

def evaluate_surrogate(y_true, y_pred, name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    corr, _ = spearmanr(y_true, y_pred)
    if np.isnan(corr):
        corr = 0.0

    print(f"\n{name}")
    print(f"MAE:      {mae:.4f}")
    print(f"RMSE:     {rmse:.4f}")
    print(f"R2:       {r2:.4f}")
    print(f"Spearman: {corr:.4f}")

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "spearman": corr
    }


# DATASET ARCHITEKTUR

def generate_valid_architecture():
    for _ in range(5):
        try:
            arch = generate_architecture_with_ollama()
            if is_valid_architecture(arch):
                return arch
        except Exception as e:
            print("Ollama error / invalid JSON, fallback:", e)

    while True:
        arch = generate_random_architecture()
        if is_valid_architecture(arch):
            return arch


def build_initial_architecture_dataset(n_archs=NUM_INITIAL_ARCHS):
    dataset = []

    for i in range(n_archs):
        print("\n" + "-" * 60)
        print(f"Architecture {i + 1}/{n_archs}")

        arch = generate_valid_architecture()
        print(json.dumps(arch, indent=2))

        acc = train_and_evaluate_architecture(arch)
        print(f"Accuracy: {acc:.4f}")

        dataset.append({
            "architecture": arch,
            "accuracy": acc
        })

        with open("architecture_dataset.json", "w", encoding="utf-8") as f:
            json.dump(dataset, f, indent=2)

    return dataset



# SURROGATE MODELS

def train_surrogate_models(dataset):
    X = np.array([encode_architecture(item["architecture"]) for item in dataset])
    y = np.array([item["accuracy"] for item in dataset])

    if len(dataset) < 5:
        raise ValueError("Za mało architektur do treningu surrogate models.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42
    )

    mlp = MLPRegressor(
        hidden_layer_sizes=(128, 64),
        max_iter=1000,
        random_state=42
    )

    mlp.fit(X_train, y_train)
    pred_mlp = mlp.predict(X_test)

    xgb_model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        objective="reg:squarederror",
        random_state=42
    )

    xgb_model.fit(X_train, y_train)
    pred_xgb = xgb_model.predict(X_test)

    metrics_mlp = evaluate_surrogate(y_test, pred_mlp, "MLP Surrogate")
    metrics_xgb = evaluate_surrogate(y_test, pred_xgb, "XGBoost Surrogate")

    return {
        "mlp": mlp,
        "xgb": xgb_model,
        "metrics_mlp": metrics_mlp,
        "metrics_xgb": metrics_xgb
    }



# ITERACYJNY PIPELINE

def iterative_search(dataset, surrogate_pack, iterations=NUM_ITERATIONS):
    xgb_model = surrogate_pack["xgb"]

    best_real = max(item["accuracy"] for item in dataset)
    print("\nCurrent best real accuracy:", best_real)

    for it in range(iterations):
        print("\n" + "=" * 60)
        print(f"ITERATION {it + 1}/{iterations}")
        print("=" * 60)

        candidate = generate_valid_architecture()
        encoded = encode_architecture(candidate).reshape(1, -1)

        predicted_acc = float(xgb_model.predict(encoded)[0])

        print("Candidate architecture:")
        print(json.dumps(candidate, indent=2))
        print(f"Predicted Accuracy: {predicted_acc:.4f}")
        print(f"Current Best Real Accuracy: {best_real:.4f}")

        if predicted_acc > best_real:
            print("Candidate accepted. Training real CNN...")
            real_acc = train_and_evaluate_architecture(candidate)
            print(f"Real Accuracy: {real_acc:.4f}")

            dataset.append({
                "architecture": candidate,
                "accuracy": real_acc
            })

            if real_acc > best_real:
                best_real = real_acc

            with open("architecture_dataset.json", "w", encoding="utf-8") as f:
                json.dump(dataset, f, indent=2)

            print("Retraining surrogate models...")
            surrogate_pack = train_surrogate_models(dataset)
            xgb_model = surrogate_pack["xgb"]

        else:
            print("Candidate rejected by surrogate model.")

    return dataset, surrogate_pack



# WYKRESY
def plot_architecture_results(dataset):
    accuracies = [item["accuracy"] for item in dataset]

    plt.figure(figsize=(10, 4))
    plt.plot(accuracies, marker="o")
    plt.title("Real Accuracy of Evaluated CNN Architectures")
    plt.xlabel("Architecture index")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(7, 4))
    plt.hist(accuracies, bins=10)
    plt.title("Distribution of CNN Accuracy")
    plt.xlabel("Accuracy")
    plt.ylabel("Count")
    plt.show()



# MAIN

if __name__ == "__main__":
    print("\nBuilding initial architecture dataset...")
    dataset = build_initial_architecture_dataset(NUM_INITIAL_ARCHS)

    print("\nTraining surrogate models...")
    surrogate_pack = train_surrogate_models(dataset)

    print("\nStarting iterative architecture search...")
    dataset, surrogate_pack = iterative_search(dataset, surrogate_pack, NUM_ITERATIONS)

    print("\nFinal results:")
    plot_architecture_results(dataset)

    best = max(dataset, key=lambda x: x["accuracy"])
    print("\nBest architecture:")
    print(json.dumps(best["architecture"], indent=2))
    print(f"Best real accuracy: {best['accuracy']:.4f}")