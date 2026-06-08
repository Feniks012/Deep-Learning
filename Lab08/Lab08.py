import os
import random
import kagglehub
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

import snntorch as snn
from snntorch import surrogate



# CONFIG


SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 64
EPOCHS = 5
MAX_LEN = 160
MAX_VOCAB = 12000
TFIDF_FEATURES = 1000

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

print("DEVICE:", DEVICE)

# 1. DOWNLOAD DATASET


def download_dataset():
    path = kagglehub.dataset_download("mirzayasirabdullah07/50k-bug-dataset")
    print("Path to dataset files:", path)
    return path


def find_csv_file(path):
    csv_files = []
    for root, dirs, files in os.walk(path):
        for f in files:
            if f.endswith(".csv"):
                csv_files.append(os.path.join(root, f))

    if not csv_files:
        raise FileNotFoundError("Nie znaleziono pliku CSV w pobranym datasiecie.")

    print("CSV files found:")
    for f in csv_files:
        print(" -", f)

    return csv_files[0]


# 2. LOAD AND PREPARE DATA

def prepare_dataframe():
    path = download_dataset()
    csv_path = find_csv_file(path)

    df = pd.read_csv(csv_path)
    print("\nDataset shape:", df.shape)
    print("\nColumns:")
    print(df.columns.tolist())

    if "bug_domain" not in df.columns:
        raise ValueError("Brakuje kolumny 'bug_domain'.")

    if "error_code" not in df.columns:
        print("Uwaga: brak kolumny error_code. Tworzę pustą kolumnę.")
        df["error_code"] = "unknown"

    df = df.dropna(subset=["bug_domain"])
    df["bug_domain"] = df["bug_domain"].astype(str)
    df["error_code"] = df["error_code"].astype(str)

    text_columns = []
    for col in df.columns:
        if col != "bug_domain" and df[col].dtype == "object":
            text_columns.append(col)

    if "error_code" not in text_columns:
        text_columns.append("error_code")

    print("\nText columns used:")
    print(text_columns)

    df["combined_text"] = ""
    for col in text_columns:
        df["combined_text"] += " " + df[col].fillna("").astype(str)

    df["combined_text"] = df["combined_text"].str.lower()

    le = LabelEncoder()
    df["label"] = le.fit_transform(df["bug_domain"])

    print("\nClasses:")
    for i, cls in enumerate(le.classes_):
        print(i, "->", cls)

    return df, le



# 3. TOKENIZER FOR CNN
def build_vocab(texts, max_vocab=MAX_VOCAB):
    counter = {}

    for text in texts:
        for word in str(text).split():
            counter[word] = counter.get(word, 0) + 1

    most_common = sorted(counter.items(), key=lambda x: x[1], reverse=True)[:max_vocab - 2]

    vocab = {
        "<PAD>": 0,
        "<UNK>": 1
    }

    for word, _ in most_common:
        vocab[word] = len(vocab)

    return vocab


def encode_text(text, vocab, max_len=MAX_LEN):
    tokens = str(text).split()
    ids = [vocab.get(tok, vocab["<UNK>"]) for tok in tokens]

    if len(ids) < max_len:
        ids += [vocab["<PAD>"]] * (max_len - len(ids))
    else:
        ids = ids[:max_len]

    return ids


class BugTextDataset(Dataset):
    def __init__(self, texts, labels, error_codes, vocab):
        self.texts = texts
        self.labels = labels
        self.error_codes = error_codes
        self.vocab = vocab

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        x = encode_text(self.texts[idx], self.vocab)
        y = self.labels[idx]
        err = self.error_codes[idx]

        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long), err



# 4. CNN VGG-LIKE

class VGGTextCNN(nn.Module):
    def __init__(self, vocab_size, num_classes, embed_dim=128):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        self.features = nn.Sequential(
            nn.Conv1d(embed_dim, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.embedding(x)
        x = x.transpose(1, 2)
        x = self.features(x)
        return self.classifier(x)


# 5. CNN INCEPTION-LIKE

class InceptionBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        branch_channels = out_channels // 4

        self.b1 = nn.Conv1d(in_channels, branch_channels, kernel_size=1)

        self.b3 = nn.Sequential(
            nn.Conv1d(in_channels, branch_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(branch_channels, branch_channels, kernel_size=3, padding=1)
        )

        self.b5 = nn.Sequential(
            nn.Conv1d(in_channels, branch_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(branch_channels, branch_channels, kernel_size=5, padding=2)
        )

        self.bp = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, branch_channels, kernel_size=1)
        )

        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        x = torch.cat([
            self.b1(x),
            self.b3(x),
            self.b5(x),
            self.bp(x)
        ], dim=1)

        return F.relu(self.bn(x))


class InceptionTextCNN(nn.Module):
    def __init__(self, vocab_size, num_classes, embed_dim=128):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        self.stem = nn.Sequential(
            nn.Conv1d(embed_dim, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )

        self.inception1 = InceptionBlock1D(128, 256)
        self.inception2 = InceptionBlock1D(256, 256)

        self.pool = nn.AdaptiveMaxPool1d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.embedding(x)
        x = x.transpose(1, 2)

        x = self.stem(x)
        x = self.inception1(x)
        x = self.inception2(x)
        x = self.pool(x)

        return self.classifier(x)



# 6. SNN MODEL WITH TF-IDF INPUT


class BugTFIDFDataset(Dataset):
    def __init__(self, X, y, error_codes):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.error_codes = error_codes

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.error_codes[idx]


class SimpleBugSNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_steps=20):
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

        for _ in range(self.num_steps):
            cur1 = self.fc1(x)
            spk1, mem1 = self.lif1(cur1, mem1)

            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)

            spk2_rec.append(spk2)

        return torch.stack(spk2_rec, dim=0)



# 7. TRAINING / EVALUATION FUNCTIONS
def train_cnn_model(model, train_loader, test_loader, label_encoder, name):
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for x, y, _ in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)

            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"[{name}] Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss / len(train_loader):.4f}")

    return evaluate_cnn_model(model, test_loader, label_encoder, name)


def evaluate_cnn_model(model, test_loader, label_encoder, name):
    model.eval()

    y_true = []
    y_pred = []
    error_codes = []

    with torch.no_grad():
        for x, y, err in test_loader:
            x = x.to(DEVICE)
            out = model(x)
            pred = out.argmax(dim=1).cpu().numpy()

            y_pred.extend(pred)
            y_true.extend(y.numpy())
            error_codes.extend(err)

    acc = accuracy_score(y_true, y_pred)

    print(name)
    print("Accuracy:", acc)
    print(classification_report(
        y_true,
        y_pred,
        target_names=label_encoder.classes_,
        zero_division=0
    ))

    return {
        "name": name,
        "accuracy": acc,
        "y_true": np.array(y_true),
        "y_pred": np.array(y_pred),
        "error_codes": np.array(error_codes)
    }


def train_snn_model(model, train_loader, test_loader, label_encoder, name):
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for x, y, _ in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)

            optimizer.zero_grad()
            spk_rec = model(x)
            out = spk_rec.sum(dim=0)

            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"[{name}] Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss / len(train_loader):.4f}")

    return evaluate_snn_model(model, test_loader, label_encoder, name)


def evaluate_snn_model(model, test_loader, label_encoder, name):
    model.eval()

    y_true = []
    y_pred = []
    error_codes = []

    with torch.no_grad():
        for x, y, err in test_loader:
            x = x.to(DEVICE)
            spk_rec = model(x)
            out = spk_rec.sum(dim=0)

            pred = out.argmax(dim=1).cpu().numpy()

            y_pred.extend(pred)
            y_true.extend(y.numpy())
            error_codes.extend(err)

    acc = accuracy_score(y_true, y_pred)

    print(name)
    print("Accuracy:", acc)
    print(classification_report(
        y_true,
        y_pred,
        target_names=label_encoder.classes_,
        zero_division=0
    ))

    return {
        "name": name,
        "accuracy": acc,
        "y_true": np.array(y_true),
        "y_pred": np.array(y_pred),
        "error_codes": np.array(error_codes)
    }



# 8. VISUALIZATIONS


def plot_accuracy_comparison(results):
    names = [r["name"] for r in results]
    accs = [r["accuracy"] for r in results]

    plt.figure(figsize=(8, 5))
    plt.bar(names, accs)
    plt.title("Porównanie accuracy modeli")
    plt.ylabel("Accuracy")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(result, label_encoder):
    cm = confusion_matrix(result["y_true"], result["y_pred"])

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=False,
        cmap="Blues",
        xticklabels=label_encoder.classes_,
        yticklabels=label_encoder.classes_
    )
    plt.title(f"Confusion Matrix - {result['name']}")
    plt.xlabel("Predicted bug_domain")
    plt.ylabel("True bug_domain")
    plt.tight_layout()
    plt.show()


def show_prediction_examples(result, df_test, label_encoder, n=10):
    print(f"Przykładowe predykcje — {result['name']}")

    y_true = result["y_true"]
    y_pred = result["y_pred"]
    error_codes = result["error_codes"]

    for i in range(min(n, len(y_true))):
        true_label = label_encoder.inverse_transform([y_true[i]])[0]
        pred_label = label_encoder.inverse_transform([y_pred[i]])[0]
        err = error_codes[i]

        print(f"\nPróbka {i+1}")
        print(f"error_code: {err}")
        print(f"true bug_domain: {true_label}")
        print(f"pred bug_domain: {pred_label}")



# 9. MAIN

def main():
    df, label_encoder = prepare_dataframe()

    # żeby szybciej działało — możesz zwiększyć np. do 50000
    if len(df) > 20000:
        df = df.sample(20000, random_state=SEED).reset_index(drop=True)

    texts = df["combined_text"].values
    labels = df["label"].values
    error_codes = df["error_code"].values

    X_train_text, X_test_text, y_train, y_test, err_train, err_test = train_test_split(
        texts,
        labels,
        error_codes,
        test_size=0.2,
        random_state=SEED,
        stratify=labels
    )

    # CNN DATA

    vocab = build_vocab(X_train_text)

    train_cnn_ds = BugTextDataset(X_train_text, y_train, err_train, vocab)
    test_cnn_ds = BugTextDataset(X_test_text, y_test, err_test, vocab)

    train_cnn_loader = DataLoader(train_cnn_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_cnn_loader = DataLoader(test_cnn_ds, batch_size=BATCH_SIZE, shuffle=False)


    # SNN DATA — TF-IDF

    vectorizer = TfidfVectorizer(
        max_features=TFIDF_FEATURES,
        stop_words="english"
    )

    X_train_tfidf = vectorizer.fit_transform(X_train_text).toarray()
    X_test_tfidf = vectorizer.transform(X_test_text).toarray()

    train_snn_ds = BugTFIDFDataset(X_train_tfidf, y_train, err_train)
    test_snn_ds = BugTFIDFDataset(X_test_tfidf, y_test, err_test)

    train_snn_loader = DataLoader(train_snn_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_snn_loader = DataLoader(test_snn_ds, batch_size=BATCH_SIZE, shuffle=False)

    num_classes = len(label_encoder.classes_)
    vocab_size = len(vocab)

    print("\nVocab size:", vocab_size)
    print("Num classes:", num_classes)


    # MODEL 1 — VGG-like CNN

    vgg_model = VGGTextCNN(
        vocab_size=vocab_size,
        num_classes=num_classes
    )

    result_vgg = train_cnn_model(
        vgg_model,
        train_cnn_loader,
        test_cnn_loader,
        label_encoder,
        name="VGG-like CNN"
    )


    # MODEL 2 — Inception-like CNN
    inception_model = InceptionTextCNN(
        vocab_size=vocab_size,
        num_classes=num_classes
    )

    result_inception = train_cnn_model(
        inception_model,
        train_cnn_loader,
        test_cnn_loader,
        label_encoder,
        name="Inception-like CNN"
    )

    # MODEL 3 — Spiking Neural Network

    snn_model = SimpleBugSNN(
        input_size=TFIDF_FEATURES,
        hidden_size=256,
        num_classes=num_classes,
        num_steps=20
    )

    result_snn = train_snn_model(
        snn_model,
        train_snn_loader,
        test_snn_loader,
        label_encoder,
        name="Spiking Neural Network"
    )

    results = [result_vgg, result_inception, result_snn]

    # VISUALIZATIONS

    plot_accuracy_comparison(results)

    for result in results:
        plot_confusion_matrix(result, label_encoder)
        show_prediction_examples(result, None, label_encoder, n=10)


    print("PODSUMOWANIE")

    for result in results:
        print(f"{result['name']}: Accuracy = {result['accuracy']:.4f}")


if __name__ == "__main__":
    main()