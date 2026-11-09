import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from datasets import load_dataset
from torch.utils.data import DataLoader, TensorDataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


os.makedirs("results", exist_ok=True)

SEED = 42
MAX_FEATURES = 10000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

print("Device utilisé :", DEVICE)
print("Chargement du dataset IMDb depuis HuggingFace...")

dataset = load_dataset("imdb")

train_texts = list(dataset["train"]["text"])
train_labels = list(dataset["train"]["label"])

test_texts = list(dataset["test"]["text"])
test_labels = list(dataset["test"]["label"])

train_df = pd.DataFrame({
    "text": train_texts,
    "label": train_labels
})

test_df = pd.DataFrame({
    "text": test_texts,
    "label": test_labels
})

train_neg = train_df[train_df["label"] == 0].sample(n=2000, random_state=SEED)
train_pos = train_df[train_df["label"] == 1].sample(n=2000, random_state=SEED)

test_neg = test_df[test_df["label"] == 0].sample(n=500, random_state=SEED)
test_pos = test_df[test_df["label"] == 1].sample(n=500, random_state=SEED)

train_df = pd.concat([train_neg, train_pos]).sample(frac=1, random_state=SEED).reset_index(drop=True)
test_df = pd.concat([test_neg, test_pos]).sample(frac=1, random_state=SEED).reset_index(drop=True)

X_train_text = train_df["text"].astype(str).tolist()
y_train = train_df["label"].astype(int).tolist()

X_test_text = test_df["text"].astype(str).tolist()
y_test = test_df["label"].astype(int).tolist()

print("Colonnes train :", train_df.columns.tolist())
print("Nombre d'exemples train :", len(X_train_text))
print("Nombre d'exemples test :", len(X_test_text))
print("Répartition train :")
print(train_df["label"].value_counts())

vectorizer = TfidfVectorizer(max_features=MAX_FEATURES)

X_train = vectorizer.fit_transform(X_train_text).toarray()
X_test = vectorizer.transform(X_test_text).toarray()

input_dim = X_train.shape[1]

print("Nombre de caractéristiques TF-IDF :", input_dim)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)


class MLP(nn.Module):
    def __init__(self, input_dim, use_dropout=False):
        super().__init__()

        self.use_dropout = use_dropout
        self.fc1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))

        if self.use_dropout:
            x = self.dropout(x)

        x = torch.relu(self.fc2(x))

        if self.use_dropout:
            x = self.dropout(x)

        x = torch.relu(self.fc3(x))
        x = self.fc4(x)

        return x


def train_model(model, train_loader, criterion, optimizer, epochs):
    model.train()

    for epoch in range(epochs):
        total_loss = 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{epochs} - Loss : {avg_loss:.4f}")


def evaluate_model(model, test_loader):
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(DEVICE)

            outputs = model(X_batch)
            probabilities = torch.sigmoid(outputs)
            predictions = (probabilities > 0.5).float().cpu().numpy()

            y_pred.extend(predictions.flatten())
            y_true.extend(y_batch.numpy().flatten())

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    return accuracy, precision, recall, f1


def run_experiment(name, batch_size, epochs, lr, use_dropout=False, weighted_loss=False):
    print("\n======================================")
    print("Expérience :", name)
    print("======================================")

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    model = MLP(input_dim=input_dim, use_dropout=use_dropout).to(DEVICE)

    if weighted_loss:
        nb_neg = (y_train_tensor == 0).sum().item()
        nb_pos = (y_train_tensor == 1).sum().item()
        pos_weight = torch.tensor([nb_neg / nb_pos], dtype=torch.float32).to(DEVICE)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        loss_name = "BCEWithLogitsLoss pondérée"
    else:
        criterion = nn.BCEWithLogitsLoss()
        loss_name = "BCEWithLogitsLoss simple"

    print("Loss utilisée :", loss_name)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_model(model, train_loader, criterion, optimizer, epochs)

    accuracy, precision, recall, f1 = evaluate_model(model, test_loader)

    print("Accuracy :", round(accuracy, 4))
    print("Precision :", round(precision, 4))
    print("Recall :", round(recall, 4))
    print("F1-score :", round(f1, 4))

    return {
        "experience": name,
        "batch_size": batch_size,
        "epochs": epochs,
        "learning_rate": lr,
        "dropout": use_dropout,
        "weighted_loss": weighted_loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }


results = []

results.append(
    run_experiment(
        name="MLP_base",
        batch_size=64,
        epochs=3,
        lr=0.001,
        use_dropout=False,
        weighted_loss=False
    )
)

results.append(
    run_experiment(
        name="MLP_hyperparametres_adaptes",
        batch_size=32,
        epochs=4,
        lr=0.0005,
        use_dropout=False,
        weighted_loss=False
    )
)

results.append(
    run_experiment(
        name="MLP_avec_dropout",
        batch_size=32,
        epochs=4,
        lr=0.0005,
        use_dropout=True,
        weighted_loss=False
    )
)

results.append(
    run_experiment(
        name="MLP_loss_ponderee",
        batch_size=32,
        epochs=4,
        lr=0.0005,
        use_dropout=True,
        weighted_loss=True
    )
)

results_df = pd.DataFrame(results)
results_df.to_csv("results/partie1_mlp_imdb_results.csv", index=False)

print("\n======================================")
print("Tableau comparatif final")
print("======================================")
print(results_df)

print("\nRésultats sauvegardés dans : results/partie1_mlp_imdb_results.csv")