# VERSION MODIFIÉE DU CODE (ANTI-TRICHE)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.feature_extraction.text import TfidfVectorizer

#Chargement des données
USE_HF = False  # mettre True si datasets installé

if USE_HF:
    from datasets import load_dataset
    dataset = load_dataset("imdb")
    X_train = dataset["train"]["text"][:5000]
    y_train = dataset["train"]["label"][:5000]
else:
    import pandas as pd
    df1 = pd.read_csv('imdb_train.csv')  
    X_train = df1['text']
    y_train = df1['label']
    df2 = pd.read_csv('imdb_test.csv')  
    X_test = df2['text']
    y_test = df2['label']


vectorizer = TfidfVectorizer(max_features=10000)
X_train = vectorizer.fit_transform(X_train).toarray()
X_test = vectorizer.transform(X_test).toarray()

# Conversion en tenseurs
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

# DataLoader (batch)
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)

# Modèle MLP
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10000, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)  # BCEWithLogitsLoss => pas de sigmoid ici
        return x

model = MLP()

# ====== Loss (MODIFIÉ) ======
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Entraînement
def train():
    model.train()
    for epoch in range(5):
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
             # Compléter les étapes d'entrainement

        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# Évaluation
def evaluate():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = torch.sigmoid(model(X_batch))
            preds = (outputs > 0.5).float()
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)
    print("Accuracy:", correct / total)

# ====== Run ======
train()
evaluate()
