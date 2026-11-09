"""
Exporte le modèle de détection de spam vers un fichier JSON lisible par le navigateur.

La page docs/index.html refait le calcul en JavaScript : TF-IDF puis régression
logistique. Ce script écrit tout ce dont elle a besoin (vocabulaire, IDF,
coefficients) dans docs/model/spam_model.json.

Le modèle exporté est la régression logistique, car elle donne une probabilité
et non un simple verdict, ce qui permet d'afficher un niveau de confiance et le
poids de chaque mot dans la décision.

Usage :
    python export_model_for_web.py [chemin/vers/spam.csv]
"""

import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

DATA = sys.argv[1] if len(sys.argv) > 1 else "data/spam.csv"
OUT = os.path.join("docs", "model", "spam_model.json")
SEED = 42

# 1. Mêmes données et même découpage que train_spam_models.py
df = pd.read_csv(DATA, sep="\t", header=None, names=["label", "text"])
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=SEED, stratify=df["label"]
)

# 2. Mêmes réglages de vectorisation
vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 3. Même modèle
model = LogisticRegression(max_iter=1000, class_weight="balanced")
model.fit(X_train_tfidf, y_train)

y_pred = model.predict(X_test_tfidf)
scores = {
    "accuracy": accuracy_score(y_test, y_pred),
    "precision": precision_score(y_test, y_pred, pos_label="spam"),
    "recall": recall_score(y_test, y_pred, pos_label="spam"),
    "f1": f1_score(y_test, y_pred, pos_label="spam"),
}
print("Scores sur le jeu de test :")
for k, v in scores.items():
    print(f"  {k:10s} {v:.4f}")

# 4. Le vecteur de coefficients porte sur la classe « spam »
assert list(model.classes_) == ["ham", "spam"], model.classes_

vocab = vectorizer.vocabulary_                      # terme -> indice de colonne
terms = [""] * len(vocab)
for term, idx in vocab.items():
    terms[idx] = term

payload = {
    "model": "logistic_regression",
    "trained_on": os.path.basename(DATA),
    "n_train": int(X_train.shape[0]),
    "n_test": int(X_test.shape[0]),
    "scores": {k: round(float(v), 6) for k, v in scores.items()},
    # Le JavaScript refait : minuscules, découpage en mots de 2 caractères ou plus,
    # comptage, multiplication par l'IDF, normalisation L2, puis produit scalaire.
    "token_pattern": r"\b\w\w+\b",
    "terms": terms,
    "idf": [round(float(v), 6) for v in vectorizer.idf_],
    "coef": [round(float(v), 6) for v in model.coef_[0]],
    "intercept": round(float(model.intercept_[0]), 6),
}

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))

print(f"\n{len(terms)} termes exportés vers {OUT} ({os.path.getsize(OUT) / 1024:.0f} Ko)")

# 5. Vérification : les valeurs arrondies doivent redonner les mêmes prédictions
idf = np.array(payload["idf"])
coef = np.array(payload["coef"])
analyzer = vectorizer.build_analyzer()
index = {t: i for i, t in enumerate(terms)}
ok = 0
for text, truth in zip(X_test, y_pred):
    vec = np.zeros(len(terms))
    for tok in analyzer(text):
        j = index.get(tok)
        if j is not None:
            vec[j] += 1.0
    vec *= idf
    norm = np.linalg.norm(vec)
    if norm:
        vec /= norm
    guess = "spam" if vec @ coef + payload["intercept"] > 0 else "ham"
    ok += guess == truth
print(f"Vérification : {ok}/{len(y_pred)} prédictions identiques après arrondi.")
