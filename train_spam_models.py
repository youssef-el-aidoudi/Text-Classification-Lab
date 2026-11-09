import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report


# =========================
# 1. Création des dossiers
# =========================
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)


# =========================
# 2. Chargement du dataset
# =========================
df = pd.read_csv("data/spam.csv", sep="\t", header=None, names=["label", "text"])

print("Aperçu du dataset :")
print(df.head())

print("\nRépartition des classes :")
print(df["label"].value_counts())


# =========================
# 3. Préparation des données
# =========================
X = df["text"]
y = df["label"]

# Découpage train/test
# stratify=y permet de garder la même proportion ham/spam dans train et test
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTaille train :", len(X_train))
print("Taille test :", len(X_test))


# =========================
# 4. Vectorisation TF-IDF
# =========================
vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words="english"
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


# =========================
# 5. Modèles à comparer
# =========================
models = {
    "naive_bayes": MultinomialNB(),
    "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "svm": LinearSVC(class_weight="balanced"),
    "random_forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced"
    )
}


# =========================
# 6. Entraînement + évaluation
# =========================
results = []

for name, model in models.items():
    print("\n==============================")
    print(f"Modèle : {name}")
    print("==============================")

    model.fit(X_train_tfidf, y_train)

    y_pred = model.predict(X_test_tfidf)

    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label="spam")
    recall = recall_score(y_test, y_pred, pos_label="spam")
    f1 = f1_score(y_test, y_pred, pos_label="spam")

    print("Accuracy :", round(acc, 4))
    print("Precision :", round(precision, 4))
    print("Recall :", round(recall, 4))
    print("F1-score :", round(f1, 4))

    print("\nMatrice de confusion :")
    print(confusion_matrix(y_test, y_pred, labels=["ham", "spam"]))

    print("\nRapport de classification :")
    print(classification_report(y_test, y_pred))

    results.append({
        "model": name,
        "accuracy": acc,
        "precision_spam": precision,
        "recall_spam": recall,
        "f1_spam": f1
    })

    # Sauvegarde du modèle
    joblib.dump(model, f"models/spam_{name}.joblib")


# =========================
# 7. Sauvegarde du vectorizer
# =========================
joblib.dump(vectorizer, "models/spam_vectorizer.joblib")


# =========================
# 8. Sauvegarde des résultats
# =========================
results_df = pd.DataFrame(results)
results_df.to_csv("results/comparaison_spam.csv", index=False)

print("\nRésultats comparatifs :")
print(results_df)


# =========================
# 9. Sauvegarde exemples bien/mal classés
# =========================
best_model_name = results_df.sort_values(by="f1_spam", ascending=False).iloc[0]["model"]
best_model = models[best_model_name]

print("\nMeilleur modèle selon F1-score spam :", best_model_name)

y_pred_best = best_model.predict(X_test_tfidf)

examples_df = pd.DataFrame({
    "text": X_test.values,
    "true_label": y_test.values,
    "predicted_label": y_pred_best
})

examples_df["correct"] = examples_df["true_label"] == examples_df["predicted_label"]

correct_examples = examples_df[examples_df["correct"] == True].head(3)
wrong_examples = examples_df[examples_df["correct"] == False].head(2)

final_examples = pd.concat([correct_examples, wrong_examples])
final_examples.to_csv("results/exemples_spam_bien_mal_classes.csv", index=False)

print("\n3 exemples bien classés + 2 mal classés sauvegardés dans :")
print("results/exemples_spam_bien_mal_classes.csv")

print("\nTous les modèles sont sauvegardés dans le dossier models/")