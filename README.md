# Text-Classification-Lab

Ce TP compare plusieurs méthodes d'IA pour classer des textes, sur deux cas :

1. **Détection de spam** (SMS) avec des modèles classiques de scikit-learn.
2. **Analyse de sentiment** (critiques de films IMDb) avec un réseau de neurones MLP écrit en PyTorch.

Dans les deux cas, le texte est transformé en nombres avec **TF-IDF**, puis les modèles sont comparés avec les mêmes métriques : accuracy, precision, recall et F1-score.

## Contenu du projet

| Fichier | Rôle |
|---|---|
| `train_spam_models.py` | Entraîne et compare 4 modèles de détection de spam, sauvegarde les modèles et les résultats |
| `test_spam.py` | Vérifie rapidement que le jeu de données se charge bien |
| `partie1_mlp_imdb.py` | MLP PyTorch sur IMDb, avec 4 expériences |
| `partie2_semi_supervise_imdb.py` | Apprentissage semi-supervisé sur IMDb *(à venir)* |
| `partie3_comparaison_modeles.py` | Comparaison finale des modèles *(à venir)* |
| `data/spam.csv` | Jeu de données SMS (colonnes : label, texte, séparées par une tabulation) |
| `results/` | Tableaux de résultats en CSV |

## Installation

```bash
python -m venv .venv
source .venv/bin/activate        # Windows : .venv\Scripts\activate
pip install -r requirements.txt
```

## Lancer les scripts

```bash
python train_spam_models.py      # partie spam
python partie1_mlp_imdb.py       # partie IMDb (télécharge IMDb via HuggingFace)
```

Les modèles entraînés sont enregistrés dans `models/` (non inclus dans le dépôt : il suffit de relancer le script pour les recréer). Les résultats sont écrits dans `results/`.

## Partie spam : détection de SMS indésirables

- Données séparées en 80 % entraînement et 20 % test (`stratify`, pour garder la même proportion de spam et de ham).
- Vectorisation TF-IDF : 5000 mots maximum, sans les mots vides anglais.
- Quatre modèles comparés. Les métriques precision, recall et F1 sont calculées sur la classe **spam**.

| Modèle | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Naive Bayes | 0.972 | 1.000 | 0.792 | 0.884 |
| Régression logistique | 0.977 | 0.913 | 0.913 | 0.913 |
| **SVM linéaire** | **0.981** | 0.957 | 0.899 | **0.927** |
| Random Forest | 0.976 | 1.000 | 0.819 | 0.900 |

**Lecture des résultats**

- Le **SVM linéaire** donne le meilleur F1 (0,927) et la meilleure accuracy.
- **Naive Bayes** et **Random Forest** ont une precision de 100 % : ils ne classent jamais un vrai message comme spam. Mais ils laissent passer beaucoup de spams (recall de 79 % et 82 %).
- La **régression logistique** est la plus équilibrée entre precision et recall.
- Le choix dépend du besoin : si on veut éviter à tout prix de bloquer un vrai message, on préfère une precision élevée. Si on veut attraper le plus de spams possible, on regarde le recall.

Le script sauvegarde aussi 3 exemples bien classés et 2 mal classés (`results/exemples_spam_bien_mal_classes.csv`). Par exemple, un spam de type « voucher holder… call… » n'est pas détecté, alors qu'un message normal très court (« Alright i have a new goal now ») est pris à tort pour du spam.

## Partie 1 : MLP sur IMDb (sentiment positif / négatif)

- Données : sous-échantillon équilibré de IMDb : 4000 critiques pour l'entraînement (2000 négatives, 2000 positives) et 1000 pour le test.
- Vectorisation TF-IDF avec 10 000 variables.
- Réseau MLP : 10 000 → 256 → 128 → 64 → 1, avec BatchNorm sur la première couche et Dropout (0,3) en option.
- Optimiseur Adam, loss `BCEWithLogitsLoss`, graine fixée (`SEED = 42`) pour pouvoir refaire les résultats.

| Expérience | Batch | Epochs | LR | Dropout | Loss pondérée | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|---|---|
| MLP de base | 64 | 3 | 0.001 | non | non | 0.825 | 0.852 | 0.786 | 0.818 |
| **Hyperparamètres adaptés** | 32 | 4 | 0.0005 | non | non | **0.846** | 0.853 | 0.836 | **0.844** |
| Avec dropout | 32 | 4 | 0.0005 | oui | non | 0.829 | 0.839 | 0.814 | 0.826 |
| Loss pondérée | 32 | 4 | 0.0005 | oui | oui | 0.839 | 0.847 | 0.828 | 0.837 |

**Lecture des résultats**

- Le meilleur résultat vient du réglage des hyperparamètres (batch plus petit, learning rate plus faible, un epoch de plus) : F1 de 0,844 contre 0,818 pour le modèle de base.
- Le dropout n'améliore pas le score ici. Avec seulement 4 epochs et peu de données, le modèle ne sur-apprend pas encore beaucoup.
- Les classes sont parfaitement équilibrées (2000 / 2000), donc le poids de la loss pondérée vaut 1 : elle est identique à la loss simple. La différence entre les deux dernières lignes vient surtout du hasard de l'entraînement, pas d'un vrai effet de la pondération. Cette option servirait surtout sur des données déséquilibrées, comme dans le cas du spam.
- Les scores sont calculés sur un seul entraînement par expérience : de petits écarts (1 à 2 points) ne sont pas forcément significatifs.

## Suite du travail

- Partie 2 : apprentissage semi-supervisé sur IMDb.
- Partie 3 : comparaison finale des modèles.

## Technologies

Python, PyTorch, scikit-learn, pandas, NumPy, HuggingFace `datasets`, joblib.
