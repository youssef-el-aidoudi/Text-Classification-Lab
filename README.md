# Text-Classification-Lab

**Du texte brut à une décision : spam ou pas spam, avis positif ou négatif.**

[![Voir la démo et les résultats](https://img.shields.io/badge/%F0%9F%93%8A_D%C3%A9mo_et_r%C3%A9sultats-en_ligne-1d4ed8?style=for-the-badge)](https://youssef-el-aidoudi.github.io/Text-Classification-Lab/)

Ce projet (TP de Master BDIA) compare plusieurs méthodes d'IA pour classer des textes, sur deux cas :

1. **Détection de spam** (SMS) avec quatre modèles classiques de scikit-learn.
2. **Analyse de sentiment** (critiques de films IMDb) avec un réseau de neurones MLP écrit en PyTorch.

Dans les deux cas, le texte est transformé en nombres avec **TF-IDF**, puis les modèles sont comparés avec les mêmes mesures : accuracy, precision, recall et F1-score.

La page en ligne va plus loin que les tableaux : le modèle de détection de spam **tourne directement dans le navigateur**. On écrit un SMS, il répond, et il montre quels mots ont fait pencher la décision.

## Contenu du projet

| Fichier | Rôle |
|---|---|
| `train_spam_models.py` | Entraîne et compare les 4 modèles de détection de spam, sauvegarde les modèles et les résultats |
| `test_spam.py` | Vérifie rapidement que le jeu de données se charge bien |
| `partie1_mlp_imdb.py` | MLP PyTorch sur IMDb, avec 4 expériences |
| `export_model_for_web.py` | Exporte le modèle de spam (vocabulaire, IDF, coefficients) vers `docs/model/spam_model.json` pour la démo |
| `partie2_semi_supervise_imdb.py` | Apprentissage semi-supervisé sur IMDb *(à venir)* |
| `partie3_comparaison_modeles.py` | Comparaison finale des modèles *(à venir)* |
| `data/spam.csv` | Jeu de données SMS (label et texte, séparés par une tabulation) |
| `results/` | Tableaux de résultats en CSV |
| `docs/` | Page web des résultats (HTML, CSS, JavaScript, Chart.js), publiée avec GitHub Pages |

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
python export_model_for_web.py   # régénère le modèle de la démo web
```

Les modèles entraînés sont enregistrés dans `models/` (exclus du dépôt : il suffit de relancer le script pour les recréer). Les résultats sont écrits dans `results/`.

## Partie spam : détection de SMS indésirables

- 5 572 SMS, séparés en 80 % entraînement (4 457) et 20 % test (1 115), avec `stratify` pour garder la même proportion de spam dans les deux parties.
- Vectorisation TF-IDF : 5 000 mots au maximum, sans les mots vides anglais.
- Quatre modèles comparés. Precision, recall et F1 portent sur la classe **spam**.

| Modèle | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Naive Bayes | 97,2 % | 100,0 % | 79,2 % | 88,4 % |
| Régression logistique | 97,7 % | 91,3 % | 91,3 % | 91,3 % |
| **SVM linéaire** | **98,1 %** | 95,7 % | 90,0 % | **92,7 %** |
| Random Forest | 97,6 % | 100,0 % | 81,9 % | 90,0 % |

### Ce que ça donne en nombre de messages

Le jeu de test contient 1 115 SMS : 966 messages normaux et 149 spams.

| Modèle | Vrais messages bloqués à tort | Spams non détectés |
|---|---|---|
| Naive Bayes | 0 | 31 |
| Régression logistique | 13 | 13 |
| SVM linéaire | 6 | 15 |
| Random Forest | 0 | 27 |

**Lecture des résultats**

- Le **SVM linéaire** donne le meilleur F1 et la meilleure accuracy : il bloque 134 spams sur 149 en ne se trompant que 6 fois sur les messages normaux.
- **Naive Bayes** et **Random Forest** ne bloquent jamais un vrai message, mais ils laissent passer beaucoup plus de spams (31 et 27).
- La **régression logistique** est la plus équilibrée : autant de fausses alertes que de spams ratés.
- Le choix dépend donc de l'usage. Si bloquer un vrai message coûte très cher, la precision prime. Si l'important est d'attraper le plus de spams, c'est le recall.

Le script sauvegarde aussi des exemples bien et mal classés (`results/exemples_spam_bien_mal_classes.csv`). Un spam du type « voucher holder… call… » passe à travers, alors qu'un message normal très court (« Alright i have a new goal now ») est pris pour du spam.

## La démo web

`export_model_for_web.py` réentraîne la régression logistique et écrit son vocabulaire, ses valeurs IDF et ses coefficients dans un fichier JSON. La page refait ensuite le même calcul en JavaScript : passage en minuscules, découpage en mots, TF-IDF, normalisation L2, produit scalaire, sigmoïde.

Le portage a été vérifié sur les 1 115 SMS du jeu de test : **le navigateur et scikit-learn rendent le même verdict dans 100 % des cas**, avec un écart de probabilité d'au plus 6 × 10⁻⁷ dû à l'arrondi des coefficients.

La régression logistique a été retenue plutôt que le SVM parce qu'elle donne une probabilité, ce qui permet d'afficher un niveau de confiance et le poids de chaque mot dans la décision.

## Partie 1 : MLP sur IMDb (sentiment positif ou négatif)

- Sous-échantillon équilibré d'IMDb : 4 000 critiques pour l'entraînement (2 000 négatives, 2 000 positives) et 1 000 pour le test.
- Vectorisation TF-IDF avec 10 000 variables.
- Réseau MLP : 10 000 → 256 → 128 → 64 → 1, avec BatchNorm sur la première couche et Dropout (0,3) en option.
- Optimiseur Adam, loss `BCEWithLogitsLoss`, graine fixée (`SEED = 42`).

| Expérience | Batch | Epochs | LR | Dropout | Loss pondérée | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|---|---|
| MLP de base | 64 | 3 | 0,001 | non | non | 82,5 % | 85,2 % | 78,6 % | 81,8 % |
| **Hyperparamètres adaptés** | 32 | 4 | 0,0005 | non | non | **84,6 %** | 85,3 % | 83,6 % | **84,4 %** |
| Avec dropout | 32 | 4 | 0,0005 | oui | non | 82,9 % | 83,9 % | 81,4 % | 82,6 % |
| Loss pondérée | 32 | 4 | 0,0005 | oui | oui | 83,9 % | 84,7 % | 82,8 % | 83,7 % |

**Lecture des résultats**

- Le meilleur résultat vient du réglage des hyperparamètres (batch plus petit, learning rate plus faible, un epoch de plus) : F1 de 84,4 % contre 81,8 % pour le modèle de base.
- Le dropout n'améliore pas le score ici. Avec seulement 4 epochs et peu de données, le modèle ne sur-apprend pas encore beaucoup.
- Les classes sont parfaitement équilibrées (2 000 / 2 000), donc le poids calculé pour la loss pondérée vaut 1 : elle revient à la loss simple. L'écart entre les deux dernières lignes vient du hasard de l'entraînement, pas de la pondération. Cette option servirait sur des données déséquilibrées, comme les SMS.
- Chaque expérience n'a été lancée qu'une fois : un écart de 1 à 2 points n'est pas forcément significatif.

## Limites connues

- Un seul entraînement par expérience sur IMDb, donc pas de mesure de la variabilité.
- Le réseau a appris sur 4 000 critiques seulement, et non sur les 50 000 du jeu complet.
- Le modèle de la démo ne connaît que l'anglais, et seulement le vocabulaire de SMS de 2012.
- TF-IDF compte les mots mais ignore leur ordre : « pas du tout bon » et « pas du tout mauvais » se ressemblent beaucoup pour lui.

## Suite du travail

- Partie 2 : apprentissage semi-supervisé sur IMDb.
- Partie 3 : comparaison finale des modèles.

## Technologies

Python, PyTorch, scikit-learn, pandas, NumPy, HuggingFace `datasets`, joblib. Page web en HTML, CSS et JavaScript, avec Chart.js — sans aucune dépendance chargée depuis un service externe.

---

## Auteur

**Youssef EL AIDOUDI** — M2 Bases de Données et Intelligence Artificielle (BDIA), Université Bourgogne Europe, Dijon.

À la recherche d'un stage de fin d'études de 4 à 6 mois en data (analyse, ingénierie ou BI), disponible à partir d'avril 2027.

[GitHub](https://github.com/youssef-el-aidoudi) · [LinkedIn](https://www.linkedin.com/in/youssef-el-aidoudi) · [youssefelaidoudi20@gmail.com](mailto:youssefelaidoudi20@gmail.com)
