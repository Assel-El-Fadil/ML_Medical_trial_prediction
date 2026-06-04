# Décisions de Prétraitement — Phase 2
## Projet : Prédiction d'Abandon d'Essais Cliniques

| Variable | Type | % Manquant | Action | Justification |
|---|---|---|---|---|
| phase | Catégorique | 48.7% | Remplacer NaN → catégorie "NA" | L'absence de phase est une information pertinente (les études observationnelles/précoces n'ont pas de phase) ; traitée comme une modalité valide plutôt que d'imputer une fausse phase. |
| has_dmc | Binaire | 16.5% | Remplacer NaN → mode (0) | La plupart des essais n'ont pas de DMC ; l'imputation par le mode préserve le signal dominant ; appliquée via le pipeline sur les statistiques d'entraînement (train) uniquement. |
| enrollment_count | Numérique | 3.3% | Remplacer NaN → médiane | La médiane est robuste à la forte asymétrie à droite des nombres de participants ; appliquée via le pipeline sur les statistiques d'entraînement (train) uniquement. |
| phase | Catégorique | — | OneHotEncoder (drop='first') | Variable nominale, sans ordre naturel, 8 modalités → encodage OHE approprié. |
| sponsor_type | Catégorique | — | OneHotEncoder (drop='first') | Variable nominale, 7 modalités. |
| intervention_type | Catégorique | — | OneHotEncoder (drop='first') | Variable nominale, 9 modalités. |
| allocation | Catégorique | — | OneHotEncoder (drop='first') | Variable nominale, 2 modalités. |
| masking | Catégorique | — | OneHotEncoder (drop='first') | Variable nominale, 5 modalités (NONE/SINGLE/DOUBLE/TRIPLE/QUADRUPLE — traitée comme nominale car le niveau de masquage n'a pas d'ordre universel lié au risque d'abandon). |
| primary_purpose | Catégorique | — | OneHotEncoder (drop='first') | Variable nominale, 8 modalités. |
| enrollment_count | Numérique | — | RobustScaler + Winsorize 1–99% | Forte asymétrie à droite + valeurs aberrantes ; le RobustScaler utilise la médiane/l'écart interquartile (IQR), ce qui le rend robuste aux extrêmes ; la winsorisation plafonne les erreurs de données. |
| n_arms | Numérique | — | RobustScaler | Présence de valeurs aberrantes ; les modèles arborescents n'ont pas besoin de mise à l'échelle, mais le pipeline est conçu pour être compatible avec les modèles de régression logistique et SVM. |
| n_primary_outcomes | Numérique | — | RobustScaler | Même justification. |
| n_secondary_outcomes | Numérique | — | RobustScaler | Même justification. |
| outcomes_ratio | Numérique | — | RobustScaler | Ratio dérivé, peut présenter des valeurs extrêmes. |
| n_locations | Numérique | — | RobustScaler | Asymétrie à droite, certains essais comptent plus de 1000 sites. |
| n_collaborators | Numérique | — | RobustScaler | Distribution éparse, beaucoup de zéros. |
| has_dmc | Binaire | — | Passthrough | Déjà codée 0/1, aucune transformation nécessaire. |
| is_multicenter | Binaire | — | Passthrough | Déjà codée 0/1. |
| has_us_site | Binaire | — | Passthrough | Déjà codée 0/1. |
| log_enrollment | Ingénierie | — | RobustScaler | log1p(enrollment_count) — normalise l'asymétrie. |
| log_n_locations | Ingénierie | — | RobustScaler | log1p(n_locations) — normalise l'asymétrie. |
| protocol_complexity | Ingénierie | — | RobustScaler | n_arms × n_primary_outcomes — caractéristique d'interaction. |
| enrollment_per_site | Ingénierie | — | RobustScaler | enrollment_count / (n_locations+1) — ratio de pression de recrutement. |

## Traitement des Valeurs Aberrantes (Outliers)
La **winsorisation au 1er et 99e centile** a été choisie plutôt que la suppression de lignes. Les valeurs extrêmes dans les essais cliniques sont réelles (par exemple, les grands essais de Phase 3 comptant plus de 10 000 patients) et contiennent un fort signal prédictif. Plafonner ces valeurs permet de limiter la distorsion des modèles tout en conservant les données.

## Prévention des Fuites de Données (Data Leakage)
Toutes les statistiques d'imputation (médiane, mode) et les paramètres de mise à l'échelle (médiane et IQR pour le RobustScaler) ont été ajustés (*fitted*) **EXCLUSIVEMENT sur `X_train`**. Ils ont ensuite été appliqués sur `X_val` et `X_test` via le pipeline ajusté. Cette étape s'assure qu'aucune information des jeux de test ou de validation n'est utilisée lors de la préparation des données.

## Division des Données (Train / Validation / Test Split)
- **Répartition :** 70% Entraînement / 15% Validation / 15% Test
- L'échantillonnage a été **stratifié sur la variable cible `abandoned`** afin de préserver rigoureusement le ratio d'environ 86% / 14% dans les trois sous-ensembles.
- La graine aléatoire a été fixée à `random_state=42`.

## Stratégies de Déséquilibre (Imbalance) Préparées pour la Phase 3
1. **Pas de rééchantillonnage :** Utilisation de `class_weight='balanced'` directement dans les modèles.
2. **SMOTE :** Création de données synthétiques pour la classe minoritaire par interpolation entre les plus proches voisins.
3. **RandomUnderSampler :** Sous-échantillonnage de la classe majoritaire par suppression aléatoire pour équilibrer les classes.
4. **SMOTETomek (Bonus) :** Combinaison hybride de création de données synthétiques (SMOTE) et de nettoyage des frontières (Liens Tomek).
