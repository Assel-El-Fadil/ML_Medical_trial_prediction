# Prédiction de l'abandon des essais cliniques

Application de Machine Learning qui estime, à partir des seules métadonnées de conception d'un essai clinique, s'il présente un risque élevé d'être **abandonné** avant sa complétion. Le modèle est entraîné sur des données issues de [ClinicalTrials.gov](https://clinicaltrials.gov/) et exposé via une API REST (FastAPI) et une interface Streamlit.

**Équipe :** El Fadil Assel · El Maaroufi Siham · El Ouazzani Touhami Aymane  
**Module :** Machine Learning — 2ème année GI — 2025-2026

---

## Aperçu des performances (jeu de test, seuil 0.51)

| Métrique | Score | Objectif |
|---|---|---|
| Recall (classe abandonnée) | 0.70 | ≥ 0.70 |
| Precision | 0.37 | ≥ 0.40 |
| F1-score | 0.49 | ≥ 0.50 |
| ROC-AUC | 0.83 | — |
| PR-AUC | 0.61 | — |

Le modèle est calibré pour **détecter ~70 % des abandons** tout en limitant les fausses alertes. Il est conçu comme un outil de **triage précoce**, pas comme un filtre définitif.

---

## Démarrage rapide avec Docker (recommandé)

**Prérequis :** Docker et Docker Compose installés.

```bash
git clone <url-du-depot>
cd Projet_ML
docker-compose up --build
```

| Service | URL |
|---|---|
| API FastAPI | http://localhost:8000 |
| Documentation Swagger | http://localhost:8000/docs |
| Interface Streamlit | http://localhost:8501 |

Arrêter les services : `Ctrl+C` puis `docker-compose down`.

---

## Installation locale (sans Docker)

### 1. Environnement Python

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
pip install streamlit==1.35.0 requests==2.32.3
```

### 2. Lancer l'API

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 3. Lancer l'interface Streamlit (autre terminal)

```bash
streamlit run app/streamlit_app.py
```

---

## Exemple d'utilisation — API

### Vérifier que l'API est opérationnelle

```bash
curl http://localhost:8000/health
```

### Prédiction unitaire

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "phase": "PHASE2",
    "sponsor_type": "INDUSTRY",
    "enrollment_count": 150,
    "intervention_type": "DRUG",
    "n_arms": 2,
    "has_dmc": 1,
    "allocation": "RANDOMIZED",
    "masking": "DOUBLE",
    "primary_purpose": "TREATMENT",
    "n_primary_outcomes": 1,
    "n_secondary_outcomes": 3,
    "n_locations": 10,
    "is_multicenter": 1,
    "has_us_site": 1,
    "n_collaborators": 2
  }'
```

**Réponse attendue :**

```json
{
  "prediction": "abandonne",
  "probability": 0.58,
  "threshold": 0.51,
  "confidence": "medium"
}
```

### Prédiction par lot (CSV)

```bash
curl -X POST http://localhost:8000/predict/batch \
  -F "file=@tests/test_trials.csv" \
  -o predictions.csv
```

### Métadonnées du modèle

```bash
curl http://localhost:8000/model/info
```

---

## Architecture du dépôt

```
Projet_ML/
├── app/
│   ├── main.py              # API FastAPI (endpoints /predict, /health, …)
│   ├── predictor.py         # Chargement du modèle et inférence
│   ├── schemas.py           # Validation Pydantic des entrées
│   └── streamlit_app.py     # Interface utilisateur
├── data/
│   ├── dataset.csv          # Dataset brut collecté (12 272 essais)
│   ├── sample.csv           # Échantillon de démonstration
│   └── processed/           # Jeux train / val / test prétraités
├── models/
│   ├── preprocessor.joblib  # Pipeline de prétraitement (Phase 2)
│   ├── tuned_model.joblib   # Modèle tuné (Phase 3)
│   └── final_model.joblib   # Modèle final + seuil optimal (Phase 3)
├── notebooks/
│   ├── 01_discovery.ipynb   # Phase 1 — collecte et exploration initiale
│   ├── 02_eda.ipynb         # Phase 2 — analyse exploratoire
│   ├── 03_preprocessing.ipynb
│   ├── 04_modeling.ipynb    # Phase 3 — comparaison des modèles
│   ├── 05_tuning.ipynb      # Phase 3 — optimisation des hyperparamètres
│   └── 06_evaluation.ipynb  # Phase 3 — évaluation finale et seuil
├── src/
│   └── data_collection.py   # Script de collecte ClinicalTrials.gov
├── tests/
│   └── test_trials.csv      # Fichier CSV de test pour /predict/batch
├── cadrage.md               # Fiche de cadrage Phase 1
├── preprocessing_decisions.md
├── DATASET.md               # Documentation du dataset
├── report.md                # Rapport final (CRISP-DM)
├── Dockerfile               # Image Docker API
├── Dockerfile.ui            # Image Docker Streamlit
├── docker-compose.yml
└── requirements.txt
```

---

## Endpoints API

| Endpoint | Méthode | Description |
|---|---|---|
| `/` | GET | Informations générales et lien vers Swagger |
| `/health` | GET | État de l'API et du modèle |
| `/model/info` | GET | Type, métriques, seuil de décision |
| `/predict` | POST | Prédiction unitaire (JSON) |
| `/predict/batch` | POST | Prédiction par lot (upload CSV) |
| `/docs` | GET | Documentation Swagger interactive |

---

## Modèle déployé

| Composant | Détail |
|---|---|
| Algorithme | SVM linéaire (`SVC`, kernel=linear, C=0.1) |
| Rééquilibrage | RandomUnderSampler (sampling_strategy=1.0) |
| Prétraitement | ColumnTransformer (imputation + OneHotEncoder + RobustScaler) |
| Seuil de décision | **0.51** (optimisé sur validation : recall ≥ 70 %, précision maximale) |
| Features | 17 variables de design + 4 features engineerées |

---

## Notebooks — ordre d'exécution

1. `01_discovery.ipynb` — Collecte et première exploration
2. `02_eda.ipynb` — EDA approfondie
3. `03_preprocessing.ipynb` — Nettoyage, feature engineering, pipeline
4. `04_modeling.ipynb` — 6 modèles × 3 stratégies de déséquilibre
5. `05_tuning.ipynb` — GridSearchCV sur SVM + RUS
6. `06_evaluation.ipynb` — Évaluation test + optimisation du seuil

---

## Limites connues

- **Signal limité** : seules les métadonnées de conception sont disponibles ; les causes d'abandon (efficacité, sécurité, financement) ne sont pas capturées.
- **Précision modérée** (~37 %) : le modèle génère des fausses alertes ; une revue humaine reste nécessaire pour les essais signalés.
- **Généralisation géographique** : le dataset est dominé par des essais enregistrés sur ClinicalTrials.gov (forte présence US).
- **SVM linéaire** : interprétable via les coefficients, mais moins performant que des modèles ensemblistes (Random Forest) sur le F1 brut.
- **Seuil fixe** : le seuil 0.51 est calibré sur un split historique ; un recalibrage périodique est recommandé.
