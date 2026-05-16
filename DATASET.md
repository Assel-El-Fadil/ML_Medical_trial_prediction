# DATASET.md — Documentation du Dataset
## Prédiction de l'abandon des essais cliniques

---

## a) Identification

| Champ | Valeur |
|---|---|
| **Nom** | ClinicalTrials Dataset |
| **Auteur(s)** | - El Fadil Assel    - El Maaroufi Siham    - El Ouazzani Touhami Aymane |
| **Date de collecte** | Mai 2025 |
| **Version** | 1.0 |
| **Fichier principal** | `data/dataset.csv` |
| **Extrait** | `data/sample.csv` (100 lignes) |

---

## b) Source

| Champ | Détail |
|---|---|
| **API** | ClinicalTrials.gov API v2 |
| **URL de base** | `https://clinicaltrials.gov/api/v2/studies` |
| **Authentification** | Aucune (API publique et gratuite) |
| **Documentation** | https://clinicaltrials.gov/data-api/api |
| **Date d'accès** | Mai 2025 |

### Endpoints interrogés

```
GET https://clinicaltrials.gov/api/v2/studies
    ?format=json
    &pageSize=1000
    &filter.overallStatus=COMPLETED|TERMINATED|SUSPENDED|WITHDRAWN
```

### Filtre appliqué

Seuls les essais avec un statut **final** parmi les 4 valeurs suivantes ont été récupérés :

| Statut API | Signification |
|---|---|
| `COMPLETED` | Essai mené à terme avec succès |
| `TERMINATED` | Essai arrêté prématurément (décision sponsor, DSMB, FDA...) |
| `SUSPENDED` | Essai temporairement suspendu (souvent définitif) |
| `WITHDRAWN` | Retiré avant le premier enregistrement de participant |

---

## c) Description

### Objectif

Ce dataset est conçu pour répondre à la question métier suivante :

> **Peut-on prédire, dès la conception d'un essai clinique, qu'il a de fortes chances d'être abandonné avant sa complétion ?**

Il s'agit d'un problème de **classification supervisée binaire** : à partir des seules métadonnées de design disponibles au moment de l'enregistrement d'un essai (avant tout recrutement), prédire si cet essai sera abandonné (`abandoned = 1`) ou complété (`abandoned = 0`).

### Dimensions

| Caractéristique | Valeur |
|---|---|
| **Nombre de lignes** | ≥ 10 000 (voir rapport de collecte) |
| **Nombre de colonnes** | 23 (dont 3 colonnes méta : nct_id, overall_status, abandoned) |
| **Features utilisables** | 20 (dont 3 features engineerées dérivées) |

---

## d) Schéma détaillé des variables

### Variables méta (non utilisées comme features ML)

| Nom | Type | Description | Valeurs |
|---|---|---|---|
| `nct_id` | String | Identifiant unique de l'essai sur ClinicalTrials.gov | Ex : `NCT01234567` |
| `overall_status` | Catégorielle | Statut final brut de l'essai | `COMPLETED`, `TERMINATED`, `SUSPENDED`, `WITHDRAWN` |

---

### Variable cible

| Nom | Type | Description | Valeurs | Distribution attendue |
|---|---|---|---|---|
| `abandoned` | Binaire (int) | **Variable à prédire** : 1 si l'essai a été interrompu avant complétion | `0` = complété, `1` = abandonné (TERMINATED + SUSPENDED + WITHDRAWN) | ~80–85 % de 0, ~15–20 % de 1 |

---

### Features de design (variables d'entrée)

#### Variables catégorielles

| Nom | Type | Description métier | Valeurs possibles |
|---|---|---|---|
| `phase` | Catégorielle | Phase de développement clinique de l'essai | `PHASE1`, `PHASE2`, `PHASE3`, `PHASE4`, `PHASE1/PHASE2`, `PHASE2/PHASE3`, `NA`, `EARLY_PHASE1` |
| `sponsor_type` | Catégorielle | Classe du sponsor principal (qui finance et gère l'essai) | `INDUSTRY`, `NIH`, `FED`, `OTHER`, `INDIV`, `NETWORK`, `UNKNOWN` |
| `intervention_type` | Catégorielle | Type de l'intervention principale étudiée | `DRUG`, `DEVICE`, `BIOLOGICAL`, `PROCEDURE`, `RADIATION`, `BEHAVIORAL`, `DIETARY_SUPPLEMENT`, `COMBINATION_PRODUCT`, `GENETIC`, `OTHER` |
| `allocation` | Catégorielle | Mode d'allocation des participants aux bras | `RANDOMIZED`, `NON_RANDOMIZED`, `NA` |
| `masking` | Catégorielle | Niveau d'aveugle (masquage) du protocole | `NONE` (open-label), `SINGLE`, `DOUBLE`, `TRIPLE`, `QUADRUPLE` |
| `primary_purpose` | Catégorielle | Objectif principal de l'essai | `TREATMENT`, `PREVENTION`, `DIAGNOSTIC`, `SUPPORTIVE_CARE`, `SCREENING`, `HEALTH_SERVICES_RESEARCH`, `BASIC_SCIENCE`, `DEVICE_FEASIBILITY`, `OTHER`, `UNKNOWN` |
| `condition_category` | Catégorielle (texte) | Première condition ou terme MeSH de la pathologie étudiée | Texte libre (terme MeSH ou condition brute, tronqué à 80 chars) |

#### Variables numériques

| Nom | Type | Description métier | Unité | Plage typique |
|---|---|---|---|---|
| `enrollment_count` | Entier | Nombre de participants prévus (enrollment anticipé au moment de l'enregistrement) | Personnes | 1 – 100 000+ |
| `study_duration_days` | Entier | Durée planifiée de l'essai en jours (start → estimated completion) | Jours | 30 – 5000+ |
| `n_arms` | Entier | Nombre de bras dans le protocole de l'essai | Bras | 0 – 10+ |
| `n_primary_outcomes` | Entier | Nombre de critères de jugement principaux pré-spécifiés | Critères | 0 – 20+ |
| `n_secondary_outcomes` | Entier | Nombre de critères de jugement secondaires pré-spécifiés | Critères | 0 – 100+ |
| `n_locations` | Entier | Nombre de sites de recrutement planifiés | Sites | 0 – 5000+ |
| `n_collaborators` | Entier | Nombre d'institutions collaboratrices (hors sponsor principal) | Institutions | 0 – 50+ |

#### Variables binaires

| Nom | Type | Description métier | Valeurs |
|---|---|---|---|
| `has_dmc` | Binaire (int) | Présence d'un Data Safety Monitoring Board (comité de surveillance indépendant) | `1` = oui, `0` = non |
| `is_multicenter` | Binaire (int) | Essai multicentrique (au moins 2 sites de recrutement) | `1` = oui, `0` = non |
| `has_us_site` | Binaire (int) | Au moins un site de recrutement aux États-Unis | `1` = oui, `0` = non |

---

### Features engineerées (dérivées lors du traitement)

| Nom | Type | Description | Formule |
|---|---|---|---|
| `log_enrollment` | Numérique | Transformation log de l'enrollment pour réduire l'asymétrie de distribution | `log1p(enrollment_count)` |
| `log_duration` | Numérique | Transformation log de la durée planifiée | `log1p(study_duration_days)` |
| `outcomes_ratio` | Numérique | Ratio secondaires/principaux (indicateur de complexité du protocole) | `n_secondary_outcomes / max(n_primary_outcomes, 1)` |

---

## e) Variable cible — Distribution des classes

```
Classe 0 (abandoned = 0) → COMPLETED       : ~82%  (classe majoritaire)
Classe 1 (abandoned = 1) → TERMINATED
                            + SUSPENDED     : ~18%  (classe minoritaire)
                            + WITHDRAWN
```

**Ratio de déséquilibre** : environ **4,5:1** → déséquilibre modéré, dans la plage imposée (5–25 % pour la classe minoritaire).

> 📊 Voir le graphique de distribution dans le notebook `notebooks/01_discovery.ipynb`

---

## f) Notes importantes sur la constitution du dataset

### Principe de design-time features

**Toutes les features ont été sélectionnées selon un critère strict** : elles doivent être connues **avant le début du recrutement**, c'est-à-dire au moment de l'enregistrement de l'essai sur ClinicalTrials.gov.

Variables **exclues** pour cette raison :
- Dates d'enrollment réelles (post-recrutement)
- Nombre de participants effectivement recrutés (vs. anticipé)
- Résultats intermédiaires ou rapports de sécurité
- Raisons d'arrêt (why_stopped) — connues uniquement après l'abandon
- Date effective d'arrêt

### Limitations connues

1. **Enrollment "ACTUAL" vs. "ESTIMATED"** : Pour certains essais terminés, seul le chiffre "ACTUAL" est disponible. Nous l'utilisons comme proxy de l'enrollment planifié, bien qu'il soit légèrement post-hoc. Cette ambiguïté est documentée et acceptable dans le cadre académique.

2. **`condition_category`** : Variable texte libre requérant une normalisation (encodage des termes MeSH de haut niveau) en Phase 2.

3. **Valeurs manquantes** : Certains champs optionnels de l'API (has_dmc, allocation, masking) peuvent être absents pour les essais plus anciens. Une stratégie d'imputation sera définie en Phase 2.

4. **Biais temporel** : Les essais récents (post-2020) sont sur-représentés dans ClinicalTrials.gov. Ce biais doit être documenté dans la Phase 2.
