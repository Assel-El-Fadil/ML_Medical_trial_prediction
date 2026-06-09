# Fiche de Cadrage — Phase 1
## Projet Machine Learning | 2ème année GI | 2025-2026

---

## 1. Identification du projet

| Champ | Valeur |
|---|---|
| **Titre** | Prédiction de l'abandon des essais cliniques |
| **Domaine métier** | Santé / Recherche clinique |
| **API source** | ClinicalTrials.gov API v2 |
| **URL API** | `https://clinicaltrials.gov/api/v2/studies` |
| **Clé API requise** | Non (API publique, sans authentification) |
| **Date de cadrage** | Mai 2025 |

---

## 2. Problématique métier

### 2.1. Contexte

Les essais cliniques constituent l'étape finale et la plus coûteuse du développement de nouveaux médicaments et thérapies. Un essai de phase III peut coûter entre **100 et 300 millions de dollars** et mobiliser des milliers de patients pendant plusieurs années. Or, une proportion significative de ces essais n'arrive jamais à terme : ils sont interrompus, suspendus ou retirés avant même d'avoir produit des résultats exploitables.

Ces abandons représentent un gaspillage colossal : financier pour les sponsors (industrie pharmaceutique, institutions académiques), humain pour les patients qui s'y sont engagés, et sociétal pour les systèmes de santé qui attendent ces traitements.

### 2.2. Question métier centrale

> **Est-il possible de prédire, dès la conception d'un essai clinique et en se basant uniquement sur ses métadonnées de design, qu'il présente un risque élevé d'être abandonné avant sa complétion ?**

Il ne s'agit pas d'une analyse rétrospective après constatation de l'échec, mais bien d'une **détection précoce** fondée exclusivement sur des informations disponibles au moment de l'enregistrement de l'essai : phase, type de sponsor, nombre de participants prévus, durée planifiée, type d'intervention, etc.

### 2.3. Impact attendu

Un modèle fiable permettrait de :
- **Réorienter les financements** vers les essais les plus susceptibles d'aboutir.
- **Alerter les comités de révision** sur les conceptions à risque avant le lancement.
- **Réduire l'exposition inutile des patients** à des protocoles voués à l'échec.
- **Optimiser les portefeuilles R&D** des acteurs pharmaceutiques et académiques.

---

## 3. Variable cible

La variable cible est **binaire** :

| Valeur | Signification | Statut ClinicalTrials.gov |
|---|---|---|
| `0` | Essai complété | `COMPLETED` |
| `1` | Essai abandonné | `TERMINATED`, `SUSPENDED`, `WITHDRAWN` |

**Justification du regroupement** : `TERMINATED` (arrêt décidé par le sponsor/DSMB), `SUSPENDED` (arrêt temporaire, souvent définitif en pratique) et `WITHDRAWN` (retiré avant même le premier enregistrement de participant) représentent toutes des formes d'**abandon avant complétion**, opposées à `COMPLETED`.

**Déséquilibre attendu** : environ 15–20 % de la classe `abandoned = 1`, ce qui satisfait la contrainte de 5–25 % imposée par le projet.

---

## 4. Features retenues (design-time only)

Un principe fondamental guide la sélection des features : **toutes les variables doivent être connues au moment de la conception de l'essai**, avant le début du recrutement. Aucune variable post-hoc n'est admise.

| Feature | Type | Description | Source API |
|---|---|---|---|
| `phase` | Catégorielle | Phase de l'essai (PHASE1, PHASE2, PHASE3, PHASE4, NA) | `phases` |
| `sponsor_type` | Catégorielle | Type de sponsor principal (INDUSTRY, NIH, FED, OTHER, INDIV) | `leadSponsor.class` |
| `enrollment_count` | Numérique | Nombre de participants prévus (enrollment anticipé) | `enrollmentInfo.count` (type=ANTICIPATED) |
| `study_duration_days` | Numérique | Durée planifiée en jours (startDate → completionDate estimée) | `startDateStruct`, `completionDateStruct` |
| `intervention_type` | Catégorielle | Type d'intervention principale (DRUG, DEVICE, BIOLOGICAL, PROCEDURE, etc.) | `interventions[0].type` |
| `n_arms` | Numérique | Nombre de bras dans le protocole | `armGroups` (count) |
| `has_dmc` | Binaire | Présence d'un comité de surveillance des données (DSMB) | `oversightModule.isFdaRegulatedDrug` / `oversightModule.oversightHasDmc` |
| `allocation` | Catégorielle | Mode d'allocation (RANDOMIZED, NON_RANDOMIZED, NA) | `designModule.designInfo.allocation` |
| `masking` | Catégorielle | Type de masquage (NONE, SINGLE, DOUBLE, TRIPLE, QUADRUPLE) | `designModule.designInfo.maskingInfo.masking` |
| `primary_purpose` | Catégorielle | Objectif principal (TREATMENT, PREVENTION, DIAGNOSTIC, etc.) | `designModule.designInfo.primaryPurpose` |
| `n_primary_outcomes` | Numérique | Nombre de critères de jugement principaux définis | `outcomesModule.primaryOutcomes` (count) |
| `n_secondary_outcomes` | Numérique | Nombre de critères de jugement secondaires | `outcomesModule.secondaryOutcomes` (count) |
| `condition_category` | Catégorielle | Catégorie MeSH de la condition étudiée (dérivée) | `conditionsModule.meshTerms` |
| `n_locations` | Numérique | Nombre de sites de recrutement prévus | `contactsLocationsModule.locations` (count) |
| `is_multicenter` | Binaire | Essai multicentrique (n_locations > 1) | dérivée de `n_locations` |
| `has_us_site` | Binaire | Au moins un site aux États-Unis | `contactsLocationsModule.locations[].country` |
| `n_collaborators` | Numérique | Nombre de collaborateurs institutionnels | `sponsorCollaboratorsModule.collaborators` (count) |

> **Total : 17 features** → ≥ 8 après feature engineering, contrainte satisfaite.

---

## 5. Objectifs métiers quantifiés

| # | Objectif métier |
|---|---|
| OM-1 | Identifier au moins **70 % des essais qui seront abandonnés** parmi tous les essais enregistrés, afin de permettre une intervention préventive avant le début du recrutement, tout en préservant une précision acceptable. |
| OM-2 | Limiter les **fausses alertes** (essais signalés à tort comme "à risque") à un taux raisonnable pour ne pas décourager les projets viables : maintenir une précision ≥ 40 % sur la classe abandonnée. |
| OM-3 | Fournir un modèle **interprétable** (feature importance lisible) pour permettre aux comités de révision de comprendre et justifier les décisions d'alerte. |

---

## 6. Tableau de traduction métier → ML

| Objectif métier | Objectif ML | Métrique principale | Seuil cible |
|---|---|---|---|
| OM-1 : Détecter 70 % des essais abandonnés | Maintenir un recall minimal sur la classe `abandoned = 1` tout en maximisant la précision | **Recall (classe 1)** | ≥ 0,70 |
| OM-2 : Limiter les fausses alertes | Maintenir une précision acceptable sur la classe `abandoned = 1` | **Precision (classe 1)** | ≥ 0,40 |
| OM-1 + OM-2 combinés | Équilibrer recall et precision sur la classe minoritaire | **F1-score (classe 1)** | ≥ 0,55 |
| Vue globale du modèle sur données déséquilibrées | Évaluer la qualité du classifieur indépendamment du seuil | **PR-AUC** | À maximiser |

**Métriques exclues comme métrique principale** :
- ~~Accuracy~~ : trompeuse sur données déséquilibrées (un modèle naïf "tout = 0" atteindrait déjà ~82 % d'accuracy).
- ~~ROC-AUC seule~~ : optimiste sur données déséquilibrées, ne reflète pas les performances réelles sur la classe minoritaire.

---

## 7. Analyse du coût métier asymétrique

### Question fondamentale : que coûte plus cher — un faux positif ou un faux négatif ?

Dans notre contexte :

**Faux négatif (FN)** — Un essai prédit comme "va se compléter" alors qu'il sera en réalité abandonné :
- Coût : lancement d'un essai voué à l'échec → **dépenses de 10 à 300 M$** selon la phase, mobilisation de patients pendant des mois, retard de traitements alternatifs.
- Estimation conservatrice : **50–100 M$ par essai de Phase III raté**.

**Faux positif (FP)** — Un essai prédit comme "à risque d'abandon" alors qu'il se complèterait :
- Coût : revue approfondie du protocole par un comité (~1–2 semaines de travail), possible retard de démarrage de 1–3 mois, friction avec le sponsor.
- Estimation : **50 000–200 000 $** par essai faussement signalé.

### Ratio d'asymétrie estimé

```
Coût(FN) / Coût(FP) ≈ 500 à 2000×
```

**Conclusion** : L'asymétrie est fortement en faveur d'un **modèle recall-oriented**. Il vaut bien mieux sur-signaler quelques essais sains (FP peu coûteux) que de laisser passer des essais à risque (FN catastrophiques). Cette analyse justifie :
1. Le choix du **recall comme métrique principale**.
2. L'utilisation d'un **seuil de décision abaissé** (< 0.5) lors du déploiement.

---

## 8. Choix des métriques — Justification

| Métrique | Rôle | Justification |
|---|---|---|
| **Recall (classe 1)** | Métrique principale | Reflète directement OM-1 ; minimise les FN coûteux |
| **Precision (classe 1)** | Contrainte opérationnelle | Évite une explosion des fausses alertes (OM-2) |
| **F1-score (classe 1)** | Métrique d'équilibre | Synthèse recall/precision pour la sélection de modèle |
| **PR-AUC** | Évaluation globale | Robuste au déséquilibre, compare les modèles indépendamment du seuil |

---

## 9. Contraintes techniques respectées

| Critère | Exigence | Notre projet |
|---|---|---|
| Type de tâche | Classification supervisée | ✅ Classification binaire (abandoned 0/1) |
| Taille totale | ≥ 10 000 lignes | ✅ ClinicalTrials.gov contient > 400 000 essais éligibles |
| Nombre de features | ≥ 8 après feature engineering | ✅ 17 features identifiées |
| Classe minoritaire | 5 % – 25 % | ✅ ~15–20 % attendu (TERMINATED + SUSPENDED + WITHDRAWN) |
| Types de variables | Numérique + catégorielle | ✅ Mix des deux types |
| Source | API publique et gratuite | ✅ ClinicalTrials.gov API v2, sans clé, sans quota strict |
