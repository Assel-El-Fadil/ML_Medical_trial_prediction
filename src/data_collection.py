"""
data_collection.py
==================
Script de collecte de données depuis l'API ClinicalTrials.gov v2.

Objectif : Construire un dataset pour la prédiction de l'abandon des essais cliniques
           (classification binaire : abandoned = 1 si TERMINATED/SUSPENDED/WITHDRAWN,
                                     abandoned = 0 si COMPLETED)

API : https://clinicaltrials.gov/api/v2/studies
Authentification : aucune (API publique et gratuite)
Documentation : https://clinicaltrials.gov/data-api/api

Usage :
    python data_collection.py                  # collecte complète (≥ 10 000 lignes)
    python data_collection.py --sample 100     # mode test rapide (100 lignes)
    python data_collection.py --max 5000       # limiter à N lignes
"""

import argparse
import json
import logging
import math
import os
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_BASE_URL = "https://clinicaltrials.gov/api/v2/studies"

# Statuts filtrés : on ne récupère QUE ces 4 statuts finaux
TARGET_STATUSES = ["COMPLETED", "TERMINATED", "SUSPENDED", "WITHDRAWN"]

# Statuts qui définissent la classe "abandoned = 1"
ABANDONED_STATUSES = {"TERMINATED", "SUSPENDED", "WITHDRAWN"}

# Paramètres de pagination
PAGE_SIZE = 1000          # maximum autorisé par l'API v2
REQUEST_DELAY = 0.5       # secondes entre requêtes (respecter le rate limiting)
MAX_RETRIES = 5           # tentatives en cas d'erreur réseau
RETRY_BACKOFF = 2.0       # facteur d'attente exponentielle

# Chemins de sortie
RAW_DIR = Path("../data/raw")
PROCESSED_DIR = Path("../data")
LOG_DIR = Path("../logs")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging() -> logging.Logger:
    """Configure le système de logging (console + fichier)."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / f"collection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialisé → {log_file}")
    return logger


logger = setup_logging()

# ---------------------------------------------------------------------------
# Fonctions utilitaires
# ---------------------------------------------------------------------------

def safe_get(d: dict, *keys, default=None):
    """
    Accès sécurisé à un chemin de clés imbriquées dans un dictionnaire.

    Exemple : safe_get(study, 'designModule', 'phases', default=[])
    """
    for key in keys:
        if isinstance(d, dict):
            d = d.get(key, default)
        elif isinstance(d, list) and isinstance(key, int):
            d = d[key] if key < len(d) else default
        else:
            return default
    return d if d is not None else default


def fetch_page(params: dict, session: requests.Session) -> dict:
    """
    Récupère une page de résultats depuis l'API avec retry exponentiel.

    Args:
        params : paramètres de requête GET
        session : session HTTP réutilisable

    Returns:
        dict : réponse JSON de l'API

    Raises:
        RuntimeError : si toutes les tentatives échouent
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = session.get(API_BASE_URL, params=params, timeout=30)
            logger.info(
                f"GET {API_BASE_URL} | params={params.get('pageToken', 'page-1')} "
                f"| status={response.status_code}"
            )

            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:
                wait = RETRY_BACKOFF ** attempt * 10
                logger.warning(f"Rate limit (429). Attente {wait:.0f}s avant retry {attempt}/{MAX_RETRIES}")
                time.sleep(wait)
            elif response.status_code >= 500:
                wait = RETRY_BACKOFF ** attempt
                logger.warning(f"Erreur serveur ({response.status_code}). Retry {attempt}/{MAX_RETRIES} dans {wait:.0f}s")
                time.sleep(wait)
            else:
                logger.error(f"Erreur HTTP inattendue : {response.status_code} — {response.text[:300]}")
                raise RuntimeError(f"HTTP {response.status_code}")

        except requests.exceptions.ConnectionError as e:
            wait = RETRY_BACKOFF ** attempt
            logger.warning(f"Erreur réseau (tentative {attempt}/{MAX_RETRIES}) : {e}. Retry dans {wait:.0f}s")
            time.sleep(wait)
        except requests.exceptions.Timeout:
            wait = RETRY_BACKOFF ** attempt
            logger.warning(f"Timeout (tentative {attempt}/{MAX_RETRIES}). Retry dans {wait:.0f}s")
            time.sleep(wait)

    raise RuntimeError(f"Échec après {MAX_RETRIES} tentatives pour les params : {params}")


# ---------------------------------------------------------------------------
# Extraction des features depuis un study JSON
# ---------------------------------------------------------------------------

def extract_features(study: dict) -> dict | None:
    """
    Extrait les features de design-time depuis un objet study de l'API v2.

    Toutes les features extraites sont disponibles AU MOMENT DE LA CONCEPTION
    de l'essai (avant le début du recrutement). Aucune variable post-hoc.

    Args:
        study : dictionnaire JSON d'un essai clinique (format API v2)

    Returns:
        dict : dictionnaire de features, ou None si l'essai est invalide
    """
    protocol = study.get("protocolSection", {})

    # Modules principaux
    id_module          = protocol.get("identificationModule", {})
    status_module      = protocol.get("statusModule", {})
    sponsor_module     = protocol.get("sponsorCollaboratorsModule", {})
    oversight_module   = protocol.get("oversightModule", {})
    description_module = protocol.get("descriptionModule", {})
    conditions_module  = protocol.get("conditionsModule", {})
    design_module      = protocol.get("designModule", {})
    outcomes_module    = protocol.get("outcomesModule", {})
    eligibility_module = protocol.get("eligibilityModule", {})
    contacts_module    = protocol.get("contactsLocationsModule", {})
    interventions      = protocol.get("armsInterventionsModule", {})


    # ---- Identifiant ----
    nct_id = id_module.get("nctId", "")

    # ---- Variable cible ----
    overall_status = status_module.get("overallStatus", "")
    if overall_status not in TARGET_STATUSES:
        return None  # on ignore les statuts hors scope

    abandoned = 1 if overall_status in ABANDONED_STATUSES else 0

    # ---- Phase ----
    phases_list = design_module.get("phases", [])
    if phases_list:
        # Exemples : ["PHASE1"], ["PHASE1", "PHASE2"], ["NA"]
        phase = phases_list[0] if len(phases_list) == 1 else "/".join(phases_list)
    else:
        phase = "NA"

    # ---- Type de sponsor ----
    lead_sponsor = sponsor_module.get("leadSponsor", {})
    sponsor_type = lead_sponsor.get("class", "UNKNOWN")

    # ---- Enrollment (anticipé) ----
    enrollment_info = design_module.get("enrollmentInfo", {})
    enrollment_count = None
    if enrollment_info.get("type") == "ESTIMATED":
        enrollment_count = enrollment_info.get("count")
    elif enrollment_info.get("type") == "ACTUAL":
        enrollment_count = enrollment_info.get("count")

    # ---- Type d'intervention principale ----
    interventions_list = interventions.get("interventions", [])
    intervention_type = interventions_list[0].get("type", "UNKNOWN") if interventions_list else "UNKNOWN"

    # ---- Nombre de bras ----
    arm_groups = interventions.get("armGroups", [])
    n_arms = len(arm_groups)

    # ---- Comité de surveillance (DMC) ----
    has_dmc = oversight_module.get("oversightHasDmc", None)
    if isinstance(has_dmc, bool):
        has_dmc = int(has_dmc)
    elif isinstance(has_dmc, str):
        has_dmc = 1 if has_dmc.upper() == "YES" else 0
    else:
        has_dmc = None

    # ---- Allocation ----
    design_info = design_module.get("designInfo", {})
    allocation = design_info.get("allocation", "")
    if not allocation or allocation == "NA":
        return None

    # ---- Masquage ----
    masking_info = design_info.get("maskingInfo", {})
    masking = masking_info.get("masking", "NONE")

    # ---- Objectif principal ----
    primary_purpose = design_info.get("primaryPurpose", "UNKNOWN")

    # ---- Nombre de critères de jugement ----
    n_primary_outcomes = len(outcomes_module.get("primaryOutcomes", []))
    n_secondary_outcomes = len(outcomes_module.get("secondaryOutcomes", []))

    # ---- Nombre de sites ----
    locations = contacts_module.get("locations", [])
    n_locations = len(locations)
    is_multicenter = int(n_locations > 1)

    # ---- Site aux États-Unis ----
    countries = [loc.get("country", "") for loc in locations]
    has_us_site = int("United States" in countries)

    # ---- Nombre de collaborateurs ----
    collaborators = sponsor_module.get("collaborators", [])
    n_collaborators = len(collaborators)

    return {
        "nct_id": nct_id,
        "overall_status": overall_status,
        "abandoned": abandoned,
        "phase": phase,
        "sponsor_type": sponsor_type,
        "enrollment_count": enrollment_count,
        "intervention_type": intervention_type,
        "n_arms": n_arms,
        "has_dmc": has_dmc,
        "allocation": allocation,
        "masking": masking,
        "primary_purpose": primary_purpose,
        "n_primary_outcomes": n_primary_outcomes,
        "n_secondary_outcomes": n_secondary_outcomes,
        "n_locations": n_locations,
        "is_multicenter": is_multicenter,
        "has_us_site": has_us_site,
        "n_collaborators": n_collaborators,
    }


# ---------------------------------------------------------------------------
# Collecte principale
# ---------------------------------------------------------------------------

def collect_data(max_records: int = 15000) -> list[dict]:
    """
    Collecte les données depuis l'API ClinicalTrials.gov v2 avec pagination.

    Args:
        max_records : nombre maximum de lignes à collecter (0 = illimité)

    Returns:
        list[dict] : liste de feature dicts extraits
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    records = []
    page_token = None
    page_num = 0
    total_fetched = 0
    total_valid = 0

    # Paramètres de base de la requête
    base_params = {
        "format": "json",
        "pageSize": PAGE_SIZE,
        "filter.overallStatus": "|".join(TARGET_STATUSES)
    }

    session = requests.Session()
    session.headers.update({
        "Accept": "application/json",
        "User-Agent": "ClinicalTrialsMLProject/1.0"
    })

    logger.info(f"Début de la collecte | max_records={max_records} | statuts={TARGET_STATUSES}")

    while True:
        # Construction des paramètres de la page courante
        params = dict(base_params)
        if page_token:
            params["pageToken"] = page_token

        # Appel API
        try:
            data = fetch_page(params, session)
        except RuntimeError as e:
            logger.error(f"Arrêt de la collecte sur erreur : {e}")
            break

        studies = data.get("studies", [])
        page_num += 1
        total_fetched += len(studies)

        if not studies:
            logger.info("Plus de résultats disponibles. Collecte terminée.")
            break

        # Sauvegarde brute de la page (JSON) pour éviter de re-requêter
        raw_path = RAW_DIR / f"page_{page_num:04d}.json"
        with open(raw_path, "w", encoding="utf-8") as f:
            json.dump(studies, f, ensure_ascii=False, indent=2)

        # Extraction des features
        page_records = []
        for study in studies:
            features = extract_features(study)
            if features is not None:
                page_records.append(features)
                total_valid += 1

        records.extend(page_records)

        logger.info(
            f"Page {page_num:4d} | fetched={len(studies)} | valid={len(page_records)} "
            f"| cumul_valid={total_valid}"
        )

        # Condition d'arrêt : quota atteint
        if max_records > 0 and total_valid >= max_records:
            logger.info(f"Quota de {max_records} lignes atteint. Arrêt de la collecte.")
            break

        # Pagination : nextPageToken indique qu'il y a une page suivante
        next_token = data.get("nextPageToken")
        if not next_token:
            logger.info("Dernière page atteinte. Collecte terminée.")
            break

        page_token = next_token
        time.sleep(REQUEST_DELAY)

    logger.info(
        f"Collecte terminée | pages={page_num} | total_fetched={total_fetched} "
        f"| total_valid={total_valid}"
    )
    return records


# ---------------------------------------------------------------------------
# Transformation en DataFrame et post-traitement
# ---------------------------------------------------------------------------

def build_dataframe(records: list[dict]) -> pd.DataFrame:
    """
    Transforme la liste de feature dicts en DataFrame pandas avec post-traitements.

    Args:
        records : liste de dicts issus de extract_features()

    Returns:
        pd.DataFrame : dataset prêt pour la Phase 2
    """
    df = pd.DataFrame(records)

    if df.empty:
        logger.warning("DataFrame vide ! Vérifiez la collecte.")
        return df

    logger.info(f"Shape brut : {df.shape}")

    # ---- Feature engineering supplémentaire ----

    # Ratio outcomes : n_secondary / max(n_primary, 1)
    df["outcomes_ratio"] = df["n_secondary_outcomes"] / df["n_primary_outcomes"].clip(lower=1)

    # ---- Nettoyage ----

    # Supprimer les doublons sur nct_id
    before = len(df)
    df = df.drop_duplicates(subset=["nct_id"])
    if len(df) < before:
        logger.info(f"Doublons supprimés : {before - len(df)}")

    # Réordonner les colonnes
    col_order = [
        "phase", "sponsor_type", "enrollment_count", "intervention_type", "n_arms",
        "has_dmc", "allocation", "masking", "primary_purpose",
        "n_primary_outcomes", "n_secondary_outcomes", "outcomes_ratio",
        "n_locations", "is_multicenter", "has_us_site",
        "n_collaborators", "abandoned"
    ]
    existing_cols = [c for c in col_order if c in df.columns]
    df = df[existing_cols]

    # ---- Rapport de qualité ----
    logger.info(f"Shape final : {df.shape}")
    logger.info(f"\nDistribution de la variable cible :\n{df['abandoned'].value_counts(normalize=True).round(3)}")
    logger.info(f"\nValeurs manquantes (%) :\n{(df.isnull().mean() * 100).round(1).to_string()}")

    return df


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Collecte de données ClinicalTrials.gov pour prédiction d'abandon d'essais cliniques"
    )
    parser.add_argument(
        "--sample", type=int, default=0,
        help="Mode test : collecter seulement N lignes (0 = collecte complète)"
    )
    parser.add_argument(
        "--max", type=int, default=12000,
        help="Nombre maximum de lignes a collecter (defaut : 12000)"
    )
    args = parser.parse_args()

    max_records = args.sample if args.sample > 0 else args.max

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Collecte
    records = collect_data(max_records=max_records)

    if not records:
        logger.error("Aucune donnée collectée. Vérifiez la connexion et l'API.")
        return

    # 2. Construction du DataFrame
    df = build_dataframe(records)

    # 3. Sauvegarde du dataset complet
    output_path = PROCESSED_DIR / "dataset.csv"
    df.to_csv(output_path, index=False, encoding="utf-8")
    logger.info(f"Dataset complet sauvegardé → {output_path} ({len(df):,} lignes)")

    # 4. Sauvegarde de l'extrait (100 lignes pour vérification rapide)
    sample_path = PROCESSED_DIR / "sample.csv"
    # On prend un échantillon stratifié pour représenter les deux classes
    try:
        sample_df = df.groupby("abandoned", group_keys=False).apply(
            lambda g: g.sample(min(len(g), 50), random_state=42)
        ).reset_index(drop=True)
        sample_df = sample_df.head(100)
    except Exception:
        sample_df = df.head(100)
    sample_df.to_csv(sample_path, index=False, encoding="utf-8")
    logger.info(f"Extrait (100 lignes) sauvegardé → {sample_path}")

    # 5. Sauvegarde Parquet (optionnel, plus efficace pour la Phase 2)
    try:
        parquet_path = PROCESSED_DIR / "dataset.parquet"
        df.to_parquet(parquet_path, index=False)
        logger.info(f"Dataset Parquet sauvegardé → {parquet_path}")
    except ImportError:
        logger.warning("pyarrow non installé. Parquet non généré. pip install pyarrow")

    logger.info("=== Collecte et traitement terminés avec succès ===")
    print(f"\n Dataset prêt : {len(df):,} essais | {df['abandoned'].mean():.1%} abandonnés")
    print(f"   -> {output_path}")
    print(f"   -> {sample_path}")


if __name__ == "__main__":
    main()
