import io
import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime

import pandas as pd
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse

from app.predictor import ModelPredictor
from app.schemas import PredictRequest, PredictResponse

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Métriques de performance figées (Phase 3 — test set, seuil déployé = 0.51)
# ---------------------------------------------------------------------------
_MODEL_METRICS = {
    "model_type": "SVC (kernel=linear, C=0.1)",
    "balancing_strategy": "RandomUnderSampler",
    "training_date": "2026-06-09",
    "threshold_deployed": 0.35,         # seuil recall-oriented utilisé en prod
    "threshold_phase3": 0.51,           # seuil optimal issu de la Phase 3
    "performance": {
        "recall":    round(0.6972111553784861, 4),
        "precision": round(0.3747323340471092, 4),
        "f1":        round(0.4874651810584958, 4),
        "accuracy":  round(0.7994550408719346, 4),
        "roc_auc":   round(0.8294058614028734, 4),
        "pr_auc":    round(0.6075462211531307, 4),
    },
    "confusion_matrix": {
        "true_negatives":  1292,
        "false_positives":  292,
        "false_negatives":   76,
        "true_positives":   175,
    },
}

# ---------------------------------------------------------------------------
# Singleton du prédicteur (chargé au démarrage)
# ---------------------------------------------------------------------------
_predictor: ModelPredictor | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Charge le modèle UNE SEULE FOIS au démarrage de l'application."""
    global _predictor
    logger.info("=== Démarrage de l'API — chargement du modèle ===")
    try:
        _predictor = ModelPredictor()
        logger.info("Modèle chargé avec succès. API prête.")
    except FileNotFoundError as exc:
        logger.critical("Impossible de charger le modèle : %s", exc)
        # L'API démarre quand même ; /health signalera model_loaded=false
    yield
    logger.info("=== Arrêt de l'API ===")


# ---------------------------------------------------------------------------
# Application FastAPI
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Clinical Trial Completion Predictor",
    description=(
        "API de prédiction de l'issue (abandon / complétion) d'un essai clinique. "
        "Basée sur un pipeline SVC entraîné sur la base ClinicalTrials.gov. "
        "Seuil de décision orienté recall (0.35) pour maximiser la détection des abandons."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)


# ---------------------------------------------------------------------------
# Middleware de logging des requêtes
# ---------------------------------------------------------------------------
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration_ms = round((time.time() - start) * 1000, 1)
    logger.info(
        "REQUEST  %s %s → %s  [%.1f ms]",
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
    )
    return response


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get(
    "/",
    summary="Page d'accueil de l'API",
    description="Retourne les informations générales sur l'API et un lien vers la documentation.",
    tags=["Général"],
)
async def root():
    return {
        "name": "Clinical Trial Completion Predictor",
        "description": (
            "Prédit si un essai clinique sera abandonné ou complété "
            "à partir de ses caractéristiques de conception."
        ),
        "version": "1.0.0",
        "documentation": "/docs",
        "health_check": "/health",
    }


@app.get(
    "/health",
    summary="Vérification de l'état de l'API",
    description="Indique si l'API est en ligne et si le modèle est correctement chargé.",
    tags=["Général"],
)
async def health():
    return {
        "status": "ok",
        "model_loaded": _predictor is not None,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }


@app.get(
    "/model/info",
    summary="Informations sur le modèle déployé",
    description=(
        "Retourne le type de modèle, la date d'entraînement, "
        "les métriques de performance sur le jeu de test (Phase 3) "
        "et le seuil de décision utilisé en production."
    ),
    tags=["Modèle"],
)
async def model_info():
    return _MODEL_METRICS


@app.post(
    "/predict",
    response_model=PredictResponse,
    summary="Prédiction sur un seul essai clinique",
    description=(
        "Reçoit les caractéristiques d'un essai clinique et retourne : "
        "la classe prédite ('abandonne' / 'complete'), "
        "la probabilité d'abandon, le seuil utilisé et le niveau de confiance."
    ),
    tags=["Prédiction"],
)
async def predict(request: PredictRequest) -> PredictResponse:
    if _predictor is None:
        raise HTTPException(
            status_code=500,
            detail="Le modèle n'est pas chargé. Vérifiez les logs du serveur.",
        )
    try:
        result = _predictor.predict(request.model_dump())
        return PredictResponse(**result)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Input invalide : {exc}") from exc
    except Exception as exc:
        logger.exception("Erreur inattendue lors de la prédiction")
        raise HTTPException(status_code=500, detail=f"Erreur serveur : {exc}") from exc


@app.post(
    "/predict/batch",
    summary="Prédiction en lot depuis un fichier CSV",
    description=(
        "Uploade un fichier CSV contenant une ligne par essai clinique. "
        "Les colonnes doivent correspondre aux champs de PredictRequest. "
        "Retourne le même CSV enrichi de deux colonnes : `prediction` et `probability`."
    ),
    tags=["Prédiction"],
    responses={
        200: {"content": {"text/csv": {}}, "description": "CSV enrichi en retour"},
        400: {"description": "Fichier CSV invalide ou colonnes manquantes"},
        500: {"description": "Erreur serveur lors de la prédiction"},
    },
)
async def predict_batch(file: UploadFile = File(..., description="Fichier CSV à traiter")):
    if _predictor is None:
        raise HTTPException(
            status_code=500,
            detail="Le modèle n'est pas chargé. Vérifiez les logs du serveur.",
        )

    # --- Lecture du CSV ---
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Impossible de lire le CSV : {exc}") from exc

    if df.empty:
        raise HTTPException(status_code=400, detail="Le fichier CSV est vide.")

    # --- Prédictions ligne par ligne ---
    predictions = []
    errors = []
    for idx, row in df.iterrows():
        try:
            result = _predictor.predict(row.to_dict())
            predictions.append(
                {"prediction": result["prediction"], "probability": result["probability"]}
            )
        except Exception as exc:
            logger.warning("Ligne %d ignorée — erreur : %s", idx, exc)
            predictions.append({"prediction": "error", "probability": None})
            errors.append({"row": int(idx), "error": str(exc)})

    # --- Enrichissement du DataFrame ---
    results_df = pd.DataFrame(predictions)
    df["prediction"] = results_df["prediction"].values
    df["probability"] = results_df["probability"].values

    if errors:
        logger.warning("%d lignes en erreur sur %d", len(errors), len(df))

    # --- Réponse CSV en streaming ---
    output = io.StringIO()
    df.to_csv(output, index=False)
    output.seek(0)

    filename = file.filename.replace(".csv", "_predictions.csv") if file.filename else "predictions.csv"
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )
