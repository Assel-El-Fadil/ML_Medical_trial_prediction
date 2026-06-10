import logging
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

logger = logging.getLogger(__name__)

# Chemins vers les modèles (relatifs à la racine du projet)
_MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
_FINAL_MODEL_PATH = _MODELS_DIR / "final_model.joblib"
_FALLBACK_MODEL_PATH = _MODELS_DIR / "preprocessor.joblib"


class ModelPredictor:
    """
    Charge le pipeline entraîné une seule fois au démarrage et expose
    une méthode predict() pour réaliser des inférences.

    Attributs de classe
    -------------------
    THRESHOLD : float
        Seuil de décision orienté recall (0.35) — un essai est prédit
        « abandonné » si P(abandon) >= THRESHOLD.
    """

    THRESHOLD: float = 0.35

    def __init__(self) -> None:
        raw = self._load_model()
        # final_model.joblib peut être sauvegardé comme un dict {"pipeline": ...}
        self._pipeline = raw["pipeline"] if isinstance(raw, dict) else raw

    # ------------------------------------------------------------------
    # Chargement du modèle
    # ------------------------------------------------------------------

    def _load_model(self):
        """Charge final_model.joblib ; fallback sur preprocessor.joblib."""
        if _FINAL_MODEL_PATH.exists():
            logger.info("Chargement du modèle principal : %s", _FINAL_MODEL_PATH)
            model = joblib.load(_FINAL_MODEL_PATH)
            logger.info("Modèle principal chargé avec succès.")
            return model

        if _FALLBACK_MODEL_PATH.exists():
            logger.warning(
                "final_model.joblib introuvable — fallback sur : %s",
                _FALLBACK_MODEL_PATH,
            )
            model = joblib.load(_FALLBACK_MODEL_PATH)
            logger.info("Modèle fallback chargé avec succès.")
            return model

        raise FileNotFoundError(
            f"Aucun modèle trouvé dans {_MODELS_DIR}. "
            "Vérifiez que final_model.joblib ou preprocessor.joblib existe."
        )

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------

    @staticmethod
    def _add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
        """Calcule les features dérivées attendues par le pipeline."""
        df = df.copy()

        # Transformations logarithmiques (log1p pour éviter log(0))
        df["log_enrollment"] = np.log1p(df["enrollment_count"])
        df["log_n_locations"] = np.log1p(df["n_locations"])

        # Complexité du protocole : somme pondérée des outcomes et des bras
        df["protocol_complexity"] = (
            df["n_primary_outcomes"]
            + df["n_secondary_outcomes"]
            + df["n_arms"]
        )

        # Participants par site (évite la division par zéro)
        df["enrollment_per_site"] = df["enrollment_count"] / (
            df["n_locations"].replace(0, 1)
        )

        # Ratio outcomes secondaires / primaires (évite la division par zéro)
        df["outcomes_ratio"] = df["n_secondary_outcomes"] / (
            df["n_primary_outcomes"].replace(0, 1)
        )

        return df

    # ------------------------------------------------------------------
    # Inférence
    # ------------------------------------------------------------------

    def predict(self, input_dict: dict) -> dict:
        """
        Réalise une prédiction à partir d'un dictionnaire de features.

        Paramètres
        ----------
        input_dict : dict
            Dictionnaire issu d'un objet PredictRequest (`.model_dump()`).

        Retourne
        --------
        dict avec les clés :
            - prediction  : str  — "abandonne" ou "complete"
            - probability : float — probabilité associée à la prédiction
            - threshold   : float — seuil de décision utilisé
            - confidence  : str  — "high", "medium" ou "low"
        """
        # 1. Conversion en DataFrame (une seule ligne)
        df = pd.DataFrame([input_dict])

        # 2. Ajout des features dérivées
        df = self._add_derived_features(df)

        # 3. Prédiction via le pipeline (préprocessing + modèle)
        proba_matrix = self._pipeline.predict_proba(df)

        # La colonne 1 correspond à la classe positive (abandon = 1)
        abandon_proba: float = float(proba_matrix[0][1])

        # 4. Décision selon le seuil recall-oriented
        predicted_label = "abandonne" if abandon_proba >= self.THRESHOLD else "complete"

        # 5. Niveau de confiance
        confidence = self._compute_confidence(abandon_proba, predicted_label)

        return {
            "prediction": predicted_label,
            "probability": round(abandon_proba, 4),
            "threshold": self.THRESHOLD,
            "confidence": confidence,
        }

    # ------------------------------------------------------------------
    # Niveau de confiance
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_confidence(proba: float, prediction: str) -> str:
        """
        Calcule le niveau de confiance en fonction de la probabilité brute.

        Règles
        ------
        - high   : proba > 0.7
        - medium : proba > 0.5
        - low    : sinon
        """
        if proba > 0.7:
            return "high"
        if proba > 0.5:
            return "medium"
        return "low"


# NOTE: Ne pas instancier ModelPredictor ici.
# Le singleton est géré par main.py via le lifespan FastAPI.