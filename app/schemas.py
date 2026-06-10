from pydantic import BaseModel, Field, field_validator
from typing import Literal


class PredictRequest(BaseModel):
    """Modèle de validation des entrées pour la prédiction."""

    phase: Literal[
        "PHASE1", "PHASE2", "PHASE3", "PHASE4",
        "PHASE1/PHASE2", "PHASE2/PHASE3", "NA", "EARLY_PHASE1"
    ]

    sponsor_type: Literal[
        "INDUSTRY", "NIH", "FED", "OTHER", "INDIV", "NETWORK", "UNKNOWN"
    ]

    enrollment_count: int = Field(..., ge=1, description="Nombre de participants >= 1")

    intervention_type: str

    n_arms: int = Field(..., ge=0, description="Nombre de bras >= 0")

    has_dmc: Literal[0, 1] = Field(..., description="Présence d'un DMC (0 ou 1)")

    allocation: Literal["RANDOMIZED", "NON_RANDOMIZED"]

    masking: Literal["NONE", "SINGLE", "DOUBLE", "TRIPLE", "QUADRUPLE"]

    primary_purpose: str

    n_primary_outcomes: int = Field(..., ge=0, description="Nombre d'outcomes primaires >= 0")

    n_secondary_outcomes: int = Field(..., ge=0, description="Nombre d'outcomes secondaires >= 0")

    n_locations: int = Field(..., ge=0, description="Nombre de sites >= 0")

    is_multicenter: Literal[0, 1] = Field(..., description="Essai multicentrique (0 ou 1)")

    has_us_site: Literal[0, 1] = Field(..., description="Présence d'un site US (0 ou 1)")

    n_collaborators: int = Field(..., ge=0, description="Nombre de collaborateurs >= 0")

    model_config = {
        "json_schema_extra": {
            "example": {
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
                "n_collaborators": 2,
            }
        }
    }


class PredictResponse(BaseModel):
    """Modèle de réponse de l'API."""

    prediction: Literal["abandonne", "complete"] = Field(
        ..., description="Résultat prédit : 'abandonne' ou 'complete'"
    )

    probability: float = Field(
        ..., ge=0.0, le=1.0, description="Probabilité associée à la prédiction"
    )

    threshold: float = Field(
        ..., ge=0.0, le=1.0, description="Seuil de décision utilisé"
    )

    confidence: Literal["high", "medium", "low"] = Field(
        ..., description="Niveau de confiance : 'high', 'medium' ou 'low'"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "prediction": "complete",
                "probability": 0.82,
                "threshold": 0.51,
                "confidence": "high",
            }
        }
    }
