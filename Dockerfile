# ============================================================
#  Clinical Trial Completion Predictor — Dockerfile
#  Image de base : python:3.11-slim
#  Port exposé   : 8000
# ============================================================

FROM python:3.11-slim

# --- Métadonnées ---
LABEL maintainer="ML Medical Trial Prediction Team"
LABEL description="API FastAPI de prédiction d'issue d'essai clinique"
LABEL version="1.0.0"

# --- Variables d'environnement ---
ENV MODEL_PATH=models/final_model.joblib \
    LOG_LEVEL=info \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# --- Répertoire de travail ---
WORKDIR /app

# --- Dépendances (couche cachée séparément du code) ---
# Copier requirements.txt EN PREMIER pour bénéficier du cache Docker :
# si le code change mais pas les dépendances, cette couche n'est pas reconstruite.
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# --- Code applicatif ---
COPY app/ ./app/

# --- Modèles entraînés ---
COPY models/ ./models/

# --- Port exposé ---
EXPOSE 8000

# --- Healthcheck ---
# Vérifie GET /health toutes les 30 s.
# Délai initial de 10 s pour laisser le temps à uvicorn de démarrer.
# Timeout de 5 s par tentative ; 3 échecs consécutifs → unhealthy.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" \
    || exit 1

# --- Démarrage ---
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
