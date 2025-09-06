"""
API REST para el clasificador de sentimientos usando FastAPI.
Version nueva: Sirve con train_bayes.py
Backups en Others.
lsof -i :5000 -> MlFlow
kill -9 $(lsof -ti :8000) -> Fast Api
pkill -f mlflow -> Todos los de MLFlow
"""
# Agregar el directorio src al path
import joblib
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ConfigDict
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException
import mlflow
from datetime import datetime
from typing import Dict, List

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import predict_sentiment, load_model
# from src.quality_and_lineage import *


# Cargar variables de entorno
load_dotenv()

# Configuración
MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000')
MODEL_NAME = os.getenv('MODEL_NAME', 'movie_sentiment_classifier')
MODEL_STAGE = os.getenv('MODEL_STAGE', 'Production')

# Crear aplicación FastAPI
app = FastAPI(
    title="Movie Sentiment API",
    description="API para clasificación de sentimientos",
    version="1.0.0"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modelos Pydantic


class ReviewRequest(BaseModel):
    text: str = Field(...,
                      description="Texto de la reseña a clasificar", min_length=10)

    class Config:
        json_schema_extra = {
            "example": {
                "text": "This movie was absolutely fantastic! I loved every minute of it."
            }
        }


class SentimentResponse(BaseModel):
    sentiment: str = Field(...,
                           description="Sentimiento predicho (Positivo/Negativo)")
    confidence: float = Field(...,
                              description="Confianza de la predicción (0-1)")
    prediction_time: str = Field(..., description="Timestamp de la predicción")

    class Config:
        json_schema_extra = {
            "example": {
                "sentiment": "Positivo",
                "confidence": 0.95,
                "prediction_time": "2024-01-20T10:30:00"
            }
        }


class BatchReviewRequest(BaseModel):
    reviews: List[str] = Field(...,
                               description="Lista de reseñas a clasificar")

    class Config:
        json_schema_extra = {
            "example": {
                "reviews": [
                    "Great movie!",
                    "Terrible film, waste of time."
                ]
            }
        }


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    status: str
    model_loaded: bool
    model_version: str
    timestamp: str


# Variables globales para el modelo
model = None
vectorizer = None
model_version = "local"


def load_mlflow_model():
    """Intenta cargar el modelo desde MLflow, si falla usa el modelo local."""
    global model, vectorizer, model_version

    # intentar MLflow (opcional)
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

        # Verificar si existe el modelo en el registry
        client = mlflow.tracking.MlflowClient()
        try:
            # Buscar modelos registrados
            model_versions = client.search_model_versions(
                f"name='{MODEL_NAME}'")

            if model_versions:
                # Cargar el modelo en producción
                model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
                model = mlflow.pyfunc.load_model(model_uri)
                vectorizer = None  # MLflow models should be pipelines
                model_version = f"mlflow_{MODEL_STAGE}"
                print(
                    f"✓ Modelo cargado desde MLflow: {MODEL_NAME}/{MODEL_STAGE}")
                return
        except Exception as e:
            print(f"Modelo no encontrado en MLflow Registry: {e}")

    except Exception as e:
        print(f"No se pudo conectar a MLflow: {e}")

    # Segundo intentar cargar modelos locales (más confiable para desarrollo)
    model_files = [
        ('models/logistic_regression_multiclass.pkl', 'Logistic Regression Bayesian'),
        ('models/xgboost_multiclass_calibrated.pkl', 'XGBoost Bayesian'),
        ('models/random_forest_multiclass_calibrated.pkl', 'Random Forest Bayesian'),
    ]

    for model_path, model_name in model_files:
        if os.path.exists(model_path):
            try:
                model = joblib.load(model_path)
                vectorizer = None  # Pipeline incluye TF-IDF
                model_version = f"local_{model_name}"
                print(f"✓ Modelo local cargado: {model_name}")
                return
            except Exception as e:
                print(f"Error cargando {model_path}: {e}")

    # Si todo falla
    raise Exception(
        "No se encontró ningún modelo. Ejecuta train.py o train_bayesian.py")


# Cargar modelo al iniciar
try:
    load_mlflow_model()
except Exception as e:
    print(f"Error cargando modelo: {e}")


@app.on_event("startup")
async def startup_event():
    """Evento de inicio de la aplicación."""
    print("Iniciando Movie Sentiment API...")
    if model is None:
        try:
            load_mlflow_model()
        except Exception as e:
            print(f"Error en startup: {e}")


@app.get("/", response_model=Dict[str, str])
async def root():
    """Endpoint raíz con información de la API."""
    return {
        "message": "Movie Sentiment API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Verificar el estado de la API y el modelo."""
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        model_version=model_version,
        timestamp=datetime.now().isoformat()
    )


@app.post("/predict", response_model=SentimentResponse)
async def predict(request: ReviewRequest):
    """
    Predecir el sentimiento de una reseña.

    Args:
        request: Objeto con el texto de la reseña

    Returns:
        Predicción del sentimiento con confianza
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Modelo no disponible. Por favor, entrena un modelo primero."
        )

    try:
        # Verificar si es un pipeline (modelos bayesianos) o modelo tradicional
        if hasattr(model, 'predict'):
            # Es un pipeline completo
            prediction_proba = model.predict_proba([request.text])[0]
            prediction = model.predict([request.text])[0]
            confidence = prediction_proba.max()
        else:
            # Modelo tradicional con vectorizer separado
            if vectorizer is None:
                raise HTTPException(
                    status_code=503,
                    detail="Vectorizador no disponible"
                )
            prediction, confidence = predict_sentiment(
                request.text, model, vectorizer)

        # Convertir predicción a texto
        sentiment = "Positivo" if prediction == 1 else "Negativo"

        return SentimentResponse(
            sentiment=sentiment,
            confidence=float(confidence),
            prediction_time=datetime.now().isoformat()
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error al procesar la predicción: {str(e)}"
        )


@app.post("/predict/batch", response_model=List[SentimentResponse])
async def predict_batch(request: BatchReviewRequest):
    """
    Predecir el sentimiento de múltiples reseñas.
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Modelo no disponible"
        )

    if len(request.reviews) > 100:
        raise HTTPException(
            status_code=400,
            detail="Máximo 100 reseñas por batch"
        )

    results = []

    for review in request.reviews:
        try:
            # Mismo enfoque que en predict individual
            if hasattr(model, 'predict'):
                # Pipeline completo (modelos bayesianos)
                prediction = model.predict([review])[0]
                prediction_proba = model.predict_proba([review])[0]
                confidence = prediction_proba.max()
            else:
                # Modelo tradicional
                if vectorizer is None:
                    raise HTTPException(
                        status_code=503,
                        detail="Vectorizador no disponible"
                    )
                prediction, confidence = predict_sentiment(
                    review, model, vectorizer)

            sentiment = "Positivo" if prediction == 1 else "Negativo"

            results.append(SentimentResponse(
                sentiment=sentiment,
                confidence=float(confidence),
                prediction_time=datetime.now().isoformat()
            ))
        except Exception as e:
            # En caso de error, agregar resultado con confianza 0
            results.append(SentimentResponse(
                sentiment="Error",
                confidence=0.0,
                prediction_time=datetime.now().isoformat()
            ))

    return results


@app.post("/reload-model")
async def reload_model():
    """Recargar el modelo (útil para actualizar a una nueva versión)."""
    try:
        load_mlflow_model()
        return {
            "status": "success",
            "message": "Modelo recargado exitosamente",
            "model_version": model_version
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error al recargar el modelo: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn

    # Obtener configuración del entorno
    host = os.getenv('API_HOST', '0.0.0.0')
    port = int(os.getenv('API_PORT', '8000'))

    # Ejecutar servidor
    uvicorn.run(
        "api:app",
        host=host,
        port=port,
        reload=True
    )
