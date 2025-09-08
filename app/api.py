"""
API REST para el clasificador de sentimientos de productos usando FastAPI.
Sistema de clasificación multiclase: Positivo, Neutro, Negativo
Integrado con MLflow para tracking de modelos y experimentos.

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
from typing import Dict, List, Optional

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import predict_sentiment, load_model

# Cargar variables de entorno
load_dotenv()

# Configuración
MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000')
MODEL_NAME = os.getenv('MODEL_NAME', 'product_sentiment_classifier')
MODEL_STAGE = os.getenv('MODEL_STAGE', 'Production')

# Crear aplicación FastAPI
app = FastAPI(
    title="Product Sentiment Analysis API",
    description="API para clasificación de sentimientos en reseñas de productos (Positivo/Neutro/Negativo)",
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

# ==================== Modelos Pydantic ====================

class ProductReviewRequest(BaseModel):
    """Modelo para solicitud de análisis de reseña individual"""
    text: str = Field(
        ...,
        description="Texto de la reseña del producto a clasificar",
        min_length=10,
        max_length=5000
    )
    product_name: Optional[str] = Field(
        None,
        description="Nombre del producto (opcional, para mejor procesamiento)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "text": "Este smartphone superó mis expectativas. La cámara es excelente y la batería dura todo el día.",
                "product_name": "Smartphone"
            }
        }


class SentimentResponse(BaseModel):
    """Modelo para respuesta de predicción de sentimiento"""
    sentiment: str = Field(
        ...,
        description="Sentimiento predicho (Positivo/Neutro/Negativo)"
    )
    confidence: float = Field(
        ...,
        description="Confianza de la predicción (0-1)"
    )
    product: Optional[str] = Field(
        None,
        description="Producto analizado"
    )
    prediction_time: str = Field(
        ...,
        description="Timestamp de la predicción"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "sentiment": "Positivo",
                "confidence": 0.92,
                "product": "Smartphone",
                "prediction_time": "2024-01-20T10:30:00"
            }
        }


class BatchProductReviewRequest(BaseModel):
    """Modelo para solicitud de análisis por lotes"""
    reviews: List[str] = Field(
        ...,
        description="Lista de reseñas de productos a clasificar",
        min_items=1,
        max_items=100
    )
    product_names: Optional[List[str]] = Field(
        None,
        description="Lista opcional de nombres de productos correspondientes"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "reviews": [
                    "El teclado funciona perfectamente, muy cómodo para escribir.",
                    "La tablet llegó con defectos, muy decepcionado.",
                    "El monitor está bien, cumple su función pero nada especial."
                ],
                "product_names": ["Teclado", "Tablet", "Monitor"]
            }
        }


class ModelHealthResponse(BaseModel):
    """Modelo para respuesta de estado del sistema"""
    model_config = ConfigDict(protected_namespaces=())
    status: str
    model_loaded: bool
    model_version: str
    model_type: str
    timestamp: str
    total_predictions: int


class ModelMetricsResponse(BaseModel):
    """Modelo para respuesta de métricas del modelo"""
    accuracy: Optional[float]
    precision: Optional[float]
    recall: Optional[float]
    f1_score: Optional[float]
    last_updated: str


# ==================== Variables globales ====================

model = None
vectorizer = None
model_version = "unknown"
model_type = "unknown"
prediction_counter = 0


# ==================== Funciones auxiliares ====================

def load_mlflow_model():
    """
    Intenta cargar el modelo desde MLflow, si falla usa el modelo local.
    Prioriza modelos entrenados con optimización bayesiana.
    """
    global model, vectorizer, model_version, model_type

    # Intentar MLflow (opcional)
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = mlflow.tracking.MlflowClient()
        
        try:
            # Buscar modelos registrados de productos
            model_versions = client.search_model_versions(f"name='{MODEL_NAME}'")
            
            if model_versions:
                # Cargar el modelo en producción
                model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
                model = mlflow.pyfunc.load_model(model_uri)
                vectorizer = None  # MLflow models should be pipelines
                model_version = f"mlflow_{MODEL_STAGE}"
                model_type = "MLflow Pipeline"
                print(f"✔ Modelo de productos cargado desde MLflow: {MODEL_NAME}/{MODEL_STAGE}")
                return
        except Exception as e:
            print(f"Modelo de productos no encontrado en MLflow Registry: {e}")
            
    except Exception as e:
        print(f"No se pudo conectar a MLflow: {e}")

    # Cargar modelos locales (orden de prioridad: bayesianos primero)
    model_files = [
        ('models/xgboost_multiclass_calibrated.pkl', 'XGBoost Bayesian Optimized'),
        ('models/random_forest_multiclass_calibrated.pkl', 'Random Forest Bayesian Optimized'),
        ('models/logistic_regression_multiclass.pkl', 'Logistic Regression Bayesian Optimized'),
    ]

    for model_path, model_name in model_files:
        if os.path.exists(model_path):
            try:
                model = joblib.load(model_path)
                vectorizer = None  # Pipeline incluye TF-IDF
                model_version = f"local_{model_name.replace(' ', '_').lower()}"
                model_type = model_name
                print(f"✔ Modelo de productos local cargado: {model_name}")
                return
            except Exception as e:
                print(f"Error cargando {model_path}: {e}")

    # Si no se encuentra ningún modelo
    raise Exception(
        "No se encontró ningún modelo de clasificación de productos. "
        "Ejecuta train_bayesian.py para entrenar los modelos."
    )


# ==================== Eventos de inicio ====================

@app.on_event("startup")
async def startup_event():
    """Evento de inicio de la aplicación."""
    print("========================================")
    print("Iniciando Product Sentiment Analysis API")
    print("========================================")
    
    if model is None:
        try:
            load_mlflow_model()
            print(f"Modelo tipo: {model_type}")
            print(f"Versión: {model_version}")
        except Exception as e:
            print(f"⚠️ Error en startup: {e}")
    
    print("========================================")
    print("API lista para recibir peticiones")
    print("========================================")


# ==================== Endpoints ====================

@app.get("/", response_model=Dict[str, str])
async def root():
    """Endpoint raíz con información de la API."""
    return {
        "message": "Product Sentiment Analysis API",
        "description": "Clasificador de sentimientos para reseñas de productos",
        "version": "1.0.0",
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "predict": "/predict",
            "batch": "/predict/batch",
            "metrics": "/metrics"
        }
    }


@app.get("/health", response_model=ModelHealthResponse)
async def health_check():
    """Verificar el estado de la API y el modelo."""
    return ModelHealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        model_version=model_version,
        model_type=model_type,
        timestamp=datetime.now().isoformat(),
        total_predictions=prediction_counter
    )


@app.get("/metrics", response_model=ModelMetricsResponse)
async def get_metrics():
    """Obtener métricas del modelo actual."""
    # En producción, estas métricas vendrían de MLflow o una base de datos
    return ModelMetricsResponse(
        accuracy=0.87,  # Valores ejemplo
        precision=0.85,
        recall=0.83,
        f1_score=0.84,
        last_updated=datetime.now().isoformat()
    )


@app.post("/predict", response_model=SentimentResponse)
async def predict_product_sentiment(request: ProductReviewRequest):
    """
    Predecir el sentimiento de una reseña de producto.
    
    Clasifica la reseña en tres categorías:
    - Positivo: Reseñas favorables del producto
    - Neutro: Reseñas balanceadas o sin opinión fuerte
    - Negativo: Reseñas desfavorables del producto
    
    Args:
        request: Objeto con el texto de la reseña y opcionalmente el nombre del producto
        
    Returns:
        Predicción del sentimiento con confianza y metadata
    """
    global prediction_counter
    
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Modelo no disponible. Por favor, entrena un modelo primero ejecutando train_bayesian.py"
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
            prediction, confidence = predict_sentiment(request.text, model, vectorizer)
        
        # Convertir predicción numérica a texto
        sentiment_map = {0: "Negativo", 1: "Neutro", 2: "Positivo"}
        sentiment = sentiment_map.get(prediction, "Desconocido")
        
        prediction_counter += 1
        
        return SentimentResponse(
            sentiment=sentiment,
            confidence=float(confidence),
            product=request.product_name,
            prediction_time=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error al procesar la predicción: {str(e)}"
        )


@app.post("/predict/batch", response_model=List[SentimentResponse])
async def predict_batch_products(request: BatchProductReviewRequest):
    """
    Predecir el sentimiento de múltiples reseñas de productos.
    
    Procesa hasta 100 reseñas en una sola solicitud.
    """
    global prediction_counter
    
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
    product_names = request.product_names or [None] * len(request.reviews)
    
    for i, review in enumerate(request.reviews):
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
                prediction, confidence = predict_sentiment(review, model, vectorizer)
            
            sentiment_map = {0: "Negativo", 1: "Neutro", 2: "Positivo"}
            sentiment = sentiment_map.get(prediction, "Desconocido")
            
            prediction_counter += 1
            
            results.append(SentimentResponse(
                sentiment=sentiment,
                confidence=float(confidence),
                product=product_names[i] if i < len(product_names) else None,
                prediction_time=datetime.now().isoformat()
            ))
            
        except Exception as e:
            # En caso de error, agregar resultado con confianza 0
            results.append(SentimentResponse(
                sentiment="Error",
                confidence=0.0,
                product=product_names[i] if i < len(product_names) else None,
                prediction_time=datetime.now().isoformat()
            ))
    
    return results


@app.post("/reload-model")
async def reload_model():
    """
    Recargar el modelo (útil para actualizar a una nueva versión).
    
    Intenta cargar la versión más reciente del modelo desde MLflow
    o los archivos locales.
    """
    try:
        load_mlflow_model()
        return {
            "status": "success",
            "message": "Modelo de productos recargado exitosamente",
            "model_version": model_version,
            "model_type": model_type
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error al recargar el modelo: {str(e)}"
        )


@app.get("/model-info")
async def get_model_info():
    """Obtener información detallada sobre el modelo actual."""
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="No hay modelo cargado"
        )
    
    return {
        "model_type": model_type,
        "model_version": model_version,
        "supported_classes": ["Positivo", "Neutro", "Negativo"],
        "input_type": "text",
        "max_input_length": 5000,
        "language": "Spanish",
        "domain": "Product Reviews",
        "optimization": "Bayesian Optimization" if "bayesian" in model_version.lower() else "Standard"
    }


# ==================== Main ====================

if __name__ == "__main__":
    import uvicorn
    
    # Obtener configuración del entorno
    host = os.getenv('API_HOST', '0.0.0.0')
    port = int(os.getenv('API_PORT', '8000'))
    
    # Cargar modelo al iniciar
    try:
        load_mlflow_model()
    except Exception as e:
        print(f"⚠️ Advertencia: {e}")
        print("La API iniciará sin modelo. Usa /reload-model después de entrenar.")
    
    # Ejecutar servidor
    print(f"\n🚀 Iniciando servidor en {host}:{port}")
    print(f"📚 Documentación disponible en http://{host}:{port}/docs")
    
    uvicorn.run(
        "api:app",
        host=host,
        port=port,
        reload=True
    )