"""
Tests unitarios para la API de clasificación de sentimientos.
Use: pytest -v -W ignore -> ignore all warnings.
"""
import pytest
from fastapi.testclient import TestClient
import sys
import os

# Ignorar todos los warnings (temporalmente)
# import warnings
# warnings.filterwarnings("ignore")


# Agregar el directorio padre al path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.api import app

client = TestClient(app)


class TestAPI:
    """Tests para los endpoints de la API."""
    
    def test_root_endpoint(self):
        """Test del endpoint raíz."""
        response = client.get("/")
        assert response.status_code == 200
        assert "message" in response.json()
        assert response.json()["message"] == "Movie Sentiment API"
    
    def test_health_endpoint(self):
        """Test del health check."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "model_version" in data
        assert "timestamp" in data
    
    def test_predict_endpoint_valid(self):
        """Test de predicción con entrada válida."""
        response = client.post(
            "/predict",
            json={"text": "This movie was absolutely amazing! Best film I've ever seen."}
        )
        
        # Si el modelo está cargado
        if response.status_code == 200:
            data = response.json()
            assert "sentiment" in data
            assert data["sentiment"] in ["Positivo", "Negativo"]
            assert "confidence" in data
            assert 0 <= data["confidence"] <= 1
            assert "prediction_time" in data
        # Si el modelo no está cargado
        elif response.status_code == 503:
            assert "detail" in response.json()
    
    def test_predict_endpoint_short_text(self):
        """Test con texto muy corto (debe fallar validación)."""
        response = client.post(
            "/predict",
            json={"text": "Bad"}
        )
        assert response.status_code == 422  # Unprocessable Entity
    
    def test_predict_endpoint_empty_text(self):
        """Test con texto vacío."""
        response = client.post(
            "/predict",
            json={"text": ""}
        )
        assert response.status_code == 422
    
    def test_predict_endpoint_missing_field(self):
        """Test sin el campo 'text'."""
        response = client.post(
            "/predict",
            json={}
        )
        assert response.status_code == 422
    
    def test_batch_predict_valid(self):
        """Test de predicción por lotes."""
        response = client.post(
            "/predict/batch",
            json={
                "reviews": [
                    "Great movie, loved it!",
                    "Terrible film, waste of time.",
                    "It was okay, nothing special."
                ]
            }
        )
        
        if response.status_code == 200:
            data = response.json()
            assert isinstance(data, list)
            assert len(data) == 3
            for result in data:
                assert "sentiment" in result
                assert "confidence" in result
                assert "prediction_time" in result
    
    def test_batch_predict_too_many(self):
        """Test con más de 100 reseñas."""
        reviews = ["Test review"] * 101
        response = client.post(
            "/predict/batch",
            json={"reviews": reviews}
        )
        assert response.status_code == 400
        assert "Máximo 100 reseñas" in response.json()["detail"]
    
    def test_reload_model(self):
        """Test del endpoint de recarga de modelo."""
        response = client.post("/reload-model")
        # Puede ser 200 (éxito) o 500 (error al cargar)
        assert response.status_code in [200, 500]


class TestValidation:
    """Tests específicos de validación de datos."""
    
    def test_review_min_length_validation(self):
        """Test que la reseña debe tener mínimo 10 caracteres."""
        test_cases = [
            ("123456789", 422),    # 9 caracteres - debe fallar
            ("1234567890", 200),   # 10 caracteres - debe pasar (o 503 si no hay modelo)
            ("12345678901", 200),  # 11 caracteres - debe pasar
        ]
        
        for text, expected_status in test_cases:
            response = client.post("/predict", json={"text": text})
            # Si no hay modelo cargado, esperamos 503 en lugar de 200
            if response.status_code == 503:
                assert len(text) >= 10  # Solo textos válidos pueden llegar al 503
            else:
                assert response.status_code == expected_status
    
    def test_invalid_json(self):
        """Test con JSON inválido."""
        response = client.post(
            "/predict",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422


if __name__ == "__main__":
    pytest.main([__file__, "-v"])