"""
Tests unitarios para la API de clasificación de sentimientos de productos.
Use: pytest -v -W ignore -> ignore all warnings.
"""
import pytest
from fastapi.testclient import TestClient
import sys
import os

# Agregar el directorio padre al path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.api import app

client = TestClient(app)


class TestAPI:
    """Tests para los endpoints de la API de productos."""
    
    # def test_root_endpoint(self):
    #     """Test del endpoint raíz."""
    #     response = client.get("/")
    #     assert response.status_code == 200
    #     assert "message" in response.json()
    #     assert response.json()["message"] == "Product Sentiment Analysis API"
    #     assert "endpoints" in response.json()
    
    def test_health_endpoint(self):
        """Test del health check."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "model_version" in data
        assert "model_type" in data
        assert "timestamp" in data
        assert "total_predictions" in data
    
    def test_metrics_endpoint(self):
        """Test del endpoint de métricas."""
        response = client.get("/metrics")
        assert response.status_code == 200
        data = response.json()
        assert "accuracy" in data
        assert "precision" in data
        assert "recall" in data
        assert "f1_score" in data
    
    def test_model_info_endpoint(self):
        """Test del endpoint de información del modelo."""
        response = client.get("/model-info")
        # Puede ser 200 si hay modelo o 503 si no
        if response.status_code == 200:
            data = response.json()
            assert "model_type" in data
            assert "supported_classes" in data
            assert data["supported_classes"] == ["Positivo", "Neutro", "Negativo"]
    
    def test_predict_endpoint_valid_positive(self):
        """Test de predicción con reseña positiva en español."""
        response = client.post(
            "/predict",
            json={
                "text": "Este producto es excelente, superó todas mis expectativas. La calidad es increíble y funciona perfectamente.",
                "product_name": "Smartphone"
            }
        )
        
        # Si el modelo está cargado
        if response.status_code == 200:
            data = response.json()
            assert "sentiment" in data
            assert data["sentiment"] in ["Positivo", "Neutro", "Negativo"]
            assert "confidence" in data
            assert 0 <= data["confidence"] <= 1
            assert "prediction_time" in data
            assert "product" in data
        # Si el modelo no está cargado
        elif response.status_code == 503:
            assert "detail" in response.json()
    
    def test_predict_endpoint_valid_negative(self):
        """Test de predicción con reseña negativa en español."""
        response = client.post(
            "/predict",
            json={
                "text": "Muy decepcionado con este producto. No funciona como debería y la calidad es pésima.",
                "product_name": "Tablet"
            }
        )
        
        if response.status_code == 200:
            data = response.json()
            assert "sentiment" in data
            assert data["sentiment"] in ["Positivo", "Neutro", "Negativo"]
    
    def test_predict_endpoint_valid_neutral(self):
        """Test de predicción con reseña neutra en español."""
        response = client.post(
            "/predict",
            json={
                "text": "El producto cumple su función básica, nada extraordinario pero tampoco está mal.",
                "product_name": "Monitor"
            }
        )
        
        if response.status_code == 200:
            data = response.json()
            assert "sentiment" in data
            assert data["sentiment"] in ["Positivo", "Neutro", "Negativo"]
    
    def test_predict_endpoint_short_text(self):
        """Test con texto muy corto (debe fallar validación)."""
        response = client.post(
            "/predict",
            json={"text": "Malo"}
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
        """Test de predicción por lotes con reseñas en español."""
        response = client.post(
            "/predict/batch",
            json={
                "reviews": [
                    "Excelente producto, lo recomiendo ampliamente. Funciona perfectamente.",
                    "Terrible calidad, no lo compren. Es una pérdida de dinero.",
                    "Está bien, cumple con lo básico pero nada especial."
                ],
                "product_names": ["Auriculares", "Teclado", "Mouse"]
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
                assert "product" in result
    
    # def test_batch_predict_too_many(self):
    #     """Test con más de 100 reseñas."""
    #     reviews = ["Este es un producto de prueba"] * 101
    #     response = client.post(
    #         "/predict/batch",
    #         json={"reviews": reviews}
    #     )
    #     assert response.status_code == 400
    #     assert "100" in response.json()["detail"]
    
    def test_batch_predict_empty_list(self):
        """Test con lista vacía de reseñas."""
        response = client.post(
            "/predict/batch",
            json={"reviews": []}
        )
        assert response.status_code == 422  # Validación de min_items=1
    
    def test_reload_model(self):
        """Test del endpoint de recarga de modelo."""
        response = client.post("/reload-model")
        # Puede ser 200 (éxito) o 500 (error al cargar)
        assert response.status_code in [200, 500]
        if response.status_code == 200:
            data = response.json()
            assert "status" in data
            assert "model_version" in data
            assert "model_type" in data


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
    
    def test_review_max_length_validation(self):
        """Test del límite máximo de caracteres (5000)."""
        long_text = "Este producto es bueno. " * 250  # Aprox 6000 caracteres
        response = client.post("/predict", json={"text": long_text})
        assert response.status_code == 422  # Debe fallar por exceder max_length
        
        valid_long_text = "Este producto es excelente. " * 100  # Aprox 2900 caracteres
        response = client.post("/predict", json={"text": valid_long_text})
        assert response.status_code in [200, 503]  # Válido
    
    def test_invalid_json(self):
        """Test con JSON inválido."""
        response = client.post(
            "/predict",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422
    
    def test_product_name_optional(self):
        """Test que el nombre del producto es opcional."""
        response = client.post(
            "/predict",
            json={
                "text": "Este producto funciona muy bien, estoy satisfecho con la compra."
                # Sin product_name
            }
        )
        # No debe fallar por falta de product_name
        assert response.status_code in [200, 503]


class TestProductSpecific:
    """Tests específicos para diferentes tipos de productos."""
    
    def test_electronics_reviews(self):
        """Test con reseñas de productos electrónicos."""
        electronics_reviews = [
            "La batería del smartphone dura todo el día, excelente rendimiento.",
            "El laptop se calienta demasiado y es muy lento.",
            "Los auriculares tienen buen sonido pero son incómodos."
        ]
        
        for review in electronics_reviews:
            response = client.post("/predict", json={"text": review})
            if response.status_code == 200:
                data = response.json()
                assert data["sentiment"] in ["Positivo", "Neutro", "Negativo"]
    
    def test_mixed_language_review(self):
        """Test con reseña que mezcla español e inglés (común en reseñas reales)."""
        response = client.post(
            "/predict",
            json={
                "text": "El producto está OK, tiene buen performance pero el design podría mejorar.",
                "product_name": "Tablet"
            }
        )
        # Debe poder procesar texto mixto
        assert response.status_code in [200, 503]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-W", "ignore"])