"""
Utilidades para el proyecto de clasificación de sentimientos.
Formato esperado del JSON:
{
    "producto": "Auriculares",
    "sentimiento": "positivo",  # puede ser: positivo, negativo, neutro
    "reseña": "Funciona de maravilla, el Auriculares es rápido y confiable."
}
"""
import os
import re
import pickle
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import json

# Descargar recursos de NLTK si no están disponibles
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

try:
    from src.quality_and_lineage import validate_data_quality, lineage_tracker
except:
    from quality_and_lineage import validate_data_quality, lineage_tracker

# Configurar stopwords en español
STOPWORDS = set(stopwords.words('spanish'))

# Mapeo de sentimientos a valores numéricos
SENTIMENT_MAPPING = {'positivo': 2, 'neutro': 1, 'negativo': 0}
SENTIMENT_NORMALIZATION = {
    'positivo': 'positivo', 'positive': 'positivo',
    'neutro': 'neutro', 'neutral': 'neutro',
    'negativo': 'negativo', 'negative': 'negativo'
}
SENTIMENT_LABELS = {v: k for k, v in SENTIMENT_MAPPING.items()}


def clean_text(text: str, producto: str = None) -> str:
    """Limpia y preprocesa el texto, eliminando stopwords y el nombre del producto."""
    text = text.lower()
    text = re.sub(r'<[^>]+>', '', text)  # eliminar HTML
    text = re.sub(r'[^a-záéíóúñü\s]', '', text)  # eliminar caracteres no deseados
    if producto:
        text = re.sub(re.escape(producto.lower()), '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 2]
    return ' '.join(tokens)


def load_product_reviews(data_path: str = 'data/reseñas_productos_sintetico.json',
                         sample_size: int = None) -> pd.DataFrame:
    """Carga reseñas desde JSON, valida calidad y normaliza sentimientos."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Archivo de datos no encontrado: {data_path}")

    with open(data_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)

    df = pd.DataFrame(json_data)

    # Validar calidad de datos
    validate_data_quality(df, required_columns=['producto', 'sentimiento', 'reseña'], quality_threshold=0.95)

    # Normalizar sentimientos
    df['sentimiento'] = df['sentimiento'].str.lower().map(SENTIMENT_NORMALIZATION)
    if df['sentimiento'].isna().any():
        invalid = df[df['sentimiento'].isna()]
        raise ValueError(f"Sentimientos no reconocidos: {invalid['sentimiento'].unique()}")

    if sample_size:
        df = df.sample(n=min(sample_size, len(df)), random_state=42)

    print(f"✓ Dataset cargado desde {data_path} ({len(df)} reseñas, {df['producto'].nunique()} productos)")
    return df


def prepare_data(df: pd.DataFrame, test_size: float = 0.2, save_clean_path: str = "data/df_limpio.csv") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Limpia texto, elimina duplicados, valida calidad, registra linaje y divide en train/test."""
    df = df.copy()
    print("\nLimpiando texto...")
    df['reseña_limpia'] = df.apply(lambda row: clean_text(row['reseña'], row['producto']), axis=1)
    # df = df.drop_duplicates(subset=['reseña_limpia'])

    # Guardar CSV limpio
    df.to_csv(save_clean_path, index=False, encoding='utf-8')
    print(f"✓ DataFrame limpio guardado en {save_clean_path}")

    # Registrar linaje
    lineage_tracker.register_lineage(
        process_name="Limpieza y preparación de texto",
        input_assets=["reseñas_productos_sintetico.json"],
        output_assets=[save_clean_path],
        params={"drop_duplicates": True, "clean_text_applied": True}
    )

    # Mapear sentimientos a valores numéricos
    df['sentimiento_numerico'] = df['sentimiento'].map(SENTIMENT_MAPPING)
    df = df[df['reseña_limpia'].str.len() > 0]

    # Dividir estratificado
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=42, stratify=df['sentimiento_numerico'])
    print(f"\n✓ Datos preparados: {len(train_df)} train / {len(test_df)} test")
    return train_df, test_df


def get_features_and_labels(df: pd.DataFrame, vectorizer: TfidfVectorizer = None, fit: bool = False) -> Tuple[np.ndarray, np.ndarray, TfidfVectorizer]:
    """Extrae características TF-IDF y etiquetas."""
    if vectorizer is None:
        vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2), min_df=2, max_df=0.95)
    X = vectorizer.fit_transform(df['reseña_limpia']) if fit else vectorizer.transform(df['reseña_limpia'])
    y = df['sentimiento_numerico'].values
    return X, y, vectorizer


def save_model(model, vectorizer, model_path: str, metadata: Dict = None) -> None:
    """Guarda modelo, vectorizador y metadatos opcionales."""
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model_dict = {
        'model': model,
        'vectorizer': vectorizer,
        'sentiment_mapping': SENTIMENT_MAPPING,
        'sentiment_labels': SENTIMENT_LABELS,
        'metadata': metadata or {}
    }
    with open(model_path, 'wb') as f:
        pickle.dump(model_dict, f)
    print(f"✓ Modelo guardado en: {model_path}")


def load_model(model_path: str) -> Tuple:
    """Carga modelo, vectorizador y mapeos de sentimientos."""
    with open(model_path, 'rb') as f:
        model_dict = pickle.load(f)
    return model_dict['model'], model_dict['vectorizer'], model_dict.get('sentiment_mapping', SENTIMENT_MAPPING), model_dict.get('sentiment_labels', SENTIMENT_LABELS)


def predict_sentiment(text: str, model, vectorizer, return_probabilities: bool = False) -> Dict:
    """Predice el sentimiento de un texto."""
    cleaned_text = clean_text(text)
    text_vector = vectorizer.transform([cleaned_text])
    prediction_numeric = model.predict(text_vector)[0]
    prediction_label = SENTIMENT_LABELS[prediction_numeric]
    probabilities = model.predict_proba(text_vector)[0]
    confidence = probabilities.max()
    result = {'sentimiento': prediction_label, 'confianza': float(confidence), 'texto_procesado': cleaned_text}
    if return_probabilities:
        result['probabilidades'] = {SENTIMENT_LABELS[i]: float(prob) for i, prob in enumerate(probabilities)}
    return result


def predict_batch(texts: List[str], model, vectorizer, include_product: List[str] = None) -> pd.DataFrame:
    """Predice sentimiento para un lote de textos."""
    results = []
    for i, text in enumerate(texts):
        prediction = predict_sentiment(text, model, vectorizer, return_probabilities=True)
        row = {
            'reseña_original': text,
            'sentimiento_predicho': prediction['sentimiento'],
            'confianza': prediction['confianza'],
            'prob_positivo': prediction['probabilidades']['positivo'],
            'prob_neutro': prediction['probabilidades']['neutro'],
            'prob_negativo': prediction['probabilidades']['negativo']
        }
        if include_product:
            row['producto'] = include_product[i] if i < len(include_product) else 'N/A'
        results.append(row)
    return pd.DataFrame(results)
