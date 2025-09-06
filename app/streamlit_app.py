"""
Interfaz web con Streamlit para el clasificador de sentimientos de productos.
Clasificación multiclase: Negativo, Neutro, Positivo
"""
import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time

# Configuración de la página
st.set_page_config(
    page_title="Product Sentiment Analyzer",
    page_icon="🛍️",
    layout="wide"
)

# URL de la API
API_URL = "http://api:8000"  # docker compose
# API_URL = "http://localhost:8000"  # local host

# Estilos CSS personalizados
st.markdown("""
<style>
.positive {
    background-color: #d4edda;
    border-color: #c3e6cb;
    color: #155724;
    padding: 10px;
    border-radius: 5px;
    margin: 10px 0;
}
.neutral {
    background-color: #fff3cd;
    border-color: #ffeeba;
    color: #856404;
    padding: 10px;
    border-radius: 5px;
    margin: 10px 0;
}
.negative {
    background-color: #f8d7da;
    border-color: #f5c6cb;
    color: #721c24;
    padding: 10px;
    border-radius: 5px;
    margin: 10px 0;
}
</style>
""", unsafe_allow_html=True)


def check_api_health():
    """Verificar si la API está disponible."""
    try:
        response = requests.get(f"{API_URL}/health")
        return response.json() if response.status_code == 200 else None
    except:
        return None


def predict_sentiment(text):
    """Llamar a la API para predecir sentimiento."""
    try:
        response = requests.post(f"{API_URL}/predict", json={"text": text})
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except Exception as e:
        st.error(f"Error al conectar con la API: {e}")
        return None


def main():
    # Header
    st.title("🛍️ Product Sentiment Analyzer")
    st.markdown("### Analiza el sentimiento de reseñas de productos con Machine Learning")
    
    # Sidebar
    with st.sidebar:
        st.header("Información")
        health = check_api_health()
        if health:
            st.success("✅ API Conectada")
            st.info(f"Modelo: {health['model_version']}")
        else:
            st.error("❌ API No Disponible")
            st.warning("Asegúrate de ejecutar: `python app/api.py`")
        
        st.markdown("---")
        st.markdown("""
        ### Cómo usar:
        1. Escribe una reseña de producto
        2. Haz clic en 'Analizar'
        3. Obtén el sentimiento y confianza
        
        ### Características:
        - Modelo ML entrenado con 3 clases
        - Visualización de resultados en tiempo real
        - API REST con FastAPI
        - Tracking con MLflow
        """)
        
        # Ejemplos
        st.markdown("### Ejemplos para probar:")
        examples = {
            "Positivo 😊": "El producto llegó rápido y funciona perfectamente. Estoy muy satisfecho con la compra.",
            "Neutral 😐": "El producto está bien, cumple con lo esperado pero nada fuera de lo común.",
            "Negativo 😞": "El producto llegó defectuoso y el servicio de atención al cliente fue pésimo."
        }
        
        for label, text in examples.items():
            if st.button(label):
                st.session_state.review_text = text
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Escribe tu reseña")
        review_text = st.text_area(
            "Reseña de producto:",
            value=st.session_state.get('review_text', ''),
            height=150,
            placeholder="Escribe aquí tu opinión sobre un producto..."
        )
        
        if st.button("🔍 Analizar Sentimiento", type="primary"):
            if review_text and len(review_text) > 10:
                with st.spinner("Analizando..."):
                    time.sleep(0.5)
                    result = predict_sentiment(review_text)
                    
                    if result:
                        st.session_state.last_result = result
                        st.session_state.last_review = review_text
                        
                        sentiment = result['sentiment']
                        confidence = result['confidence']
                        
                        if sentiment == "Positivo":
                            st.markdown(f'<div class="positive"><h3>😊 Sentimiento: {sentiment}</h3><p>Confianza: {confidence:.1%}</p></div>', unsafe_allow_html=True)
                            st.balloons()
                        elif sentiment == "Neutro":
                            st.markdown(f'<div class="neutral"><h3>😐 Sentimiento: {sentiment}</h3><p>Confianza: {confidence:.1%}</p></div>', unsafe_allow_html=True)
                        else:
                            st.markdown(f'<div class="negative"><h3>😞 Sentimiento: {sentiment}</h3><p>Confianza: {confidence:.1%}</p></div>', unsafe_allow_html=True)
                        
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number+delta",
                            value=confidence*100,
                            domain={'x': [0, 1], 'y': [0, 1]},
                            title={'text': "Nivel de Confianza"},
                            delta={'reference': 80},
                            gauge={
                                'axis': {'range': [None, 100]},
                                'bar': {'color': "darkblue"},
                                'steps': [
                                    {'range': [0, 50], 'color': "lightgray"},
                                    {'range': [50, 80], 'color': "gray"}
                                ],
                                'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 90}
                            }
                        ))
                        fig.update_layout(height=300)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.error("No se pudo obtener la predicción")
            else:
                st.warning("Por favor, escribe una reseña más larga (mínimo 10 caracteres)")
    
    with col2:
        st.header("Estadísticas")
        if 'history' not in st.session_state:
            st.session_state.history = []
        
        if 'last_result' in st.session_state and st.session_state.last_result:
            st.session_state.history.append({
                'time': datetime.now(),
                'sentiment': st.session_state.last_result['sentiment'],
                'confidence': st.session_state.last_result['confidence']
            })
        
        if st.session_state.history:
            df = pd.DataFrame(st.session_state.history)
            sentiment_counts = df['sentiment'].value_counts()
            
            fig_pie = px.pie(
                values=sentiment_counts.values,
                names=sentiment_counts.index,
                title="Distribución de Sentimientos",
                color_discrete_map={'Positivo': '#28a745', 'Neutral': '#ffc107', 'Negativo': '#dc3545'}
            )
            st.plotly_chart(fig_pie, use_container_width=True)
            
            avg_confidence = df['confidence'].mean()
            st.metric("Confianza Promedio", f"{avg_confidence:.1%}")
            st.metric("Total Analizado", len(df))
            
            if st.button("🗑️ Limpiar Historial"):
                st.session_state.history = []
                st.session_state.last_result = None       
                st.experimental_rerun()
    
    with st.expander("🔧 Opciones Avanzadas (Análisis por Lotes)"):
        st.header("Análisis por Lotes")
        batch_text = st.text_area(
            "Pega múltiples reseñas (una por línea):",
            height=200,
            placeholder="Primera reseña...\nSegunda reseña...\nTercera reseña..."
        )
        
        if st.button("Analizar Lote"):
            if batch_text:
                reviews = [r.strip() for r in batch_text.split('\n') if r.strip()]
                if reviews:
                    with st.spinner(f"Analizando {len(reviews)} reseñas..."):
                        try:
                            response = requests.post(f"{API_URL}/predict/batch", json={"reviews": reviews})
                            if response.status_code == 200:
                                results = response.json()
                                batch_df = pd.DataFrame([
                                    {
                                        'Reseña': rev[:50] + '...' if len(rev) > 50 else rev,
                                        'Sentimiento': res['sentiment'],
                                        'Confianza': f"{res['confidence']:.1%}"
                                    } for rev, res in zip(reviews, results)
                                ])
                                st.dataframe(batch_df)
                                
                                col1, col2 = st.columns(2)
                                with col1:
                                    positive_count = sum(1 for r in results if r['sentiment'] == 'Positivo')
                                    neutral_count = sum(1 for r in results if r['sentiment'] == 'Neutro')
                                    negative_count = sum(1 for r in results if r['sentiment'] == 'Negativo')
                                    st.metric("Positivo", f"{positive_count}/{len(results)}")
                                    st.metric("Neutral", f"{neutral_count}/{len(results)}")
                                    st.metric("Negativo", f"{negative_count}/{len(results)}")
                                
                                with col2:
                                    avg_conf = sum(r['confidence'] for r in results) / len(results)
                                    st.metric("Confianza Promedio", f"{avg_conf:.1%}")
                        except Exception as e:
                            st.error(f"Error: {e}")
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center'>
        <p>Creado usando Streamlit, FastAPI y MLflow</p>
        <p>Product Sentiment Analyzer v1.0</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
