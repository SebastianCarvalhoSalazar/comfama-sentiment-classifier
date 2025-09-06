"""
Script de entrenamiento con optimización bayesiana de hiperparámetros.
Adaptado para clasificación multiclase (positivo, neutro, negativo).
Incluye Logistic Regression, Random Forest y XGBoost.
Integra validación de calidad, registro de linaje y métricas obligatorias.
"""
from utils import load_product_reviews, prepare_data, SENTIMENT_LABELS
from pathlib import Path
import os
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
from sklearn.model_selection import StratifiedKFold
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
from skopt.callbacks import CheckpointSaver
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import Memory
from dotenv import load_dotenv
import joblib
import warnings
from quality_and_lineage import validate_data_quality, lineage_tracker
import shap

warnings.filterwarnings("ignore")
load_dotenv()


# Configuración MLflow
MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', 'file:./mlruns')
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Crear directorios
os.makedirs('models', exist_ok=True)
os.makedirs('logs', exist_ok=True)
os.makedirs('cache_lr', exist_ok=True)
os.makedirs('cache_rf', exist_ok=True)
os.makedirs('cache_xgb', exist_ok=True)
os.makedirs('data', exist_ok=True)

# ------------------------- CLASES Y FUNCIONES AUXILIARES -------------------------


class CustomLogCallback:
    """Callback personalizado para logging detallado"""

    def __init__(self, log_path):
        self.log_path = Path(log_path)
        self.iteration = 0
        self.log_file = open(self.log_path, 'w')
        self.log_file.write(f"Optimization started at {datetime.now()}\n")
        self.log_file.write("="*60 + "\n")

    def __call__(self, res):
        self.iteration += 1
        current_params = res.x_iters[-1]
        current_score = res.func_vals[-1]
        best_score = min(res.func_vals)
        self.log_file.write(
            f"\n--- Iteration {self.iteration}/{len(res.x_iters)} ---\n")
        self.log_file.write(
            f"Timestamp: {datetime.now().strftime('%H:%M:%S')}\n")
        self.log_file.write(
            f"Current params: {dict(zip(res.space.dimension_names, current_params))}\n")
        self.log_file.write(f"Current CV score: {-current_score:.6f}\n")
        self.log_file.write(f"Best score so far: {-best_score:.6f}\n")
        self.log_file.flush()
        print(
            f"[{self.iteration}] Score: {-current_score:.6f} | Best: {-best_score:.6f}")

    def __del__(self):
        if hasattr(self, 'log_file'):
            self.log_file.write(
                f"\nOptimization completed at {datetime.now()}\n")
            self.log_file.close()


def plot_confusion_matrix(y_true, y_pred, model_name, save_path=None):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negativo', 'Neutro', 'Positivo'],
                yticklabels=['Negativo', 'Neutro', 'Positivo'])
    plt.title(f'Matriz de Confusión - {model_name}')
    plt.ylabel('Valor Real')
    plt.xlabel('Predicción')
    if save_path:
        plt.savefig(save_path)
        mlflow.log_artifact(save_path)
    plt.close()


def validate_and_register_lineage(df: pd.DataFrame, process_name: str, input_assets: list, output_csv: str):
    """Valida calidad y registra linaje antes de entrenamiento."""
    validate_data_quality(df, required_columns=[
                          'reseña_limpia', 'sentimiento'])
    df.to_csv(output_csv, index=False, encoding='utf-8')
    lineage_tracker.register_lineage(
        process_name=process_name,
        input_assets=input_assets,
        output_assets=[output_csv],
        params={"drop_duplicates": True, "clean_text_applied": True}
    )


def calculate_dir(y_true, y_pred, sensitive_attr):
    """
    Calcula Disparate Impact Ratio (DIR)
    sensitive_attr: array o serie que indica grupo protegido (0 o 1)
    """
    group_1 = (sensitive_attr == 1)
    group_0 = (sensitive_attr == 0)
    p_group1 = np.mean(y_pred[group_1])
    p_group0 = np.mean(y_pred[group_0])
    if p_group0 == 0:
        return np.inf
    return p_group1 / p_group0


def check_mandatory_metrics(y_true, y_pred, y_pred_proba, sensitive_attr=None):
    """
    Verifica que las métricas obligatorias cumplan los thresholds:
    Precisión ≥0.85, Recall ≥0.8, AUC ≥0.75, DIR entre 0.8 y 1.25
    """
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    try:
        auc = roc_auc_score(pd.get_dummies(y_true),
                            y_pred_proba, multi_class='ovr')
    except:
        auc = 0.0  # fallback
    if sensitive_attr is not None:
        dir_ratio = calculate_dir(y_true, y_pred, sensitive_attr)
    else:
        dir_ratio = 1.0  # Default si no hay grupo protegido

    mlflow.log_metric("precision_weighted", precision)
    mlflow.log_metric("recall_weighted", recall)
    mlflow.log_metric("auc_weighted", auc)
    mlflow.log_metric("dir", dir_ratio)

    if precision < 0.85 or recall < 0.8 or auc < 0.75 or not (0.8 <= dir_ratio <= 1.25):
        raise ValueError(
            f"Métricas obligatorias no cumplidas: Precision={precision:.3f}, Recall={recall:.3f}, AUC={auc:.3f}, DIR={dir_ratio:.3f}")


def log_shap_values(model, X_test, feature_names, model_name, timestamp, multiclass=True):
    """
    Calcula SHAP values y los guarda como artefactos.
    Maneja problemas multiclase generando un plot por clase.
    Convierte matrices sparse a dense si es necesario.
    """
    # Extraer clasificador y tfidf si es pipeline
    if hasattr(model, 'steps'):
        classifier = model.named_steps['classifier']
        tfidf = model.named_steps['tfidf']
        X_transformed = tfidf.transform(X_test)
    else:
        classifier = model
        X_transformed = X_test

    # Convertir sparse a dense
    if hasattr(X_transformed, "toarray"):
        X_dense = X_transformed.toarray()
    else:
        X_dense = X_transformed

    # Explainer
    explainer = shap.Explainer(classifier, X_dense)
    shap_values = explainer(X_dense)

    # Detectar si es multiclase
    n_classes = shap_values.values.shape[-1] if multiclass else 1

    for i in range(n_classes):
        # Seleccionar SHAP values de la clase i
        class_shap_values = shap_values.values[:, :,
                                               i] if multiclass else shap_values.values

        # Summary plot
        summary_path = f'logs/shap_summary_{model_name}_class{i}_{timestamp}.png'
        shap.summary_plot(class_shap_values, X_dense,
                          feature_names=feature_names, show=False)
        plt.savefig(summary_path, bbox_inches='tight')
        plt.close()
        mlflow.log_artifact(summary_path)

        # Dependence plot del primer feature
        dep_path = f'logs/shap_dependence_{model_name}_class{i}_{timestamp}.png'
        shap.dependence_plot(0, class_shap_values, X_dense,
                             feature_names=feature_names, show=False)
        plt.savefig(dep_path, bbox_inches='tight')
        plt.close()
        mlflow.log_artifact(dep_path)


# ------------------------- ENTRENAMIENTO DE MODELOS -------------------------


def train_logistic_regression_bayesian(train_df, test_df, max_features=5000):
    """Entrena Logistic Regression con optimización bayesiana y métricas obligatorias"""
    X_train = train_df['reseña_limpia']
    y_train = train_df['sentimiento_numerico']
    X_test = test_df['reseña_limpia']
    y_test = test_df['sentimiento_numerico']

    # Validación y linaje
    validate_and_register_lineage(train_df, "Preparación de datos entrenamiento LR",
                                  input_assets=[
                                      "reseñas_productos_sintetico.json"],
                                  output_csv="data/train_clean_lr.csv")
    validate_and_register_lineage(test_df, "Preparación de datos test LR",
                                  input_assets=[
                                      "reseñas_productos_sintetico.json"],
                                  output_csv="data/test_clean_lr.csv")

    with mlflow.start_run(run_name=f"logistic_regression_{datetime.now():%Y%m%d_%H%M%S}"):
        mlflow.log_param("model_type", "logistic_regression_bayesian")
        mlflow.log_param("n_classes", 3)

        memory = Memory('./cache_lr', verbose=0)
        pipe = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))),
            ('classifier', LogisticRegression(random_state=42,
             multi_class='multinomial', max_iter=1000))
        ], memory=memory)

        param_space = {
            'tfidf__max_features': Integer(50, 100),
            'tfidf__min_df': Integer(3, 5),
            'tfidf__max_df': Real(0.7, 1.0),
            'classifier__C': Real(0.001, 100, prior='log-uniform'),
            'classifier__solver': Categorical(['lbfgs', 'newton-cg', 'sag', 'saga']),
            'classifier__class_weight': Categorical(['balanced', None])
        }

        print("Iniciando optimización bayesiana LR...")
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_callback = CustomLogCallback(
            f"logs/optimization_lr_{timestamp}.txt")
        checkpoint_callback = CheckpointSaver(
            f"logs/checkpoint_lr_{timestamp}.pkl", compress=6)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        bayes_search = BayesSearchCV(
            estimator=pipe,
            search_spaces=param_space,
            n_iter=15,
            scoring='f1_weighted',
            cv=skf,
            n_jobs=-1,
            random_state=42,
            verbose=0
        )

        bayes_search.fit(X_train, y_train, callback=[
                         checkpoint_callback, log_callback])

        y_pred = bayes_search.predict(X_test)
        y_pred_proba = bayes_search.predict_proba(X_test)

        # Log SHAP values
        feature_names = bayes_search.best_estimator_.named_steps['tfidf'].get_feature_names_out(
        )
        log_shap_values(
            bayes_search.best_estimator_,
            X_test,
            feature_names,
            model_name='logistic_regression',
            timestamp=timestamp
        )

        # Validar métricas obligatorias
        check_mandatory_metrics(y_test, y_pred, y_pred_proba)

        report = classification_report(y_test, y_pred, target_names=[
                                       'negativo', 'neutro', 'positivo'])
        print(report)
        plot_confusion_matrix(y_test, y_pred, 'Logistic Regression',
                              save_path=f'logs/confusion_matrix_lr_{timestamp}.png')

        # Guardar modelo
        model_path = 'models/logistic_regression_multiclass.pkl'
        joblib.dump(bayes_search.best_estimator_, model_path)
        print(f"Modelo guardado en: {model_path}")

        return bayes_search.best_estimator_, bayes_search.best_params_


def train_random_forest_bayesian(train_df, test_df, max_features=5000):
    """Entrena Random Forest con optimización bayesiana y métricas obligatorias"""
    X_train = train_df['reseña_limpia']
    y_train = train_df['sentimiento_numerico']
    X_test = test_df['reseña_limpia']
    y_test = test_df['sentimiento_numerico']

    # Validación y linaje
    validate_and_register_lineage(train_df, "Preparación de datos entrenamiento RF",
                                  input_assets=[
                                      "reseñas_productos_sintetico.json"],
                                  output_csv="data/train_clean_rf.csv")
    validate_and_register_lineage(test_df, "Preparación de datos test RF",
                                  input_assets=[
                                      "reseñas_productos_sintetico.json"],
                                  output_csv="data/test_clean_rf.csv")

    with mlflow.start_run(run_name=f"random_forest_{datetime.now():%Y%m%d_%H%M%S}"):
        mlflow.log_param("model_type", "random_forest_bayesian")
        mlflow.log_param("n_classes", 3)

        memory = Memory('./cache_rf', verbose=0)
        pipe = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))),
            ('classifier', RandomForestClassifier(random_state=42))
        ], memory=memory)

        param_space = {
            'tfidf__max_features': Integer(50, 100),
            'tfidf__min_df': Integer(3, 5),
            'tfidf__max_df': Real(0.7, 1.0),
            'classifier__n_estimators': Integer(5, 50),
            'classifier__max_depth': Integer(10, 50),
            'classifier__min_samples_split': Integer(2, 20),
            'classifier__min_samples_leaf': Integer(1, 10),
            'classifier__max_features': Categorical(['sqrt', 'log2', None]),
            'classifier__class_weight': Categorical(['balanced', 'balanced_subsample', None])
        }

        print("Iniciando optimización bayesiana RF...")
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_callback = CustomLogCallback(
            f"logs/optimization_rf_{timestamp}.txt")
        checkpoint_callback = CheckpointSaver(
            f"logs/checkpoint_rf_{timestamp}.pkl", compress=6)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        bayes_search = BayesSearchCV(
            estimator=pipe,
            search_spaces=param_space,
            n_iter=15,
            scoring='f1_weighted',
            cv=skf,
            n_jobs=-1,
            random_state=42,
            verbose=0
        )

        bayes_search.fit(X_train, y_train, callback=[
                         checkpoint_callback, log_callback])

        # Calibración isotónica
        from sklearn.calibration import CalibratedClassifierCV
        calibrated_model = CalibratedClassifierCV(
            bayes_search.best_estimator_, method='isotonic', cv=3)
        calibrated_model.fit(X_train, y_train)

        y_pred = calibrated_model.predict(X_test)
        y_pred_proba = calibrated_model.predict_proba(X_test)

        # Log SHAP values
        feature_names = bayes_search.best_estimator_.named_steps['tfidf'].get_feature_names_out(
        )
        log_shap_values(
            bayes_search.best_estimator_,
            X_test,
            feature_names,
            model_name='random_forest',
            timestamp=timestamp
        )

        # Validar métricas obligatorias
        check_mandatory_metrics(y_test, y_pred, y_pred_proba)

        report = classification_report(y_test, y_pred, target_names=[
                                       'negativo', 'neutro', 'positivo'])
        print(report)
        plot_confusion_matrix(y_test, y_pred, 'Random Forest',
                              save_path=f'logs/confusion_matrix_rf_{timestamp}.png')

        model_path = 'models/random_forest_multiclass_calibrated.pkl'
        joblib.dump(calibrated_model, model_path)
        print(f"Modelo guardado en: {model_path}")

        return calibrated_model, bayes_search.best_params_


def train_xgboost_bayesian(train_df, test_df, max_features=5000):
    """Entrena XGBoost con optimización bayesiana y métricas obligatorias"""
    X_train = train_df['reseña_limpia']
    y_train = train_df['sentimiento_numerico']
    X_test = test_df['reseña_limpia']
    y_test = test_df['sentimiento_numerico']

    # Validación y linaje
    validate_and_register_lineage(train_df, "Preparación de datos entrenamiento XGB",
                                  input_assets=[
                                      "reseñas_productos_sintetico.json"],
                                  output_csv="data/train_clean_xgb.csv")
    validate_and_register_lineage(test_df, "Preparación de datos test XGB",
                                  input_assets=[
                                      "reseñas_productos_sintetico.json"],
                                  output_csv="data/test_clean_xgb.csv")

    with mlflow.start_run(run_name=f"xgboost_{datetime.now():%Y%m%d_%H%M%S}"):
        mlflow.log_param("model_type", "xgboost_bayesian")
        mlflow.log_param("n_classes", 3)

        memory = Memory('./cache_xgb', verbose=0)
        pipe = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))),
            ('classifier', XGBClassifier(objective='multi:softprob', num_class=3,
                                         eval_metric='mlogloss', random_state=42, use_label_encoder=False))
        ], memory=memory)

        param_space = {
            'tfidf__max_features': Integer(50, 100),
            'tfidf__min_df': Integer(3, 5),
            'tfidf__max_df': Real(0.7, 1.0),
            'classifier__n_estimators': Integer(5, 50),
            'classifier__max_depth': Integer(3, 6),
            'classifier__learning_rate': Real(0.001, 0.015, prior='log-uniform'),
            'classifier__subsample': Real(0.6, 1.0),
            'classifier__colsample_bytree': Real(0.6, 1.0),
            'classifier__reg_alpha': Real(0, 10),
            'classifier__reg_lambda': Real(1, 10),
            'classifier__gamma': Real(0, 5)
        }

        print("Iniciando optimización bayesiana XGB...")
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_callback = CustomLogCallback(
            f"logs/optimization_xgb_{timestamp}.txt")
        checkpoint_callback = CheckpointSaver(
            f"logs/checkpoint_xgb_{timestamp}.pkl", compress=6)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        bayes_search = BayesSearchCV(
            estimator=pipe,
            search_spaces=param_space,
            n_iter=15,
            scoring='f1_weighted',
            cv=skf,
            n_jobs=-1,
            random_state=42,
            verbose=0
        )

        bayes_search.fit(X_train, y_train, callback=[
                         checkpoint_callback, log_callback])

        from sklearn.calibration import CalibratedClassifierCV
        calibrated_model = CalibratedClassifierCV(
            bayes_search.best_estimator_, method='isotonic', cv=3)
        calibrated_model.fit(X_train, y_train)

        y_pred = calibrated_model.predict(X_test)
        y_pred_proba = calibrated_model.predict_proba(X_test)

        # Log SHAP values
        feature_names = bayes_search.best_estimator_.named_steps['tfidf'].get_feature_names_out(
        )
        log_shap_values(
            bayes_search.best_estimator_,
            X_test,
            feature_names,
            model_name='xgboost',
            timestamp=timestamp
        )

        # Validar métricas obligatorias
        check_mandatory_metrics(y_test, y_pred, y_pred_proba)

        report = classification_report(y_test, y_pred, target_names=[
                                       'negativo', 'neutro', 'positivo'])
        print(report)
        plot_confusion_matrix(y_test, y_pred, 'XGBoost',
                              save_path=f'logs/confusion_matrix_xgb_{timestamp}.png')

        model_path = 'models/xgboost_multiclass_calibrated.pkl'
        joblib.dump(calibrated_model, model_path)
        print(f"Modelo guardado en: {model_path}")

        return calibrated_model, bayes_search.best_params_


# ------------------------- COMPARACIÓN DE MODELOS -------------------------

def compare_models_bayesian():
    mlflow.set_experiment("product_sentiment_multiclass_bayesian")
    df = load_product_reviews()
    train_df, test_df = prepare_data(df, test_size=0.2)

    print("\n=== Entrenando Logistic Regression ===")
    lr_model, lr_params = train_logistic_regression_bayesian(train_df, test_df)

    print("\n=== Entrenando Random Forest ===")
    rf_model, rf_params = train_random_forest_bayesian(train_df, test_df)

    print("\n=== Entrenando XGBoost ===")
    xgb_model, xgb_params = train_xgboost_bayesian(train_df, test_df)

    print("\n✅ Todos los modelos entrenados y linaje registrado.")


# ------------------------- MAIN -------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Entrenar modelos con optimización bayesiana para clasificación multiclase')
    parser.add_argument('--model', type=str, choices=[
                        'logistic_regression', 'random_forest', 'xgboost', 'compare'], default='compare')
    parser.add_argument('--data-path', type=str,
                        default='data/reseñas_productos_sintetico.json')
    parser.add_argument('--sample-size', type=int, default=None)
    args = parser.parse_args()

    df = load_product_reviews(data_path=args.data_path,
                              sample_size=args.sample_size)
    train_df, test_df = prepare_data(df, test_size=0.2)

    if args.model == 'compare':
        compare_models_bayesian()
    elif args.model == 'logistic_regression':
        train_logistic_regression_bayesian(train_df, test_df)
    elif args.model == 'random_forest':
        train_random_forest_bayesian(train_df, test_df)
    elif args.model == 'xgboost':
        train_xgboost_bayesian(train_df, test_df)
