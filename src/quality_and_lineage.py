import pandas as pd
import json
from typing import List, Dict
from pathlib import Path

# ------------------- VALIDACIÓN DE CALIDAD DE DATOS -------------------

def validate_data_quality(df: pd.DataFrame, required_columns: List[str], quality_threshold: float = 0.95):
    """
    Valida la calidad de un DataFrame comprobando columnas, valores nulos y duplicados.
    Devuelve un reporte de calidad como diccionario.

    Args:
        df (pd.DataFrame): El DataFrame a validar.
        required_columns (List[str]): Columnas que deben estar presentes.
        quality_threshold (float): Umbral mínimo de valores no nulos.

    Returns:
        Dict: Reporte de calidad por columna.
    """
    report = {}
    print("\n[Data Quality] Iniciando validación...")

    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Columna requerida '{col}' no encontrada.")
        
        non_null_percentage = df[col].notna().mean()
        empty_strings_percentage = (df[col].astype(str).str.strip() == "").mean()
        duplicates_percentage = df.duplicated(subset=[col]).mean()
        
        report[col] = {
            "non_null_pct": non_null_percentage,
            "empty_strings_pct": empty_strings_percentage,
            "duplicates_pct": duplicates_percentage,
            "passed": (non_null_percentage >= quality_threshold and
                       empty_strings_percentage == 0 and
                       duplicates_percentage == 0)
        }

        if not report[col]["passed"]:
            print(f" *** Problema en columna '{col}': {report[col]}")

    print("[Data Quality] Validación completada.")
    return report

# ------------------- SEGUIMIENTO DE LINAJE (REGISTRO REAL) -------------------

class LineageTracker:
    """
    Tracker de linaje real registrando transformaciones en un archivo JSON.
    """
    def __init__(self, output_file: str = "lineage_log.json"):
        self.output_file = Path(output_file)
        if not self.output_file.exists():
            self.output_file.write_text(json.dumps([]))
        print(f"[Lineage] Tracker inicializado, registro en '{self.output_file}'")

    def register_lineage(self, process_name: str, input_assets: List[str], output_assets: List[str], params: Dict = None):
        """
        Registra una transformación de linaje en un JSON.
        """
        log_entry = {
            "process_name": process_name,
            "input_assets": input_assets,
            "output_assets": output_assets,
            "params": params or {},
        }

        data = json.loads(self.output_file.read_text())
        data.append(log_entry)
        self.output_file.write_text(json.dumps(data, indent=2))
        print(f"[Lineage] Transformación registrada: {process_name}")

# Instancia compartida
lineage_tracker = LineageTracker()
