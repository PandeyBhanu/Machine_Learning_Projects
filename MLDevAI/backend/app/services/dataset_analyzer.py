# app/services/dataset_analyzer.py
import pandas as pd
from io import BytesIO


def analyze_dataset(file_bytes: bytes, file_ext: str) -> dict:
    if file_ext == ".csv":
        df = pd.read_csv(BytesIO(file_bytes))
    elif file_ext in [".xls", ".xlsx"]:
        df = pd.read_excel(BytesIO(file_bytes))
    else:
        raise ValueError("Unsupported format")

    info = {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "dtypes": df.dtypes.apply(str).to_dict(),
        "nulls": df.isnull().sum().to_dict(),
        "preview": df.head().to_dict(orient="records"),
    }
    return info
