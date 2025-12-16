from flask import Blueprint, request, render_template, send_file
import pandas as pd
from pathlib import Path
from app.services import ollama_client
import os
import zipfile
from datetime import datetime
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import uuid

main = Blueprint("main", __name__)

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_PATH = BASE_DIR / "output" / "uploaded.csv"
GENERATED_DIR = BASE_DIR / "generated_projects"


@main.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@main.route("/analyze", methods=["POST"])
def analyze():
    file = request.files.get("dataset")
    target_column = request.form.get("target")  # ✅ Matches the HTML

    if not file:
        return render_template(
            "index.html", dataset_info={"error": "No file uploaded."}
        )

    try:
        df = pd.read_csv(file)
        UPLOAD_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(UPLOAD_PATH, index=False)

        PLOTS_DIR = BASE_DIR / "static" / "plots"
        PLOTS_DIR.mkdir(parents=True, exist_ok=True)

        unique_id = uuid.uuid4().hex[:8]  # Unique suffix to avoid caching issues

        # Histogram
        plt.figure(figsize=(15, 10))
        numeric_cols = df.select_dtypes(include="number").columns
        df[numeric_cols].hist(bins=20, edgecolor="black")
        plt.tight_layout()
        hist_file = f"hist_{unique_id}.png"
        plt.savefig(PLOTS_DIR / hist_file)
        plt.close()

        # Correlation heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
        plt.title("Correlation Heatmap")
        corr_file = f"corr_{unique_id}.png"
        plt.savefig(PLOTS_DIR / corr_file)
        plt.close()

        # Missing values
        plt.figure(figsize=(10, 6))
        sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
        plt.title("Missing Values")
        missing_file = f"missing_{unique_id}.png"
        plt.savefig(PLOTS_DIR / missing_file)
        plt.close()

        # Scatter plots
        scatter_files = []
        if target_column and target_column in df.columns:
            for col in numeric_cols:
                if col == target_column:
                    continue
                plt.figure(figsize=(6, 4))
                sns.scatterplot(data=df, x=col, y=target_column)
                plt.title(f"{col} vs {target_column}")
                scatter_file = f"scatter_{col}_{unique_id}.png"
                plt.savefig(PLOTS_DIR / scatter_file)
                plt.close()
                scatter_files.append(scatter_file)

        dataset_info = {
            "shape": df.shape,
            "columns": df.columns.tolist(),
            "dtypes": df.dtypes.to_string(),
            "head": df.head(3).to_string(index=False),
            "missing_values": df.isnull().sum().to_string(),
            "describe": df.describe().to_string(),
            "target": target_column,
            "plots": {
                "hist": hist_file,
                "corr": corr_file,
                "missing": missing_file,
                "scatter": scatter_files,
            },
        }

        return render_template("index.html", dataset_info=dataset_info)

    except Exception as e:
        return render_template(
            "index.html", dataset_info={"error": f"Error processing dataset: {e}"}
        )


@main.route("/generate-code", methods=["POST"])
def generate_code():
    prompt = request.form.get("prompt", "").strip()
    if not UPLOAD_PATH.exists():
        return render_template(
            "index.html",
            generated_code="⚠️ No dataset analyzed yet. Please upload and analyze a dataset first.",
        )

    try:
        df = pd.read_csv(UPLOAD_PATH)
    except Exception as e:
        return render_template(
            "index.html", generated_code=f"⚠️ Failed to read cached dataset: {e}"
        )

    shape = df.shape
    columns = df.columns.tolist()
    preview = df.head(3).to_dict(orient="records")

    full_prompt = f"""
You are a machine learning engineer.

You are working with the following dataset:
- Shape: {shape}
- Columns: {columns}
- Sample data (first 3 rows): {preview}

Your task:
Generate a clean, production-ready Python script that:
- Uses pandas to load the data.
- Preprocesses features (use MinMaxScaler if needed).
- Splits the data into train/test.
- Trains a scikit-learn model (like LinearRegression).
- Evaluates the model using RMSE and R² Score.
- Prints all relevant evaluation metrics.

Constraints:
- Do NOT explain anything.
- Do NOT include markdown formatting (e.g., no ```python).
- Only return raw Python code, no text or comments unless inline with code.
"""

    try:
        code = ollama_client.run_llm(full_prompt)

        # Strip markdown formatting if present
        if code.startswith("```"):
            code = code.strip("```python").strip("```").strip()

        return render_template("index.html", generated_code=code)

    except Exception as e:
        return render_template("index.html", generated_code=f"⚠️ LLM error: {e}")


@main.route("/scaffold", methods=["POST"])
def scaffold():
    task = request.form.get("task", "regression")
    model_name = request.form.get("model", "LinearRegression")
    dataset_name = request.form.get("dataset_name", "dataset.csv")

    if not UPLOAD_PATH.exists():
        return "⚠️ No dataset uploaded or analyzed yet.", 400

    try:
        df = pd.read_csv(UPLOAD_PATH)
    except Exception as e:
        return f"⚠️ Failed to read uploaded dataset: {e}", 500

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    project_name = f"{task}_project_{timestamp}"
    project_path = GENERATED_DIR / project_name
    data_path = project_path / "data"
    data_path.mkdir(parents=True, exist_ok=True)

    dataset_output_path = data_path / dataset_name
    df.to_csv(dataset_output_path, index=False)

    (project_path / "main.py").write_text(
        f"""import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import {model_name}
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

df = pd.read_csv('data/{dataset_name}')
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = {model_name}()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R² Score:", r2_score(y_test, y_pred))
"""
    )

    (project_path / "requirements.txt").write_text("pandas\nscikit-learn\nnumpy\n")
    (project_path / "README.md").write_text(
        f"# {task.title()} Project\n\nGenerated using MLDev AI.\n"
    )

    zip_path = GENERATED_DIR / f"{project_name}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(project_path):
            for file in files:
                file_path = Path(root) / file
                arcname = file_path.relative_to(GENERATED_DIR)
                zipf.write(file_path, arcname)

    if not zip_path.exists():
        return "❌ ZIP generation failed.", 500

    return send_file(zip_path, as_attachment=True)
