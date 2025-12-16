# app/main.py

from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from pathlib import Path
from app.services import dataset_analyzer, ollama_client, scaffolder

app = FastAPI()


class PromptRequest(BaseModel):
    prompt: str


@app.post("/analyze/")
async def analyze(file: UploadFile = File(...)):
    content = await file.read()
    info = dataset_analyzer.analyze_dataset(content, Path(file.filename).suffix)
    return info


@app.post("/generate-code/")
async def generate_code(data: PromptRequest):
    return {"code": ollama_client.run_llm(data.prompt)}


@app.post("/scaffold")
async def scaffold(context: dict = Body(...)):
    zip_path = scaffold_project(context)
    return FileResponse(zip_path, filename="ml_project.zip")
