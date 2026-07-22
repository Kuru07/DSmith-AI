import os

from dotenv import load_dotenv
from fastapi import FastAPI
# pyrefly: ignore [missing-import]
from pydantic import BaseModel

load_dotenv()

app=FastAPI(
    title="DSmith API",
    description="Autonomous Data Science Agent API",
    version="1.0.0"
)

class AnalyzeRequest(BaseModel):
    filename: str

@app.get("/")
def root():
    return {"message": "Data Science Agent is Running"}


@app.get("/health")
def health():
    api_key_available = bool(os.getenv("GEMINI_API_KEY"))
    return {
        "status": "Healthy",
        "gemini_configured": api_key_available
    }

@app.post("/analyze")
def analyze(request: AnalyzeRequest):
    return {
        "message": f"Received Dataset, {request.filename}!"
    }

