"""Lecture Notes Assistant: localhost checkpoint backend (Parts A-C)."""

import os
from pathlib import Path
import shutil
from time import perf_counter

from dotenv import load_dotenv

load_dotenv(Path(__file__).with_name(".env"))

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from audio_pipeline import transcribe_upload
from errors import AppError
from language_service import analyze_text, detect_transcript_language
from study_material import make_study_material

app = FastAPI(title="Lecture Notes Assistant", version="0.1.0",
              description="Upload an English lecture segment for continuous transcription and Azure Language analysis.")

SETTINGS = ["AZURE_SPEECH_KEY", "AZURE_SPEECH_REGION", "AZURE_LANGUAGE_KEY",
            "AZURE_LANGUAGE_ENDPOINT", "APPLICATIONINSIGHTS_CONNECTION_STRING"]


def require_settings(names):
    missing = [name for name in names if not os.getenv(name, "").strip()
               or os.getenv(name, "").startswith(("replace_with_", "your-"))]
    if missing:
        raise AppError(503, "Add these settings to your local .env file: " + ", ".join(missing))


@app.exception_handler(AppError)
async def application_error(request, error):
    headers = {"Retry-After": "10"} if error.status_code in (429, 503) else None
    return JSONResponse(status_code=error.status_code, content={"error": error.message}, headers=headers)


@app.get("/")
def home():
    return {"product": "Lecture Notes Assistant", "api_docs": "/docs", "main_endpoint": "POST /lecture-notes"}


@app.get("/health")
def health():
    # Reports presence only: never returns credential values. This is not an Azure connectivity test.
    configured = {name: bool(os.getenv(name, "").strip()) for name in SETTINGS}
    return {"status": "ok", "configuration_present": configured, "ffmpeg_available": shutil.which("ffmpeg") is not None}


class TextInput(BaseModel):
    text: str = Field(min_length=1, max_length=40_000)

    @field_validator("text")
    @classmethod
    def meaningful_text(cls, value):
        if not value.strip():
            raise ValueError("Provide nonempty lecture text.")
        return value.strip()


@app.post("/analyze")
def analyze(body: TextInput):
    require_settings(["AZURE_LANGUAGE_KEY", "AZURE_LANGUAGE_ENDPOINT"])
    return analyze_text(body.text)


@app.post("/transcribe")
def transcribe(audio: UploadFile = File(...)):
    require_settings(SETTINGS[:4])
    result = transcribe_upload(audio.file, audio.filename)
    result["language"] = detect_transcript_language(result["transcript"])
    return result


@app.post("/lecture-notes")
def lecture_notes(audio: UploadFile = File(...)):
    require_settings(SETTINGS[:4])
    start = perf_counter()
    transcription = transcribe_upload(audio.file, audio.filename)
    stt_end = perf_counter()
    if len(transcription["transcript"]) > 40_000:
        raise AppError(422, "For this prototype, use a lecture segment with at most 40,000 transcript characters.")
    analysis = analyze_text(transcription["transcript"])
    analysis_end = perf_counter()
    transcription["language"] = analysis["language"]
    study_guide = make_study_material(transcription["transcript"], analysis["key_terms"])
    duration = transcription["duration_seconds"]
    return {"product": "Lecture Notes Assistant", "transcription": transcription,
            "analysis": analysis, "study_guide": study_guide,
            "metrics": {"key_terms_per_minute": round(len(analysis["key_terms"]) * 60 / duration, 2) if duration else None,
                        "stt_ms": round((stt_end - start) * 1000, 1),
                        "language_ms": round((analysis_end - stt_end) * 1000, 1),
                        "total_ms": round((perf_counter() - start) * 1000, 1)}}
