"""Validate uploads, normalize audio, and collect every Speech SDK result."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
import wave

from errors import AppError

MAX_UPLOAD_BYTES = 25 * 1024 * 1024
SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".aac", ".webm", ".ogg"}
SPEECH_LOCK = threading.Lock()
TICKS_PER_SECOND = 10_000_000


def wav_duration(path: Path) -> float:
    try:
        with wave.open(str(path), "rb") as audio:
            if (audio.getnchannels(), audio.getsampwidth(), audio.getframerate()) != (1, 2, 16000):
                raise AppError(415, "Use a 16 kHz, 16-bit mono PCM WAV file, or install ffmpeg to convert audio.")
            duration = audio.getnframes() / audio.getframerate()
    except (wave.Error, EOFError, OSError):
        raise AppError(415, "The file could not be read as PCM WAV audio.") from None
    if duration <= 0:
        raise AppError(422, "The audio file contains no audio frames.")
    return duration


def normalize_audio(source: Path, destination: Path) -> Path:
    executable = shutil.which("ffmpeg")
    if not executable:
        if source.suffix.lower() != ".wav":
            raise AppError(415, "Install ffmpeg to process MP3, M4A, WebM, OGG, or AAC files. On a Mac with Homebrew: brew install ffmpeg")
        wav_duration(source)
        return source
    try:
        process = subprocess.run(
            [executable, "-nostdin", "-v", "error", "-y", "-protocol_whitelist", "file,pipe",
             "-i", str(source), "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(destination)],
            capture_output=True, timeout=90, check=False,
        )
    except subprocess.TimeoutExpired:
        raise AppError(415, "Audio conversion timed out. Try a shorter lecture recording.") from None
    if process.returncode != 0:
        raise AppError(415, "This file is not a supported, readable audio recording.")
    return destination


def parse_segment(payload: dict, text: str) -> dict:
    """Azure's offsets/durations are ticks; JSON exposed by this app uses seconds."""
    choices = payload.get("NBest") or []
    best = choices[0] if choices else {}
    words = []
    for word in best.get("Words", []):
        words.append({
            "word": word.get("Word", ""),
            "offset": round(word.get("Offset", 0) / TICKS_PER_SECOND, 3),
            "duration": round(word.get("Duration", 0) / TICKS_PER_SECOND, 3),
            "confidence": word.get("Confidence"),
        })
    return {
        "text": text,
        "offset": round(payload.get("Offset", 0) / TICKS_PER_SECOND, 3),
        "duration": round(payload.get("Duration", 0) / TICKS_PER_SECOND, 3),
        "confidence": best.get("Confidence"),
        "words": words,
    }


def combine_segments(segments: list[dict], duration: float) -> dict:
    if not segments:
        raise AppError(422, "No speech was recognized. Try a clearer recording with spoken lecture content.")
    weighted = [(s["confidence"], max(len(s["words"]), len(s["text"].split()), 1))
                for s in segments if s["confidence"] is not None]
    confidence = (sum(score * weight for score, weight in weighted) / sum(weight for _, weight in weighted)) if weighted else None
    return {
        "transcript": " ".join(s["text"] for s in segments),
        "recognition_locale": "en-US",
        "duration_seconds": round(duration, 3),
        "confidence": round(confidence, 4) if confidence is not None else None,
        "confidence_method": "word_count_weighted_utterance_confidence",
        "words": [word for s in segments for word in s["words"]],
        "segments": segments,
    }


def recognize_file(path: Path, duration: float) -> dict:
    # Import lazily so health checks and offline tests need no live Azure session.
    import azure.cognitiveservices.speech as speechsdk

    if not SPEECH_LOCK.acquire(blocking=False):
        raise AppError(429, "A recording is already being transcribed. Wait for it to finish and try again.")
    recognizer = None
    segments, failures, no_matches = [], [], []
    final_events = []
    finished = threading.Event()
    try:
        config = speechsdk.SpeechConfig(
            subscription=os.environ["AZURE_SPEECH_KEY"], region=os.environ["AZURE_SPEECH_REGION"]
        )
        config.speech_recognition_language = "en-US"
        config.output_format = speechsdk.OutputFormat.Detailed
        config.request_word_level_timestamps()
        audio = speechsdk.audio.AudioConfig(filename=str(path))
        recognizer = speechsdk.SpeechRecognizer(speech_config=config, audio_config=audio)

        def recognized(event):
            try:
                result = event.result
                final_events.append(str(result.reason))
                if result.reason == speechsdk.ResultReason.RecognizedSpeech:
                    # Use the SDK's documented JSON property directly. Failures in
                    # a native callback must not be mistaken for an unclear recording.
                    payload = json.loads(result.json or "{}")
                    if not result.text.strip():
                        failures.append("empty_recognized_text")
                        finished.set()
                        return
                    segments.append(parse_segment(payload, result.text))
                elif result.reason == speechsdk.ResultReason.NoMatch:
                    details = result.no_match_details
                    no_matches.append(str(details.reason) if details else "unspecified")
            except Exception as error:
                # Include only the exception class, never raw service messages or secrets.
                failures.append("callback_" + type(error).__name__)
                finished.set()

        def canceled(event):
            try:
                # Python exposes these on cancellation_details, not on the event.
                details = event.cancellation_details
                if details.reason == speechsdk.CancellationReason.Error:
                    failures.append("service_error_" + str(details.code))
            except Exception as error:
                failures.append("cancellation_callback_" + type(error).__name__)
            finally:
                finished.set()

        recognizer.recognized.connect(recognized)
        recognizer.canceled.connect(canceled)
        recognizer.session_stopped.connect(lambda event: finished.set())
        recognizer.start_continuous_recognition_async().get()
        # Wait for the session to end, not for the first recognized sentence.
        if not finished.wait(timeout=max(90, duration * 2 + 60)):
            raise AppError(504, "Speech transcription timed out. Try a shorter recording.")
        if failures:
            raise AppError(502, "Speech transcription failed. Diagnostic code(s): " + ", ".join(failures) + ". Check the Speech configuration or report these diagnostic codes.")
        if not segments:
            reasons = ", ".join(dict.fromkeys(no_matches)) or "none reported"
            raise AppError(422, "Azure Speech returned no recognized text. Final recognition events: "
                           + str(len(final_events)) + "; no-match reasons: " + reasons
                           + ". Audio duration: " + str(round(duration, 2))
                           + " seconds. This does not by itself establish that the recording is unclear.")
        return combine_segments(segments, duration)
    except AppError:
        raise
    except (RuntimeError, ValueError, OSError):
        raise AppError(502, "Azure Speech could not start. Check the Speech SDK installation, key, region, and network connection.") from None
    finally:
        try:
            if recognizer is not None:
                try:
                    recognizer.stop_continuous_recognition_async().get()
                except RuntimeError:
                    pass
        finally:
            SPEECH_LOCK.release()


def transcribe_upload(stream, filename: str | None) -> dict:
    suffix = Path(filename or "").suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise AppError(415, "Upload a WAV, MP3, M4A, AAC, WebM, or OGG audio file.")
    # Every exit path, including an Azure error, removes both upload and converted WAV.
    with tempfile.TemporaryDirectory(prefix="lecture-notes-") as directory:
        source = Path(directory) / ("upload" + suffix)
        size = 0
        with source.open("wb") as target:
            while block := stream.read(1024 * 1024):
                size += len(block)
                if size > MAX_UPLOAD_BYTES:
                    raise AppError(413, "The audio file exceeds the 25 MB limit.")
                target.write(block)
        if size == 0:
            raise AppError(422, "The uploaded file is empty.")
        normalized = normalize_audio(source, Path(directory) / "normalized.wav")
        return recognize_file(normalized, wav_duration(normalized))
