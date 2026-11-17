"""Offline checks for callback handling, upload cleanup, and endpoint contracts.

Run: python -m unittest discover -s tests -v
No Azure credentials or network access are used.
"""

import io
import json
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import MagicMock, patch
import wave

from fastapi.testclient import TestClient

import audio_pipeline
from audio_pipeline import combine_segments, parse_segment, transcribe_upload
from errors import AppError
from language_service import chunk_text
from main import app, SETTINGS
from study_material import make_study_material


def wav_bytes():
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as recording:
        recording.setnchannels(1)
        recording.setsampwidth(2)
        recording.setframerate(16000)
        recording.writeframes(b"\x00\x00" * 16000)
    buffer.seek(0)
    return buffer


class SpeechTests(unittest.TestCase):
    def test_confidence_is_not_invented_for_words(self):
        segment = parse_segment({"Offset": 20_000_000, "Duration": 10_000_000,
            "NBest": [{"Confidence": .9, "Words": [{"Word": "test", "Offset": 20_000_000, "Duration": 5_000_000}]}]}, "Test.")
        self.assertEqual(segment["words"][0]["offset"], 2)
        self.assertIsNone(segment["words"][0]["confidence"])

    def test_multiple_utterances_are_preserved(self):
        segments = [parse_segment({"NBest": [{"Confidence": .9}]}, "First sentence."),
                    parse_segment({"NBest": [{"Confidence": .7}]}, "Second sentence after pause.")]
        result = combine_segments(segments, 40)
        self.assertEqual(result["transcript"], "First sentence. Second sentence after pause.")
        self.assertEqual(len(result["segments"]), 2)
        self.assertGreater(result["confidence"], .7)

    def test_silence_does_not_produce_fake_notes(self):
        with self.assertRaises(AppError) as raised:
            combine_segments([], 1)
        self.assertEqual(raised.exception.status_code, 422)

    def test_continuous_recognizer_waits_for_all_events(self):
        class Signal:
            def connect(self, callback):
                self.callback = callback
        recognizer = types.SimpleNamespace(recognized=Signal(), canceled=Signal(), session_stopped=Signal())
        sdk = types.ModuleType("azure.cognitiveservices.speech")
        sdk.SpeechConfig = MagicMock()
        sdk.audio = types.SimpleNamespace(AudioConfig=MagicMock())
        sdk.SpeechRecognizer = MagicMock(return_value=recognizer)
        sdk.OutputFormat = types.SimpleNamespace(Detailed="detailed")
        sdk.ResultReason = types.SimpleNamespace(RecognizedSpeech="recognized", NoMatch="no_match")
        sdk.CancellationReason = types.SimpleNamespace(Error="error")
        sdk.PropertyId = types.SimpleNamespace(SpeechServiceResponse_JsonResult="json")

        def deliver_all():
            for text in ("Sentence one.", "Sentence two after a pause.", "Final sentence."):
                payload = json.dumps({"NBest": [{"Confidence": .91}]})
                event = types.SimpleNamespace(result=types.SimpleNamespace(reason="recognized", text=text, json=payload))
                recognizer.recognized.callback(event)
            recognizer.session_stopped.callback(None)
        future = types.SimpleNamespace(get=deliver_all)
        recognizer.start_continuous_recognition_async = MagicMock(return_value=future)
        recognizer.stop_continuous_recognition_async = MagicMock(return_value=types.SimpleNamespace(get=lambda: None))
        azure = types.ModuleType("azure")
        azure.__path__ = []
        cognitive = types.ModuleType("azure.cognitiveservices")
        cognitive.__path__ = []
        azure.cognitiveservices = cognitive
        cognitive.speech = sdk
        with patch.dict("sys.modules", {"azure": azure, "azure.cognitiveservices": cognitive, "azure.cognitiveservices.speech": sdk}), \
             patch.dict(os.environ, {"AZURE_SPEECH_KEY": "offline-test", "AZURE_SPEECH_REGION": "northcentralus"}):
            result = audio_pipeline.recognize_file(Path("offline.wav"), 40)
        self.assertEqual(len(result["segments"]), 3)
        self.assertTrue(result["transcript"].endswith("Final sentence."))
        recognizer.stop_continuous_recognition_async.assert_called_once()
        self.assertFalse(audio_pipeline.SPEECH_LOCK.locked())

    def run_diagnostic_scenario(self, event, cancel=False):
        class Signal:
            def connect(self, callback):
                self.callback = callback
        recognizer = types.SimpleNamespace(recognized=Signal(), canceled=Signal(), session_stopped=Signal())
        sdk = types.ModuleType("azure.cognitiveservices.speech")
        sdk.SpeechConfig = MagicMock()
        sdk.audio = types.SimpleNamespace(AudioConfig=MagicMock())
        sdk.SpeechRecognizer = MagicMock(return_value=recognizer)
        sdk.OutputFormat = types.SimpleNamespace(Detailed="detailed")
        sdk.ResultReason = types.SimpleNamespace(RecognizedSpeech="recognized", NoMatch="no_match")
        sdk.CancellationReason = types.SimpleNamespace(Error="error")
        def deliver():
            if cancel:
                recognizer.canceled.callback(event)
            else:
                recognizer.recognized.callback(event)
            recognizer.session_stopped.callback(None)
        recognizer.start_continuous_recognition_async = MagicMock(return_value=types.SimpleNamespace(get=deliver))
        recognizer.stop_continuous_recognition_async = MagicMock(return_value=types.SimpleNamespace(get=lambda: None))
        azure = types.ModuleType("azure")
        azure.__path__ = []
        cognitive = types.ModuleType("azure.cognitiveservices")
        cognitive.__path__ = []
        azure.cognitiveservices = cognitive
        cognitive.speech = sdk
        with patch.dict("sys.modules", {"azure": azure, "azure.cognitiveservices": cognitive, "azure.cognitiveservices.speech": sdk}), \
             patch.dict(os.environ, {"AZURE_SPEECH_KEY": "offline-test", "AZURE_SPEECH_REGION": "northcentralus"}):
            return audio_pipeline.recognize_file(Path("offline.wav"), 51)

    def test_no_match_reason_is_preserved(self):
        event = types.SimpleNamespace(result=types.SimpleNamespace(reason="no_match", no_match_details=types.SimpleNamespace(reason="InitialSilenceTimeout")))
        with self.assertRaises(AppError) as raised:
            self.run_diagnostic_scenario(event)
        self.assertEqual(raised.exception.status_code, 422)
        self.assertIn("InitialSilenceTimeout", raised.exception.message)
        self.assertFalse(audio_pipeline.SPEECH_LOCK.locked())

    def test_callback_error_is_not_reported_as_unclear_audio(self):
        event = types.SimpleNamespace(result=types.SimpleNamespace(reason="recognized", text="Real speech", json="invalid json"))
        with self.assertRaises(AppError) as raised:
            self.run_diagnostic_scenario(event)
        self.assertEqual(raised.exception.status_code, 502)
        self.assertIn("callback_JSONDecodeError", raised.exception.message)
        self.assertNotIn("Try a clearer recording", raised.exception.message)

    def test_missing_sdk_attribute_is_reported_instead_of_ignored(self):
        event = types.SimpleNamespace(result=types.SimpleNamespace(reason="recognized", text="Real speech"))
        with self.assertRaises(AppError) as raised:
            self.run_diagnostic_scenario(event)
        self.assertEqual(raised.exception.status_code, 502)
        self.assertIn("callback_AttributeError", raised.exception.message)

    def test_cancellation_uses_documented_python_details_object(self):
        # Deliberately no event.reason or event.error_code: the real Python SDK
        # puts reason and code on event.cancellation_details.
        event = types.SimpleNamespace(cancellation_details=types.SimpleNamespace(
            reason="error", code="AuthenticationFailure", error_details="private service diagnostic"))
        with self.assertRaises(AppError) as raised:
            self.run_diagnostic_scenario(event, cancel=True)
        self.assertEqual(raised.exception.status_code, 502)
        self.assertIn("service_error_AuthenticationFailure", raised.exception.message)
        self.assertNotIn("private service diagnostic", raised.exception.message)
        self.assertNotIn("callback_AttributeError", raised.exception.message)
        self.assertFalse(audio_pipeline.SPEECH_LOCK.locked())

    def test_normal_end_of_stream_is_not_a_service_error(self):
        event = types.SimpleNamespace(cancellation_details=types.SimpleNamespace(reason="EndOfStream", code="NoError"))
        with self.assertRaises(AppError) as raised:
            self.run_diagnostic_scenario(event, cancel=True)
        self.assertEqual(raised.exception.status_code, 422)
        self.assertNotIn("service_error", raised.exception.message)


class UploadTests(unittest.TestCase):
    def test_temporary_files_removed_on_azure_failure(self):
        visited = []
        def fail(path, duration):
            visited.append(path.parent)
            self.assertTrue(path.exists())
            raise AppError(502, "Offline simulated Azure failure")
        with patch("audio_pipeline.shutil.which", return_value=None), patch("audio_pipeline.recognize_file", side_effect=fail):
            with self.assertRaises(AppError):
                transcribe_upload(wav_bytes(), "lecture.wav")
        self.assertFalse(visited[0].exists())

    def test_oversized_upload_rejected(self):
        with patch("audio_pipeline.MAX_UPLOAD_BYTES", 10):
            with self.assertRaises(AppError) as raised:
                transcribe_upload(io.BytesIO(b"x" * 11), "lecture.wav")
        self.assertEqual(raised.exception.status_code, 413)

    def test_unsupported_extension(self):
        with self.assertRaises(AppError) as raised:
            transcribe_upload(io.BytesIO(b"text"), "notes.txt")
        self.assertEqual(raised.exception.status_code, 415)

    def test_unreadable_wav_is_415(self):
        with patch("audio_pipeline.shutil.which", return_value=None):
            with self.assertRaises(AppError) as raised:
                transcribe_upload(io.BytesIO(b"not audio"), "bad.wav")
        self.assertEqual(raised.exception.status_code, 415)


class StudyTests(unittest.TestCase):
    def test_quiz_answers_come_from_recording(self):
        text = "Stratified sampling divides a population into groups. Random sampling gives units a chance of selection. Sampling weights account for selection probabilities."
        result = make_study_material(text, ["Stratified sampling", "Random sampling", "Sampling weights"])
        self.assertEqual(len(result["quiz"]), 3)
        for question in result["quiz"]:
            self.assertIn(question["answer"].lower(), question["source_excerpt"].lower())
            self.assertIn("_____", question["question"])

    def test_insufficient_source_is_reported(self):
        result = make_study_material("A short sentence.", [])
        self.assertEqual(result["quiz"], [])
        self.assertNotEqual(result["quiz_status"], "ready")

    def test_chunks_preserve_words_and_stay_within_limit(self):
        text = " ".join(["sampling"] * 3000)
        chunks = chunk_text(text)
        self.assertTrue(all(len(chunk) <= 4000 for chunk in chunks))
        self.assertEqual(" ".join(chunks), text)


class EndpointTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_health_never_returns_credentials(self):
        with patch.dict(os.environ, {name: "a-private-value" for name in SETTINGS}):
            response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertNotIn("a-private-value", response.text)

    def test_whitespace_text_rejected(self):
        self.assertEqual(self.client.post("/analyze", json={"text": "   "}).status_code, 422)

    def test_missing_settings_have_clear_error(self):
        with patch.dict(os.environ, {}, clear=True):
            response = self.client.post("/analyze", json={"text": "A lecture about sampling."})
        self.assertEqual(response.status_code, 503)
        self.assertIn("AZURE_LANGUAGE_KEY", response.json()["error"])

    def test_pipeline_returns_product_json_and_domain_metric(self):
        transcription = {"transcript": "Stratified sampling divides populations into groups.", "duration_seconds": 30, "segments": [], "words": [], "confidence": .9}
        analysis = {"language": {"code": "en"}, "key_terms": ["Stratified sampling"], "entities": [], "sentiment": {"label": "neutral"}}
        with patch.dict(os.environ, {name: "offline-test" for name in SETTINGS}), \
             patch("main.transcribe_upload", return_value=transcription), patch("main.analyze_text", return_value=analysis):
            response = self.client.post("/lecture-notes", files={"audio": ("lecture.wav", b"fake audio", "audio/wav")})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["metrics"]["key_terms_per_minute"], 2)
        self.assertIn("study_guide", response.json())


if __name__ == "__main__":
    unittest.main()
