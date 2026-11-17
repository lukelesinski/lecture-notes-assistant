# Lecture Notes Assistant - localhost checkpoint

Python/FastAPI starter for CS 345 Parts A-C. It accepts an English lecture recording,
uses Azure Speech **continuous recognition**, and calls Azure Language for detected
transcript language, key phrases, named entities, and sentiment. The product endpoint
also builds an initial study guide from source sentences.

## Keep your existing local settings

Copy the Python files into your existing `lecture-notes-assistant` folder alongside
your `.env`. This archive does **not** contain a `.env` or any real credentials.
Do not replace your filled-in `.env` with `.env.example`.

From a VS Code terminal opened in that folder:

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

If you have not created a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Check `http://127.0.0.1:8000/health`, then open `http://127.0.0.1:8000/docs`.
The health endpoint checks setting presence only; it does not authenticate with Azure.

## Configuration

Use the five variables listed in `.env.example`. Speech is provisioned in
`northcentralus`. Copy the Language endpoint exactly from its resource's
**Keys and Endpoint** page. Keep the entire Application Insights connection string
on one line, including `InstrumentationKey=` and all other fields.

Never upload `.env`, keys, or connection strings to GitHub or AI tools.

## Test Language first

In `/docs`, expand **POST /analyze**, choose **Try it out**, paste the following
body, and click **Execute**:

```json
{
  "text": "Today we are studying stratified sampling. A population is divided into groups called strata, and a random sample is taken from each group. Sampling weights account for different probabilities of selection. At Wake Forest University, a researcher could sample students from each class year."
}
```

This makes real Azure Language calls. Confirm an HTTP 200 response with `key_terms`,
`entities`, `sentiment`, and detected `language` before testing audio.

## Audio formats

For MP3, M4A, WebM/Opus, OGG, AAC, or WAV in other sample rates, install the **ffmpeg
system program**, not a similarly named Python package. On a Mac with Homebrew:

```bash
brew install ffmpeg
```

Without ffmpeg, only **16 kHz, 16-bit mono PCM WAV** works. `/health` reports whether
ffmpeg is on PATH. If Homebrew is unavailable, use a WAV with those exact settings
for the first checkpoint test. Encoded or unreadable files return HTTP 415.

## Endpoints

| Endpoint | Request | Response |
| --- | --- | --- |
| GET `/health` | None | Configuration-presence flags; no secret values |
| POST `/analyze` | JSON with `text` (1-40,000 characters) | Azure Language results |
| POST `/transcribe` | Multipart file field named `audio` | Transcript, detected transcript language, recognition locale, audio duration, available confidences, words, utterance segments |
| POST `/lecture-notes` | Multipart file field named `audio` | Transcription, Language results, initial study guide, domain metric, local stage timings |

The Speech recognizer is configured for **en-US**. The separate `language` result
comes from Azure Language detection on the transcript; it is not automatic audio
language identification. This prototype does not promise multilingual recognition.
Utterance confidences are word-count weighted for the overall score. When Azure
does not supply a word confidence, its JSON value is `null`, not a fabricated score.

Uploads are limited to 25 MB. Temporary uploads and converted WAV files are deleted
after every request, including errors. The app does not keep lecture history, audio,
or transcripts after returning the response. Azure services also process the data;
app-side deletion is not a claim about Azure's own service retention. Language calls
request `disable_service_logs=True`.

Only one recognition runs at once **per Python process**. Run one local server
process for F0 testing. The in-process lock is not a distributed concurrency limit
for multiple production workers or app instances.

Example calls (replace the audio path with your own recording):

```bash
curl -X POST http://127.0.0.1:8000/transcribe -F 'audio=@/absolute/path/lecture.wav'
curl -X POST http://127.0.0.1:8000/lecture-notes -F 'audio=@/absolute/path/lecture.wav' -o checkpoint-response.json
```

Use the **real** `/lecture-notes` response from a recording you made in your K1
submission. The included offline tests do not establish live Azure connectivity.

## What is implemented and what comes later

Implemented here: local upload/transcription/Language pipeline, format validation,
temporary-file cleanup, source-based study guide, local latency measurements, and
`key_terms_per_minute` (distinct Azure-returned phrases divided by audio minutes).

The initial notes are selected transcript sentences. The glossary contains
**source excerpts**, not generated definitions. Quiz questions are fill-in-the-blank
templates. Three questions are returned when three distinct source sentences match
extracted terms; otherwise the response explicitly reports insufficient material.
These are an initial baseline, not a completed Part I distinctive feature.

Still to do: finalize the one-page brief and Track 2 proposal, choose/deepen Track 1,
add the product frontend and any spoken output specified in the approved brief,
deploy to App Service, and implement Part G instrumentation (including initialization
before framework/SDK imports, custom metrics, traces, KQL, an alert, and
`/telemetry-summary`). Application Insights is provisioned but **this checkpoint
starter does not yet send telemetry**. Its connection string is retained for Part G.

## Offline verification

```bash
python -m pip install -r requirements-dev.txt
python -m unittest discover -s tests -v
```

Tests cover multiple recognition events, missing confidence values, silence,
upload-size and format failures, temporary-file cleanup on failure, source-based
questions, and the combined response. Azure calls are mocked.

## Documentation

- https://learn.microsoft.com/azure/ai-services/speech-service/how-to-recognize-speech?pivots=programming-language-python
- https://learn.microsoft.com/python/api/overview/azure/ai-textanalytics-readme
- https://fastapi.tiangolo.com/tutorial/request-files/

AI assistance was used to prepare this starter. Read and understand the code before
submitting it. Your final reflection must describe your own observations.
