"""Azure Language analysis, with bounded document sizes for short lectures."""

import os
from errors import AppError


def chunk_text(text: str, size: int = 4000) -> list[str]:
    chunks = []
    remaining = text.strip()
    while remaining:
        if len(remaining) <= size:
            chunks.append(remaining)
            break
        boundary = remaining.rfind(" ", 0, size + 1)
        boundary = boundary if boundary > size // 2 else size
        chunks.append(remaining[:boundary])
        remaining = remaining[boundary:].lstrip()
    return chunks


def azure_failure(error) -> AppError:
    status = getattr(error, "status_code", None)
    if status == 429:
        return AppError(503, "Azure Language is rate-limited. Wait briefly, then try again.")
    if status in (401, 403):
        return AppError(502, "Azure Language rejected authentication. Check AZURE_LANGUAGE_KEY and AZURE_LANGUAGE_ENDPOINT in .env.")
    return AppError(502, "Azure Language could not complete analysis. Check the resource, network connection, and service quota.")


def language_details(client, text: str) -> dict:
    result = client.detect_language([text[:4000]], disable_service_logs=True)[0]
    if result.is_error:
        raise AppError(502, "Azure Language could not identify the transcript language.")
    language = result.primary_language
    return {"name": language.name, "code": language.iso6391_name,
            "confidence": round(language.confidence_score, 4), "source": "Azure AI Language transcript detection"}


def detect_transcript_language(text: str) -> dict:
    from azure.ai.textanalytics import TextAnalyticsClient
    from azure.core.credentials import AzureKeyCredential
    from azure.core.exceptions import AzureError

    try:
        with TextAnalyticsClient(os.environ["AZURE_LANGUAGE_ENDPOINT"], AzureKeyCredential(os.environ["AZURE_LANGUAGE_KEY"]),
                                 retry_total=2, connection_timeout=10, read_timeout=30) as client:
            return language_details(client, text)
    except AzureError as error:
        raise azure_failure(error) from None


def analyze_text(text: str) -> dict:
    from azure.ai.textanalytics import TextAnalyticsClient
    from azure.core.credentials import AzureKeyCredential
    from azure.core.exceptions import AzureError

    chunks = chunk_text(text)
    try:
        with TextAnalyticsClient(os.environ["AZURE_LANGUAGE_ENDPOINT"], AzureKeyCredential(os.environ["AZURE_LANGUAGE_KEY"]),
                                 retry_total=2, connection_timeout=10, read_timeout=30) as client:
            detected = language_details(client, text)
            phrases, entities, sentiments = [], [], []
            # Keep each API batch small; Azure's per-request document limits vary by feature.
            for start in range(0, len(chunks), 5):
                batch = chunks[start:start + 5]
                options = {"language": detected["code"], "disable_service_logs": True}
                phrase_results = client.extract_key_phrases(batch, **options)
                entity_results = client.recognize_entities(batch, **options)
                sentiment_results = client.analyze_sentiment(batch, **options)
                for index, (p, e, s) in enumerate(zip(phrase_results, entity_results, sentiment_results)):
                    if p.is_error or e.is_error or s.is_error:
                        raise AppError(502, "Azure Language returned an analysis error for part of the transcript.")
                    phrases.extend(p.key_phrases)
                    entities.extend({"text": item.text, "category": item.category,
                                     "subcategory": item.subcategory, "confidence": round(item.confidence_score, 4)}
                                    for item in e.entities)
                    sentiments.append({"label": s.sentiment, "weight": len(batch[index]),
                                       "scores": {label: getattr(s.confidence_scores, label)
                                                  for label in ("positive", "neutral", "negative")}})
    except AzureError as error:
        raise azure_failure(error) from None

    key_terms = list(dict.fromkeys(phrases))
    unique_entities = {}
    for entity in entities:
        unique_entities.setdefault((entity["text"].casefold(), entity["category"]), entity)
    total_weight = sum(s["weight"] for s in sentiments)
    scores = {label: round(sum(s["scores"][label] * s["weight"] for s in sentiments) / total_weight, 4)
              for label in ("positive", "neutral", "negative")}
    label = sentiments[0]["label"] if len(sentiments) == 1 else (
        "mixed" if {"positive", "negative"}.issubset({s["label"] for s in sentiments}) else max(scores, key=scores.get))
    return {"language": detected, "key_terms": key_terms, "entities": list(unique_entities.values()),
            "sentiment": {"label": label, "confidence_scores": scores,
                          "aggregation": "single_document" if len(sentiments) == 1 else "character_weighted_chunks"},
            "chunks_analyzed": len(chunks)}
