"""A transparent baseline study guide: source excerpts and fill-in-the-blank questions.

This is not generative summarization. Track 1 can deepen it after brief approval.
"""

import re


def make_study_material(transcript: str, key_terms: list[str]) -> dict:
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", transcript) if s.strip()]
    glossary, used = [], set()
    for term in key_terms:
        normalized = term.casefold().strip()
        if not normalized or normalized in used:
            continue
        used.add(normalized)
        excerpt = next((s for s in sentences if re.search(r"(?<!\w)" + re.escape(term) + r"(?!\w)", s, re.IGNORECASE)), None)
        if excerpt:
            glossary.append({"term": term, "source_excerpt": excerpt})
    ranked = sorted(enumerate(sentences), key=lambda item: (
        -sum(term.casefold() in item[1].casefold() for term in key_terms), item[0]))
    chosen = sorted(ranked[:8], key=lambda item: item[0])
    quiz, question_sentences = [], set()
    for entry in glossary:
        # Prefer different source sentences rather than three variants of the same sentence.
        if entry["source_excerpt"] in question_sentences:
            continue
        question_sentences.add(entry["source_excerpt"])
        blank = re.sub(re.escape(entry["term"]), "_____", entry["source_excerpt"], count=1, flags=re.IGNORECASE)
        quiz.append({"question": "Fill in the blank: " + blank, "answer": entry["term"],
                     "source_excerpt": entry["source_excerpt"]})
        if len(quiz) == 3:
            break
    return {"notes": [sentence for _, sentence in chosen], "glossary": glossary, "quiz": quiz,
            "quiz_status": "ready" if len(quiz) == 3 else "insufficient_distinct_source_sentences",
            "method": "extractive_notes_contextual_glossary_and_template_quiz",
            "limitations": "Glossary entries show lecture excerpts, not independently generated definitions. Quiz questions are template-based."}
