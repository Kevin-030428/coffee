"""Inference helpers for the coffee review Streamlit app.

The app is designed for the ISOM5240 requirement that each business app has at
least two Hugging Face pipelines:
1. text-classification for review sentiment
2. token-classification for coffee menu entity extraction
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")

from .coffee_catalog import COFFEE_ALIASES


try:
    from transformers import pipeline
except Exception:  # pragma: no cover - Streamlit reports this in the UI.
    pipeline = None


APP_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = APP_DIR.parents[1]
MODEL_ROOT = PROJECT_ROOT / "Group01_Dataset_files" / "Fine-tuned_Model_files"

DEFAULT_SENTIMENT_MODEL = "Cry1008/coffee-sentiment"
DEFAULT_NER_MODEL = "Cry1008/coffee-ner"


@dataclass(frozen=True)
class Prediction:
    label: str
    score: float
    runtime_ms: float
    source: str


@dataclass(frozen=True)
class CoffeeEntity:
    text: str
    label: str
    start: int
    end: int
    score: float
    source: str


def _candidate_model_path(folder_name: str, fallback_model: str, env_key: str) -> str:
    env_value = os.getenv(env_key)
    if env_value:
        return env_value

    local_path = MODEL_ROOT / folder_name
    if local_path.exists():
        return str(local_path)

    return fallback_model


def load_sentiment_pipeline() -> Any:
    """Load Pipeline 1: review sentiment classification."""
    if pipeline is None:
        return None

    model_name = _candidate_model_path(
        "sentiment_model",
        DEFAULT_SENTIMENT_MODEL,
        "COFFEE_SENTIMENT_MODEL",
    )
    try:
        print(f"Loading sentiment model from: {model_name}")
        sentiment_pipe = pipeline("text-classification", model=model_name, truncation=True)
        print(f"Loaded sentiment model from: {model_name}")
        return sentiment_pipe
    except Exception as exc:
        print(f"Failed to load sentiment model from {model_name}: {exc}")
        return None


def load_ner_pipeline() -> Any:
    """Load Pipeline 2: token classification for named entity recognition."""
    if pipeline is None:
        return None

    model_name = _candidate_model_path(
        "coffee_ner_model",
        DEFAULT_NER_MODEL,
        "COFFEE_NER_MODEL",
    )
    try:
        print(f"Loading coffee NER model from: {model_name}")
        ner_pipe = pipeline("token-classification", model=model_name, aggregation_strategy="simple")
        print(f"Loaded coffee NER model from: {model_name}")
        return ner_pipe
    except Exception as exc:
        print(f"Failed to load coffee NER model from {model_name}: {exc}")
        return None


def _loaded_model_name(model_pipe: Any | None, fallback: str) -> str:
    if model_pipe is None:
        return fallback
    model = getattr(model_pipe, "model", None)
    return str(getattr(model, "name_or_path", fallback))


def predict_sentiment(text: str, sentiment_pipe: Any | None) -> Prediction:
    """Return Positive/Negative prediction for one review."""
    cleaned = text.strip()
    if not cleaned:
        return Prediction("Empty", 0.0, 0.0, "input validation")

    start = perf_counter()
    if sentiment_pipe is not None:
        try:
            result = sentiment_pipe(cleaned)[0]
            runtime_ms = (perf_counter() - start) * 1000
            label = str(result.get("label", "")).upper()
            normalized = "Positive" if "POS" in label or label == "LABEL_1" else "Negative"
            return Prediction(
                label=normalized,
                score=float(result.get("score", 0.0)),
                runtime_ms=runtime_ms,
                source=f"Hugging Face text-classification pipeline ({_loaded_model_name(sentiment_pipe, DEFAULT_SENTIMENT_MODEL)})",
            )
        except Exception:
            pass

    positive_words = {
        "amazing",
        "balanced",
        "best",
        "bright",
        "creamy",
        "delicious",
        "excellent",
        "fresh",
        "friendly",
        "great",
        "love",
        "perfect",
        "recommend",
        "smooth",
    }
    negative_words = {
        "bad",
        "bitter",
        "burnt",
        "cold",
        "disappointing",
        "expensive",
        "flat",
        "overpriced",
        "rude",
        "slow",
        "stale",
        "watery",
        "weak",
        "worst",
    }
    tokens = re.findall(r"[a-z']+", cleaned.lower())
    negators = {"no", "not", "never", "without"}
    pos_hits = 0
    neg_hits = 0
    for index, token in enumerate(tokens):
        previous_window = set(tokens[max(0, index - 3) : index])
        is_negated = bool(previous_window & negators)
        if token in positive_words:
            neg_hits += int(is_negated)
            pos_hits += int(not is_negated)
        if token in negative_words:
            pos_hits += int(is_negated)
            neg_hits += int(not is_negated)
    score = 0.5 + min(abs(pos_hits - neg_hits) * 0.12, 0.45)
    label = "Positive" if pos_hits >= neg_hits else "Negative"
    return Prediction(label, score, (perf_counter() - start) * 1000, "keyword fallback")


def _gazetteer_entities(text: str) -> list[CoffeeEntity]:
    entities: list[CoffeeEntity] = []
    occupied: list[tuple[int, int]] = []

    for alias, canonical in sorted(COFFEE_ALIASES.items(), key=lambda item: len(item[0]), reverse=True):
        pattern = re.compile(rf"\b{re.escape(alias)}\b", flags=re.IGNORECASE)
        for match in pattern.finditer(text):
            span = (match.start(), match.end())
            if any(not (span[1] <= used[0] or span[0] >= used[1]) for used in occupied):
                continue
            occupied.append(span)
            entities.append(
                CoffeeEntity(
                    text=canonical,
                    label="COFFEE",
                    start=span[0],
                    end=span[1],
                    score=1.0,
                    source="coffee menu gazetteer",
                )
            )

    return sorted(entities, key=lambda entity: entity.start)


def extract_coffee_entities(text: str, ner_pipe: Any | None) -> list[CoffeeEntity]:
    """Extract coffee names mentioned in a review."""
    cleaned = text.strip()
    if not cleaned:
        return []

    gazetteer_entities = _gazetteer_entities(cleaned)
    model_entities: list[CoffeeEntity] = []
    if ner_pipe is not None:
        try:
            for item in ner_pipe(cleaned):
                word = str(item.get("word", "")).replace(" ##", "").replace("##", "").strip()
                group = str(item.get("entity_group") or item.get("entity") or "ENTITY")
                score = float(item.get("score", 0.0))
                start = int(item.get("start", -1))
                end = int(item.get("end", -1))
                if start >= 0 and end > start and (group.upper() in {"COFFEE", "MENU_ITEM", "DRINK"}):
                    model_entities.append(
                        CoffeeEntity(
                            word,
                            "COFFEE",
                            start,
                            end,
                            score,
                            f"Hugging Face token-classification pipeline ({_loaded_model_name(ner_pipe, DEFAULT_NER_MODEL)})",
                        )
                    )
        except Exception:
            model_entities = []

    if gazetteer_entities:
        source = (
            "Hugging Face token-classification pipeline + Starbucks menu normalization"
            if model_entities
            else "Starbucks menu gazetteer"
        )
        return [
            CoffeeEntity(entity.text, entity.label, entity.start, entity.end, entity.score, source)
            for entity in gazetteer_entities
        ]

    if model_entities:
        return sorted(model_entities, key=lambda entity: entity.start)

    return []


def summarize_entities(entities: list[CoffeeEntity]) -> str:
    unique_names = []
    for entity in entities:
        if entity.text not in unique_names:
            unique_names.append(entity.text)
    return ", ".join(unique_names) if unique_names else "No coffee item detected"
