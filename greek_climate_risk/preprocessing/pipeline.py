"""Preprocess Greek text for LDA modeling."""

from __future__ import annotations

import logging
import pickle
import sqlite3
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import spacy

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class CorpusArtifact:
    """Container with corpus data needed by LDA stage."""

    corpus_path: Path
    article_ids: list[int]
    dates: list[str]
    tokenized_docs: list[list[str]]
    vocabulary: dict[str, int]


def _fetch_articles(db_path: Path) -> list[tuple[int, str, str, str]]:
    """Load id, title, body, date from DB."""
    with sqlite3.connect(db_path) as connection:
        rows = connection.execute(
            "SELECT id, title, body, date FROM articles ORDER BY date, id"
        ).fetchall()
    return [(int(row[0]), str(row[1]), str(row[2]), str(row[3])) for row in rows]


def _keyword_lemma_tokens(keyword: str, nlp: Any) -> list[str]:
    """Return alpha lemma tokens for one keyword expression."""
    return [
        token.lemma_.strip().lower()
        for token in nlp(keyword.lower())
        if token.is_alpha and token.lemma_.strip()
    ]


def _contains_lemma_phrase(doc_lemmas: list[str], phrase_lemmas: list[str]) -> bool:
    """Check if a lemmatized phrase appears in document lemmas."""
    if not phrase_lemmas:
        return False
    plen = len(phrase_lemmas)
    if plen == 1:
        return phrase_lemmas[0] in set(doc_lemmas)
    return any(doc_lemmas[i : i + plen] == phrase_lemmas for i in range(len(doc_lemmas) - plen + 1))


def _contains_keywords_smart(text: str, keywords: list[str], nlp: Any) -> bool:
    """Match keywords by exact string or lemma phrase presence."""
    lowered = text.lower()
    if any(keyword.lower() in lowered for keyword in keywords):
        return True
    doc_lemmas = [
        token.lemma_.strip().lower()
        for token in nlp(lowered)
        if token.is_alpha and token.lemma_.strip()
    ]
    return any(_contains_lemma_phrase(doc_lemmas, _keyword_lemma_tokens(keyword, nlp)) for keyword in keywords)


def _load_spacy_model(model_name: str):
    """Load spaCy model, downloading it automatically if missing."""
    try:
        return spacy.load(model_name, disable=["parser", "ner"])
    except OSError:
        LOGGER.warning("spaCy model '%s' not found. Attempting auto-download.", model_name)
        try:
            proc = subprocess.run(
                [sys.executable, "-m", "spacy", "download", model_name],
                check=True,
                capture_output=True,
                text=True,
            )
            if proc.stdout:
                LOGGER.info("%s", proc.stdout.strip())
        except subprocess.CalledProcessError as exc:
            err = (exc.stderr or exc.stdout or "").strip()
            LOGGER.error("Failed to auto-download spaCy model '%s': %s", model_name, err)
            raise RuntimeError(
                f"Could not install spaCy model '{model_name}'. "
                "Deploy installs should list the model wheel in requirements.txt (see README). "
                f"Locally run: python -m spacy download {model_name}"
            ) from exc
        return spacy.load(model_name, disable=["parser", "ner"])


def run_preprocessing(config: dict[str, Any], db_path: Path, output_dir: Path) -> CorpusArtifact:
    """Build cleaned corpus and save `corpus.pkl` artifact."""
    preprocess_cfg = config["preprocessing"]
    nlp = _load_spacy_model(preprocess_cfg["spacy_model"])
    rows = _fetch_articles(db_path)
    keywords = preprocess_cfg["filter_keywords"]
    filtered_rows = [
        row
        for row in rows
        if _contains_keywords_smart(f"{row[1]} {row[2]}", keywords, nlp)
    ]
    if not filtered_rows:
        LOGGER.warning(
            "Keyword filtering retained zero rows; falling back to all scraped articles for preprocessing."
        )
        filtered_rows = rows
    LOGGER.info("Filtered articles retained: %s of %s", len(filtered_rows), len(rows))

    greek_stopwords = set(nlp.Defaults.stop_words)
    custom_stopwords = set(preprocess_cfg["custom_stopwords"])
    stopwords = greek_stopwords | custom_stopwords

    tokenized_docs: list[list[str]] = []
    article_ids: list[int] = []
    dates: list[str] = []

    for article_id, _title, text, date in filtered_rows:
        doc = nlp(text.lower())
        tokens = [
            token.lemma_.strip()
            for token in doc
            if token.is_alpha and token.lemma_ and token.lemma_ not in stopwords
        ]
        if tokens:
            tokenized_docs.append(tokens)
            article_ids.append(article_id)
            dates.append(date)
    if not tokenized_docs:
        raise ValueError(
            "Preprocessing produced no tokenized documents. "
            "Check source article text quality or relax stopword filtering."
        )

    document_frequency = Counter()
    for tokens in tokenized_docs:
        document_frequency.update(set(tokens))
    n_docs = len(tokenized_docs) or 1
    configured_min_docs = int(preprocess_cfg["min_document_frequency"])
    min_docs = min(configured_min_docs, max(2, int(n_docs * 0.2)))
    max_ratio = preprocess_cfg["max_document_ratio"]
    if min_docs != configured_min_docs:
        LOGGER.warning(
            "Adjusted min_document_frequency from %s to %s due to limited corpus size (%s docs).",
            configured_min_docs,
            min_docs,
            n_docs,
        )
    vocab_terms = {
        term
        for term, df in document_frequency.items()
        if df >= min_docs and (df / n_docs) <= max_ratio
    }
    if not vocab_terms and document_frequency:
        LOGGER.warning(
            "No terms survived strict thresholds. Falling back to relaxed thresholds for small sample."
        )
        relaxed_min_docs = 1
        relaxed_max_ratio = min(0.9, max_ratio + 0.25)
        vocab_terms = {
            term
            for term, df in document_frequency.items()
            if df >= relaxed_min_docs and (df / n_docs) <= relaxed_max_ratio
        }
    cleaned_docs: list[list[str]] = []
    cleaned_article_ids: list[int] = []
    cleaned_dates: list[str] = []
    for tokens, article_id, date in zip(tokenized_docs, article_ids, dates):
        filtered_tokens = [token for token in tokens if token in vocab_terms]
        if not filtered_tokens:
            continue
        cleaned_docs.append(filtered_tokens)
        cleaned_article_ids.append(article_id)
        cleaned_dates.append(date)

    if not cleaned_docs and tokenized_docs:
        LOGGER.warning(
            "Vocabulary filtering dropped all documents; applying emergency fallback thresholds."
        )
        emergency_vocab_terms = set(document_frequency.keys())
        for tokens, article_id, date in zip(tokenized_docs, article_ids, dates):
            filtered_tokens = [token for token in tokens if token in emergency_vocab_terms]
            if not filtered_tokens:
                continue
            cleaned_docs.append(filtered_tokens)
            cleaned_article_ids.append(article_id)
            cleaned_dates.append(date)
        vocab_terms = emergency_vocab_terms

    if not cleaned_docs:
        raise ValueError(
            "Preprocessing produced an empty corpus. "
            "Try broadening keywords or reducing frequency thresholds in config.yaml."
        )

    vocabulary = {term: idx for idx, term in enumerate(sorted(vocab_terms))}
    corpus_path = output_dir / "corpus.pkl"
    payload = {
        "docs": cleaned_docs,
        "vocabulary": vocabulary,
        "article_ids": cleaned_article_ids,
        "dates": cleaned_dates,
    }
    with corpus_path.open("wb") as handle:
        pickle.dump(payload, handle)
    LOGGER.info("Saved preprocessed corpus to %s", corpus_path)

    return CorpusArtifact(
        corpus_path=corpus_path,
        article_ids=payload["article_ids"],
        dates=payload["dates"],
        tokenized_docs=cleaned_docs,
        vocabulary=vocabulary,
    )
