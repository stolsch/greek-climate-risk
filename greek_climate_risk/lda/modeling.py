"""Train and export LDA model artifacts."""

from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import LatentDirichletAllocation
from wordcloud import WordCloud

from greek_climate_risk.preprocessing.pipeline import CorpusArtifact

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class LdaArtifact:
    """Container with outputs needed by factor construction and reporting."""

    model_dir: Path
    optimal_k: int
    article_topic_path: Path
    topic_word_path: Path
    coherence_plot_path: Path
    coherence_scores_path: Path


def _build_vocabulary(tokenized_docs: list[list[str]]) -> tuple[list[str], dict[str, int]]:
    vocab = sorted({t for doc in tokenized_docs for t in doc})
    word2idx = {w: i for i, w in enumerate(vocab)}
    return vocab, word2idx


def _build_doc_term_matrix(
    tokenized_docs: list[list[str]],
    word2idx: dict[str, int],
) -> sparse.csr_matrix:
    indptr: list[int] = [0]
    indices: list[int] = []
    data: list[int] = []
    for doc in tokenized_docs:
        for w, c in Counter(doc).items():
            if w in word2idx:
                indices.append(word2idx[w])
                data.append(int(c))
        indptr.append(len(indices))
    n_docs = len(tokenized_docs)
    n_terms = len(word2idx)
    return sparse.csr_matrix((data, indices, indptr), shape=(n_docs, n_terms), dtype=np.int64)


def _word_document_sets(tokenized_docs: list[list[str]]) -> dict[str, set[int]]:
    m: dict[str, set[int]] = defaultdict(set)
    for i, doc in enumerate(tokenized_docs):
        for w in set(doc):
            m[w].add(i)
    return m


def _umass_topic_words(top_words: list[str], word_docs: dict[str, set[int]], epsilon: float = 1e-12) -> float:
    if len(top_words) < 2:
        return 0.0
    total = 0.0
    n_pairs = 0
    for i, wi in enumerate(top_words):
        di = word_docs.get(wi)
        if not di:
            continue
        for j in range(i + 1, len(top_words)):
            wj = top_words[j]
            dj = word_docs.get(wj)
            if not dj:
                continue
            co = len(di & dj)
            total += float(np.log((co + epsilon) / len(di)))
            n_pairs += 1
    return total / n_pairs if n_pairs else 0.0


def _topic_top_words(components_row: np.ndarray, feature_names: list[str], topn: int) -> list[str]:
    top_idx = np.argsort(-components_row)[:topn]
    return [feature_names[i] for i in top_idx]


def _mean_umass_coherence(
    lda: LatentDirichletAllocation,
    feature_names: list[str],
    tokenized_docs: list[list[str]],
    word_docs: dict[str, set[int]],
    topn: int,
) -> float:
    scores: list[float] = []
    for t in range(lda.n_components):
        words = _topic_top_words(lda.components_[t], feature_names, topn)
        scores.append(_umass_topic_words(words, word_docs))
    return float(np.mean(scores)) if scores else 0.0


def _sklearn_max_iter(passes: int) -> int:
    return max(100, int(passes) * 5)


def _fit_lda(
    X: sparse.csr_matrix,
    n_topics: int,
    random_state: int,
    passes: int,
) -> LatentDirichletAllocation:
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=random_state,
        max_iter=_sklearn_max_iter(passes),
        learning_method="batch",
        n_jobs=1,
    )
    lda.fit(X)
    return lda


def _compute_coherence(
    X: sparse.csr_matrix,
    feature_names: list[str],
    tokenized_docs: list[list[str]],
    word_docs: dict[str, set[int]],
    k_values: list[int],
    random_state: int,
    passes: int,
    coherence_topn: int,
) -> tuple[int, list[tuple[int, float]]]:
    scores: list[tuple[int, float]] = []
    for k in k_values:
        lda = _fit_lda(X, k, random_state, passes)
        coherence = _mean_umass_coherence(lda, feature_names, tokenized_docs, word_docs, coherence_topn)
        scores.append((k, coherence))
        LOGGER.info("Mean UMass coherence for K=%s: %.4f", k, coherence)
    best_k = max(scores, key=lambda item: item[1])[0]
    return best_k, scores


def _save_coherence_plot(scores: list[tuple[int, float]], path: Path) -> None:
    ks = [item[0] for item in scores]
    values = [item[1] for item in scores]
    plt.figure(figsize=(8, 5))
    plt.plot(ks, values, marker="o")
    plt.title("LDA topic coherence (UMass) by number of topics")
    plt.xlabel("Number of Topics (K)")
    plt.ylabel("Mean UMass coherence (higher is better)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def run_lda_pipeline(config: dict[str, Any], corpus_artifact: CorpusArtifact, output_dir: Path) -> LdaArtifact:
    """Optimize and fit final LDA model, saving all requested artifacts."""
    if not corpus_artifact.tokenized_docs:
        raise ValueError(
            "LDA cannot run because the corpus is empty. "
            "Run preprocessing with broader filters or relaxed token thresholds."
        )
    lda_cfg = config["lda"]
    feature_names, word2idx = _build_vocabulary(corpus_artifact.tokenized_docs)
    if not feature_names:
        raise ValueError(
            "LDA cannot run because preprocessing returned no vocabulary terms. "
            "Relax min_document_frequency/max_document_ratio in config.yaml."
        )
    X = _build_doc_term_matrix(corpus_artifact.tokenized_docs, word2idx)
    word_docs = _word_document_sets(corpus_artifact.tokenized_docs)
    coherence_topn = min(20, int(lda_cfg["top_words_per_topic"]))

    optimal_k, scores = _compute_coherence(
        X=X,
        feature_names=feature_names,
        tokenized_docs=corpus_artifact.tokenized_docs,
        word_docs=word_docs,
        k_values=lda_cfg["k_values"],
        random_state=lda_cfg["random_state"],
        passes=lda_cfg["passes"],
        coherence_topn=coherence_topn,
    )
    coherence_plot_path = output_dir / "coherence_plot.png"
    _save_coherence_plot(scores, coherence_plot_path)
    coherence_scores_path = output_dir / "coherence_scores.csv"
    pd.DataFrame(scores, columns=["K", "coherence"]).to_csv(coherence_scores_path, index=False)

    model = _fit_lda(X, optimal_k, lda_cfg["random_state"], lda_cfg["passes"])
    model_dir = output_dir / "lda_model"
    model_dir.mkdir(parents=True, exist_ok=True)
    bundle = {
        "lda": model,
        "feature_names": feature_names,
    }
    joblib.dump(bundle, model_dir / "model.joblib")

    topn = int(lda_cfg["top_words_per_topic"])
    topic_word_rows: list[dict[str, Any]] = []
    for topic_id in range(optimal_k):
        row = model.components_[topic_id].astype(np.float64, copy=False)
        s = float(row.sum())
        if s <= 0:
            s = 1.0
        order = np.argsort(-row)[:topn]
        for i in order:
            w = feature_names[int(i)]
            topic_word_rows.append({"topic": topic_id + 1, "word": w, "weight": float(row[int(i)] / s)})
    topic_word_path = output_dir / "topic_word_distributions.csv"
    pd.DataFrame(topic_word_rows).to_csv(topic_word_path, index=False)

    doc_topics = model.transform(X)
    article_topic_rows: list[dict[str, Any]] = []
    for idx in range(X.shape[0]):
        row = {
            "article_id": corpus_artifact.article_ids[idx],
            "date": corpus_artifact.dates[idx],
        }
        dist = doc_topics[idx]
        for topic_id in range(optimal_k):
            row[f"topic_{topic_id + 1}"] = float(dist[topic_id])
        article_topic_rows.append(row)
    article_topic_path = output_dir / "article_topic_shares.csv"
    pd.DataFrame(article_topic_rows).to_csv(article_topic_path, index=False)

    wordcloud_dir = output_dir / "wordclouds"
    wordcloud_dir.mkdir(parents=True, exist_ok=True)
    for topic_id in range(optimal_k):
        row = model.components_[topic_id]
        s = float(row.sum()) or 1.0
        order = np.argsort(-row)[:topn]
        word_freq = {feature_names[int(i)]: float(row[int(i)] / s) for i in order}
        cloud = WordCloud(width=1200, height=800, background_color="white").generate_from_frequencies(word_freq)
        cloud.to_file(str(wordcloud_dir / f"topic_{topic_id + 1}.png"))

    topic_labels_path = output_dir / "topic_labels.json"
    with topic_labels_path.open("w", encoding="utf-8") as handle:
        json.dump({f"topic_{idx + 1}": "TODO_LABEL" for idx in range(optimal_k)}, handle, ensure_ascii=False, indent=2)

    LOGGER.info("LDA pipeline complete with optimal K=%s", optimal_k)
    return LdaArtifact(
        model_dir=model_dir,
        optimal_k=optimal_k,
        article_topic_path=article_topic_path,
        topic_word_path=topic_word_path,
        coherence_plot_path=coherence_plot_path,
        coherence_scores_path=coherence_scores_path,
    )
