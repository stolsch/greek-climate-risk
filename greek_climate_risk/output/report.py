"""Generate analytical PDF report, structured findings, and export bundles."""

from __future__ import annotations

import json
import logging
import pickle
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from greek_climate_risk.factors.construct import FactorArtifact, build_climate_factors
from greek_climate_risk.lda.modeling import LdaArtifact
from greek_climate_risk.preprocessing.pipeline import CorpusArtifact
from greek_climate_risk.scrapers.sources import SOURCES

LOGGER = logging.getLogger(__name__)


def _escape_xml(text: str) -> str:
    """Escape text for ReportLab Paragraph mini-HTML."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _register_greek_font() -> str:
    """Register a TTF that supports Greek; return registered font name."""
    font_name = "ReportBody"
    try:
        mpl_dir = Path(matplotlib.__file__).resolve().parent
        dejavu = mpl_dir / "mpl-data" / "fonts" / "ttf" / "DejaVuSans.ttf"
        if dejavu.is_file():
            pdfmetrics.registerFont(TTFont(font_name, str(dejavu)))
            return font_name
    except OSError as exc:
        LOGGER.warning("Could not register DejaVu for PDF Greek support: %s", exc)
    return "Helvetica"


def _load_corpus_stats(db_path: Path) -> dict[str, Any]:
    """Collect headline corpus statistics from the database."""
    with sqlite3.connect(db_path) as connection:
        n_articles = connection.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
        date_min, date_max = connection.execute("SELECT MIN(date), MAX(date) FROM articles").fetchone()
        source_rows = connection.execute(
            "SELECT source, COUNT(*) FROM articles GROUP BY source ORDER BY COUNT(*) DESC"
        ).fetchall()
    return {
        "n_articles": int(n_articles or 0),
        "date_min": date_min or "N/A",
        "date_max": date_max or "N/A",
        "sources": [(str(s), int(c)) for s, c in source_rows],
    }


def load_corpus_pickle_summary(output_dir: Path) -> dict[str, Any]:
    """Load document and vocabulary counts from saved corpus.pkl."""
    path = output_dir / "corpus.pkl"
    if not path.is_file():
        return {"n_documents": 0, "vocabulary_size": 0}
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    vocab = payload.get("vocabulary") or {}
    docs = payload.get("docs") or []
    return {
        "n_documents": len(docs),
        "vocabulary_size": len(vocab),
    }


def gather_analytical_findings(
    config: dict[str, Any],
    db_path: Path,
    corpus_artifact: CorpusArtifact,
    lda_artifact: LdaArtifact,
    factor_artifact: FactorArtifact,
    output_dir: Path,
) -> dict[str, Any]:
    """Assemble structured findings for JSON export and narrative sections."""
    stats = _load_corpus_stats(db_path)
    topic_words = pd.read_csv(lda_artifact.topic_word_path)
    coherence_df = pd.read_csv(lda_artifact.coherence_scores_path)
    daily = pd.read_csv(factor_artifact.factors_daily_path)
    topic_cols = [c for c in daily.columns if c.startswith("topic_")]
    daily["date"] = pd.to_datetime(daily["date"], errors="coerce")
    corr = pd.read_csv(factor_artifact.correlations_path, index_col=0)

    top_by_topic: dict[str, list[dict[str, Any]]] = {}
    for t in sorted(topic_words["topic"].unique()):
        sub = topic_words[topic_words["topic"] == t].nlargest(10, "weight")
        top_by_topic[f"topic_{int(t)}"] = [
            {"word": str(r["word"]), "weight": float(r["weight"])} for _, r in sub.iterrows()
        ]

    factor_summary: dict[str, Any] = {}
    for col in topic_cols:
        s = daily[col].dropna()
        if s.empty:
            continue
        idx_max = s.idxmax()
        factor_summary[col] = {
            "mean": float(s.mean()),
            "std": float(s.std()) if len(s) > 1 else 0.0,
            "max": float(s.max()),
            "date_of_max": str(daily.loc[idx_max, "date"].date()) if pd.notna(daily.loc[idx_max, "date"]) else None,
        }

    corr_pairs: list[dict[str, Any]] = []
    cols = list(corr.columns)
    for i, a in enumerate(cols):
        for b in cols[i + 1 :]:
            corr_pairs.append({"topic_a": a, "topic_b": b, "correlation": float(corr.loc[a, b])})
    corr_pairs_sorted = sorted(corr_pairs, key=lambda x: abs(x["correlation"]), reverse=True)[:15]

    labels_path = output_dir / "topic_labels.json"
    topic_labels: dict[str, str] = {}
    if labels_path.is_file():
        with labels_path.open("r", encoding="utf-8") as handle:
            topic_labels = json.load(handle)

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "methodology_note": (
            "Media universe: the four newspapers used in Hardouvelis–Karalas–Karanastasis–Samartzis (HKKS) to "
            "trace Greek news — Kathimerini, To Vima, Ta Nea, Naftemporiki (scraped sources: "
            + ", ".join(s.name for s in SOURCES)
            + "). "
            "Corpus retention follows Faccini, Matin & Skiadopoulos (2023), Section 3.1: an article is kept "
            "if it contains at least one of the bigrams climate change or global warming, or the Greek "
            "equivalents κλιματική αλλαγή or παγκόσμια υπερθέρμανση. "
            "Factors are constructed as levels: the daily sum of article-level topic shares, as in FMS (2023)."
        ),
        "config_snapshot": {
            "start_date": config.get("global", {}).get("start_date"),
            "seed_queries": config.get("scraping", {}).get("seed_queries"),
            "filter_keywords": config.get("preprocessing", {}).get("filter_keywords"),
            "hkks_outlets": [s.name for s in SOURCES],
            "lda_k_grid": config.get("lda", {}).get("k_values"),
        },
        "corpus": {
            "articles_in_db": stats["n_articles"],
            "date_range_db": {"min": stats["date_min"], "max": stats["date_max"]},
            "sources": [{"source": s, "count": c} for s, c in stats["sources"]],
            "preprocessed_documents": len(corpus_artifact.tokenized_docs),
            "vocabulary_size": len(corpus_artifact.vocabulary),
        },
        "lda": {
            "optimal_k": lda_artifact.optimal_k,
            "coherence_by_k": coherence_df.to_dict(orient="records"),
            "top_words_per_topic": top_by_topic,
            "topic_labels": topic_labels,
        },
        "factors": {
            "n_days": int(daily.shape[0]),
            "daily_columns": topic_cols,
            "summary_by_topic": factor_summary,
            "strongest_correlations": corr_pairs_sorted,
        },
        "output_files": {
            "pdf_report": "summary_report.pdf",
            "findings_json": "analytical_findings.json",
            "findings_txt": "analytical_findings.txt",
            "factors_daily": "factors_daily.csv",
            "correlations": "factor_correlations.csv",
            "topic_words": "topic_word_distributions.csv",
            "article_topics": "article_topic_shares.csv",
        },
    }


def _findings_to_plain_text(findings: dict[str, Any]) -> str:
    """Human-readable UTF-8 text for download."""
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append("ANALYTICAL REPORT — Greek Climate Risk Pipeline")
    lines.append(f"Generated (UTC): {findings.get('generated_at_utc', '')}")
    lines.append("=" * 72)
    lines.append("")
    lines.append(findings.get("methodology_note", ""))
    lines.append("")
    c = findings.get("corpus", {})
    lines.append("--- CORPUS ---")
    lines.append(f"Articles in database: {c.get('articles_in_db')}")
    lines.append(f"Date range (DB): {c.get('date_range_db')}")
    lines.append(f"Preprocessed documents (LDA): {c.get('preprocessed_documents')}")
    lines.append(f"Vocabulary size: {c.get('vocabulary_size')}")
    for s in c.get("sources", []):
        lines.append(f"  - {s.get('source')}: {s.get('count')}")
    lines.append("")
    lda = findings.get("lda", {})
    lines.append("--- LDA ---")
    lines.append(f"Optimal K: {lda.get('optimal_k')}")
    lines.append("Topic coherence (UMass, mean over topics) by K:")
    for row in lda.get("coherence_by_k", []):
        cval = row.get("coherence", row.get("coherence_cv"))
        lines.append(f"  K={row.get('K')}: {cval}")
    lines.append("")
    lines.append("Top words per topic (sample):")
    for topic_key, words in (lda.get("top_words_per_topic") or {}).items():
        wstr = ", ".join(f"{w['word']} ({w['weight']:.3f})" for w in words[:5])
        lines.append(f"  {topic_key}: {wstr}")
    lines.append("")
    fct = findings.get("factors", {})
    lines.append("--- FACTORS (levels) ---")
    lines.append(f"Days in factors_daily: {fct.get('n_days')}")
    for col, summ in (fct.get("summary_by_topic") or {}).items():
        lines.append(
            f"  {col}: mean={summ.get('mean'):.4f}, max={summ.get('max'):.4f}, "
            f"date_of_max={summ.get('date_of_max')}"
        )
    lines.append("")
    lines.append("Strongest correlations:")
    for p in (fct.get("strongest_correlations") or [])[:10]:
        lines.append(
            f"  {p.get('topic_a')} vs {p.get('topic_b')}: {p.get('correlation'):.3f}"
        )
    lines.append("")
    lines.append("--- OUTPUT FILES ---")
    for k, v in (findings.get("output_files") or {}).items():
        lines.append(f"  {k}: {v}")
    return "\n".join(lines)


def save_findings_exports(findings: dict[str, Any], output_dir: Path) -> tuple[Path, Path]:
    """Write JSON and plain-text findings next to other outputs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "analytical_findings.json"
    txt_path = output_dir / "analytical_findings.txt"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(findings, handle, ensure_ascii=False, indent=2)
    txt_path.write_text(_findings_to_plain_text(findings), encoding="utf-8")
    LOGGER.info("Saved findings to %s and %s", json_path, txt_path)
    return json_path, txt_path


def build_pdf_report(
    config: dict[str, Any],
    db_path: Path,
    corpus_artifact: CorpusArtifact,
    lda_artifact: LdaArtifact,
    factor_artifact: FactorArtifact,
    output_dir: Path,
) -> Path:
    """Build multi-page analytical PDF with embedded figures and tables."""
    output_dir.mkdir(parents=True, exist_ok=True)
    findings = gather_analytical_findings(
        config=config,
        db_path=db_path,
        corpus_artifact=corpus_artifact,
        lda_artifact=lda_artifact,
        factor_artifact=factor_artifact,
        output_dir=output_dir,
    )
    save_findings_exports(findings, output_dir)

    body_font = _register_greek_font()
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="BodyJustify",
            parent=styles["Normal"],
            fontName=body_font,
            fontSize=10,
            leading=14,
            alignment=TA_JUSTIFY,
        )
    )
    styles.add(
        ParagraphStyle(
            name="H2",
            parent=styles["Heading2"],
            fontName=body_font,
            fontSize=14,
            spaceAfter=12,
        )
    )
    styles.add(
        ParagraphStyle(
            name="TitleCenter",
            parent=styles["Title"],
            fontName=body_font,
            alignment=TA_CENTER,
        )
    )

    report_path = output_dir / "summary_report.pdf"
    doc = SimpleDocTemplate(
        str(report_path),
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
    )
    story: list[Any] = []

    story.append(Paragraph(_escape_xml("Αναλυτική αναφορά — Κλιματικός κίνδυνος (ελληνικός Τύπος)"), styles["TitleCenter"]))
    story.append(Spacer(1, 0.4 * cm))
    story.append(
        Paragraph(
            _escape_xml(
                f"Ημερομηνία παραγωγής (UTC): {findings['generated_at_utc'][:19]}. "
                "Η αναφορά συνοψίζει corpus, LDA και ημερήσιους παράγοντες επιπέδου."
            ),
            styles["BodyJustify"],
        )
    )
    story.append(Spacer(1, 0.3 * cm))
    story.append(Paragraph(_escape_xml(findings["methodology_note"]), styles["BodyJustify"]))
    story.append(PageBreak())

    story.append(Paragraph(_escape_xml("1. Στατιστικά corpus"), styles["H2"]))
    c = findings["corpus"]
    corp_data = [
        [_escape_xml("Μέτρηση"), _escape_xml("Τιμή")],
        [_escape_xml("Άρθρα στη βάση"), str(c["articles_in_db"])],
        [_escape_xml("Εύρος ημερομηνιών (DB)"), f"{c['date_range_db']['min']} — {c['date_range_db']['max']}"],
        [_escape_xml("Κείμενα μετά προεπεξεργασία"), str(c["preprocessed_documents"])],
        [_escape_xml("Μέγεθος λεξιλογίου"), str(c["vocabulary_size"])],
    ]
    t = Table(corp_data, colWidths=[6.5 * cm, 10 * cm])
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4472C4")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("FONTNAME", (0, 0), (-1, -1), body_font),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )
    story.append(t)
    story.append(Spacer(1, 0.3 * cm))
    src_rows = [[_escape_xml("Πηγή"), _escape_xml("Πλήθος")]] + [
        [_escape_xml(s["source"]), str(s["count"])] for s in c["sources"]
    ]
    st_src = Table(src_rows, colWidths=[8 * cm, 4 * cm])
    st_src.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#5B9BD5")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("FONTNAME", (0, 0), (-1, -1), body_font),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
            ]
        )
    )
    story.append(st_src)
    story.append(PageBreak())

    story.append(Paragraph(_escape_xml("2. LDA — επιλογή K και κορυφαίες λέξεις"), styles["H2"]))
    story.append(
        Paragraph(
            _escape_xml(f"Βέλτιστο πλήθος θεμάτων: K = {findings['lda']['optimal_k']}."),
            styles["BodyJustify"],
        )
    )
    coh = findings["lda"]["coherence_by_k"]
    coh_tbl = [[_escape_xml("K"), _escape_xml("Συνοχή (UMass)")]] + [
        [
            str(int(r["K"])),
            f"{float(r.get('coherence', r.get('coherence_cv', 0))):.4f}",
        ]
        for r in coh
    ]
    ct = Table(coh_tbl, colWidths=[3 * cm, 5 * cm])
    ct.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#70AD47")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("FONTNAME", (0, 0), (-1, -1), body_font),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
            ]
        )
    )
    story.append(ct)
    story.append(Spacer(1, 0.3 * cm))

    for topic_key, words in list((findings["lda"].get("top_words_per_topic") or {}).items())[:8]:
        label = findings["lda"].get("topic_labels", {}).get(topic_key, "")
        line = ", ".join(f"{w['word']} ({w['weight']:.2f})" for w in words[:12])
        suffix = f" — {label}" if label and label != "TODO_LABEL" else ""
        story.append(Paragraph(_escape_xml(f"{topic_key}{suffix}: {line}"), styles["BodyJustify"]))
    story.append(Spacer(1, 0.3 * cm))

    if lda_artifact.coherence_plot_path.is_file():
        story.append(
            Image(str(lda_artifact.coherence_plot_path), width=16 * cm, height=10 * cm)
        )
    story.append(PageBreak())

    story.append(Paragraph(_escape_xml("3. Παράγοντες κλιματικού κινδύνου (επίπεδα)"), styles["H2"]))
    fct = findings["factors"]
    fs_rows = [
        [_escape_xml("Θέμα"), _escape_xml("Μέση"), _escape_xml("Μέγιστο"), _escape_xml("Ημ. μέγιστου")],
    ]
    for col, summ in (fct.get("summary_by_topic") or {}).items():
        fs_rows.append(
            [
                _escape_xml(col),
                f"{summ['mean']:.4f}",
                f"{summ['max']:.4f}",
                _escape_xml(str(summ.get("date_of_max") or "")),
            ]
        )
    ft = Table(fs_rows, colWidths=[3.5 * cm, 3 * cm, 3 * cm, 5 * cm])
    ft.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#ED7D31")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("FONTNAME", (0, 0), (-1, -1), body_font),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
            ]
        )
    )
    story.append(ft)
    story.append(Spacer(1, 0.3 * cm))
    story.append(Paragraph(_escape_xml("Ισχυρότερες ζεύξεις συσχέτισης:"), styles["BodyJustify"]))
    for p in (fct.get("strongest_correlations") or [])[:12]:
        story.append(
            Paragraph(
                _escape_xml(
                    f"{p['topic_a']} vs {p['topic_b']}: ρ = {p['correlation']:.3f}"
                ),
                styles["BodyJustify"],
            )
        )

    if factor_artifact.monthly_plot_path.is_file():
        story.append(Spacer(1, 0.2 * cm))
        story.append(
            Image(str(factor_artifact.monthly_plot_path), width=16 * cm, height=8 * cm)
        )
    if factor_artifact.correlation_heatmap_path.is_file():
        story.append(
            Image(str(factor_artifact.correlation_heatmap_path), width=14 * cm, height=11 * cm)
        )

    story.append(PageBreak())
    story.append(Paragraph(_escape_xml("4. Αρχεία για λήψη / αναπαραγωγή"), styles["H2"]))
    for k, v in (findings.get("output_files") or {}).items():
        story.append(Paragraph(_escape_xml(f"{k}: {v}"), styles["BodyJustify"]))

    doc.build(story)
    LOGGER.info("Saved PDF report to %s", report_path)

    stats_csv = output_dir / "corpus_statistics.csv"
    stats = _load_corpus_stats(db_path)
    stats_rows = [
        {"metric": "n_articles", "value": stats["n_articles"]},
        {"metric": "date_min", "value": stats["date_min"]},
        {"metric": "date_max", "value": stats["date_max"]},
        {"metric": "preprocessed_docs", "value": len(corpus_artifact.tokenized_docs)},
        {"metric": "vocabulary_size", "value": len(corpus_artifact.vocabulary)},
        {"metric": "optimal_k", "value": lda_artifact.optimal_k},
    ]
    for source, count in stats["sources"]:
        stats_rows.append({"metric": f"source_{source}", "value": count})
    pd.DataFrame(stats_rows).to_csv(stats_csv, index=False)
    return report_path


def finalize_pipeline_after_lda(config: dict[str, Any], project_root: Path) -> Path | None:
    """Run factor construction and full analytical report after LDA outputs exist."""
    db_path = project_root / config["paths"]["database"]
    output_dir = project_root / config["paths"]["output_dir"]
    pkl = output_dir / "corpus.pkl"
    topics_csv = output_dir / "article_topic_shares.csv"
    words_csv = output_dir / "topic_word_distributions.csv"
    coh_csv = output_dir / "coherence_scores.csv"
    if not all(f.is_file() for f in (pkl, topics_csv, words_csv, coh_csv)):
        LOGGER.warning("finalize_pipeline_after_lda: missing LDA or corpus files.")
        return None
    with pkl.open("rb") as handle:
        payload = pickle.load(handle)
    corpus_artifact = CorpusArtifact(
        corpus_path=pkl,
        article_ids=list(payload.get("article_ids") or []),
        dates=list(payload.get("dates") or []),
        tokenized_docs=list(payload.get("docs") or []),
        vocabulary=dict(payload.get("vocabulary") or {}),
    )
    tw_df = pd.read_csv(words_csv)
    optimal_k = int(tw_df["topic"].max()) if not tw_df.empty else 0
    lda_artifact = LdaArtifact(
        model_dir=output_dir / "lda_model",
        optimal_k=optimal_k,
        article_topic_path=topics_csv,
        topic_word_path=words_csv,
        coherence_plot_path=output_dir / "coherence_plot.png",
        coherence_scores_path=coh_csv,
    )
    factor_artifact = build_climate_factors(
        config=config, db_path=db_path, lda_artifact=lda_artifact, output_dir=output_dir
    )
    return build_pdf_report(
        config=config,
        db_path=db_path,
        corpus_artifact=corpus_artifact,
        lda_artifact=lda_artifact,
        factor_artifact=factor_artifact,
        output_dir=output_dir,
    )


def rebuild_report_from_disk(config: dict[str, Any], project_root: Path) -> Path | None:
    """Rebuild PDF + findings if output CSVs and DB exist (e.g. from GUI)."""
    db_path = project_root / config["paths"]["database"]
    output_dir = project_root / config["paths"]["output_dir"]
    needed = [
        output_dir / "article_topic_shares.csv",
        output_dir / "topic_word_distributions.csv",
        output_dir / "coherence_scores.csv",
        output_dir / "corpus.pkl",
    ]
    if not all(p.is_file() for p in needed) or not db_path.is_file():
        LOGGER.warning("Cannot rebuild report: missing outputs or database.")
        return None

    summary = load_corpus_pickle_summary(output_dir)
    with (output_dir / "corpus.pkl").open("rb") as handle:
        payload = pickle.load(handle)
    corpus_artifact = CorpusArtifact(
        corpus_path=output_dir / "corpus.pkl",
        article_ids=list(payload.get("article_ids") or []),
        dates=list(payload.get("dates") or []),
        tokenized_docs=list(payload.get("docs") or []),
        vocabulary=dict(payload.get("vocabulary") or {}),
    )
    if summary["n_documents"] and len(corpus_artifact.tokenized_docs) != summary["n_documents"]:
        LOGGER.debug("Corpus pickle doc count check: artifact vs summary.")

    tw_df = pd.read_csv(output_dir / "topic_word_distributions.csv")
    optimal_k = int(tw_df["topic"].max()) if not tw_df.empty else 0
    lda_artifact = LdaArtifact(
        model_dir=output_dir / "lda_model",
        optimal_k=int(optimal_k),
        article_topic_path=output_dir / "article_topic_shares.csv",
        topic_word_path=output_dir / "topic_word_distributions.csv",
        coherence_plot_path=output_dir / "coherence_plot.png",
        coherence_scores_path=output_dir / "coherence_scores.csv",
    )
    fac_daily = output_dir / "factors_daily.csv"
    fac_corr = output_dir / "factor_correlations.csv"
    if not fac_daily.is_file() or not fac_corr.is_file():
        LOGGER.warning("Factor outputs missing; run factor stage first.")
        return None
    factor_artifact = FactorArtifact(
        factors_daily_path=output_dir / "factors_daily.csv",
        monthly_plot_path=output_dir / "factors_monthly_plot.png",
        correlations_path=output_dir / "factor_correlations.csv",
        correlation_heatmap_path=output_dir / "factor_correlations_heatmap.png",
    )
    return build_pdf_report(
        config=config,
        db_path=db_path,
        corpus_artifact=corpus_artifact,
        lda_artifact=lda_artifact,
        factor_artifact=factor_artifact,
        output_dir=output_dir,
    )
