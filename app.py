"""Streamlit GUI for the Greek climate risk pipeline."""

from __future__ import annotations

import io
import json
import logging
import sqlite3
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
import yaml

from greek_climate_risk.config import load_config
from greek_climate_risk.factors.construct import build_climate_factors
from greek_climate_risk.lda.modeling import run_lda_pipeline
from greek_climate_risk.output.report import (
    finalize_pipeline_after_lda,
    rebuild_report_from_disk,
    build_pdf_report,
)
from greek_climate_risk.preprocessing.pipeline import run_preprocessing
from greek_climate_risk.scrapers.pipeline import run_scraping_pipeline
from greek_climate_risk.scrapers.sources import SOURCES


def configure_logging() -> None:
    """Configure logging for Streamlit app execution context."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def run_full_pipeline(config: dict[str, Any], project_root: Path) -> None:
    """Run all phases from scraping through report generation."""
    db_path = project_root / config["paths"]["database"]
    output_dir = project_root / config["paths"]["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    run_scraping_pipeline(config=config, db_path=db_path)
    corpus_artifact = run_preprocessing(config=config, db_path=db_path, output_dir=output_dir)
    lda_artifact = run_lda_pipeline(config=config, corpus_artifact=corpus_artifact, output_dir=output_dir)
    factor_artifact = build_climate_factors(config=config, db_path=db_path, lda_artifact=lda_artifact, output_dir=output_dir)
    build_pdf_report(
        config=config,
        db_path=db_path,
        corpus_artifact=corpus_artifact,
        lda_artifact=lda_artifact,
        factor_artifact=factor_artifact,
        output_dir=output_dir,
    )


def save_config(path: Path, config: dict[str, Any]) -> None:
    """Persist updated config values to disk."""
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, allow_unicode=True, sort_keys=False)


def _load_findings_json(output_dir: Path) -> dict[str, Any] | None:
    """Load analytical_findings.json if present."""
    path = output_dir / "analytical_findings.json"
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _build_export_zip(project_root: Path, output_dir: Path) -> bytes | None:
    """Zip key deliverables for one-click download."""
    candidates = [
        output_dir / "summary_report.pdf",
        output_dir / "analytical_findings.json",
        output_dir / "analytical_findings.txt",
        output_dir / "corpus.pkl",
        output_dir / "factors_daily.csv",
        output_dir / "factor_correlations.csv",
        output_dir / "topic_word_distributions.csv",
        output_dir / "article_topic_shares.csv",
        output_dir / "coherence_scores.csv",
        output_dir / "coherence_plot.png",
        output_dir / "factors_monthly_plot.png",
        output_dir / "factor_correlations_heatmap.png",
        output_dir / "topic_labels.json",
    ]
    wc_dir = output_dir / "wordclouds"
    if wc_dir.is_dir():
        candidates.extend(sorted(wc_dir.glob("*.png")))

    existing = [p for p in candidates if p.is_file()]
    if not existing:
        return None
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in existing:
            arc = path.relative_to(project_root)
            zf.write(path, arcname=str(arc))
    return buf.getvalue()


def render_full_dashboard(project_root: Path, output_dir: Path, db_path: Path) -> None:
    """Show all outputs: metrics, tables, figures, downloads."""
    findings = _load_findings_json(output_dir)

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "Overview & downloads",
            "Corpus & database",
            "LDA & topics",
            "Factors",
            "Figures & word clouds",
        ]
    )

    with tab1:
        st.subheader("Summary findings")
        if findings:
            with st.expander("Full structured findings (JSON)", expanded=False):
                st.json(findings)
            corp = findings.get("corpus", {})
            lda = findings.get("lda", {})
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Articles (DB)", corp.get("articles_in_db", "—"))
            c2.metric("LDA documents", corp.get("preprocessed_documents", "—"))
            c3.metric("Vocabulary size", corp.get("vocabulary_size", "—"))
            c4.metric("Optimal K", lda.get("optimal_k", "—"))
            st.caption(findings.get("methodology_note", ""))
        else:
            st.info(
                "`analytical_findings.json` was not found. "
                "Run **Full pipeline** or **Factors + report** first."
            )

        st.subheader("Downloads")
        dcol1, dcol2, dcol3, dcol4 = st.columns(4)
        pdf_path = output_dir / "summary_report.pdf"
        if pdf_path.is_file():
            dcol1.download_button(
                "Download PDF report",
                data=pdf_path.read_bytes(),
                file_name="summary_report.pdf",
                mime="application/pdf",
            )
        json_path = output_dir / "analytical_findings.json"
        if json_path.is_file():
            dcol2.download_button(
                "Download findings (JSON)",
                data=json_path.read_text(encoding="utf-8"),
                file_name="analytical_findings.json",
                mime="application/json",
            )
        txt_path = output_dir / "analytical_findings.txt"
        if txt_path.is_file():
            dcol3.download_button(
                "Download findings (UTF-8 text)",
                data=txt_path.read_bytes(),
                file_name="analytical_findings.txt",
                mime="text/plain",
            )
        zip_bytes = _build_export_zip(project_root, output_dir)
        if zip_bytes:
            dcol4.download_button(
                "Download bundle (ZIP)",
                data=zip_bytes,
                file_name="greek_climate_risk_export.zip",
                mime="application/zip",
            )

    with tab2:
        st.subheader("Article database")
        if db_path.is_file():
            try:
                with sqlite3.connect(str(db_path)) as conn:
                    df = pd.read_sql_query(
                        "SELECT source, COUNT(*) AS n FROM articles GROUP BY source",
                        conn,
                    )
                    st.dataframe(df, use_container_width=True)
                    all_articles = pd.read_sql_query(
                        "SELECT id, date, source, substr(title,1,80) AS title_preview, url FROM articles ORDER BY date DESC",
                        conn,
                    )
                st.caption("All records")
                st.dataframe(all_articles, use_container_width=True)
            except Exception as exc:  # pylint: disable=broad-except
                st.warning(f"Could not read the database: {exc}")
        else:
            st.warning("`articles.db` does not exist yet. Run scraping.")

        st.subheader("Corpus (.pkl)")
        pkl = output_dir / "corpus.pkl"
        if pkl.is_file():
            st.success(f"`{pkl.name}` is present.")
        else:
            st.info("Run preprocessing to build the corpus file.")

    with tab3:
        coh = output_dir / "coherence_scores.csv"
        tw = output_dir / "topic_word_distributions.csv"
        at = output_dir / "article_topic_shares.csv"
        for label, path in [
            ("Coherence by K", coh),
            ("Words per topic", tw),
            ("Topic shares per article", at),
        ]:
            if path.is_file():
                st.markdown(f"**{label}** (`{path.name}`)")
                st.dataframe(pd.read_csv(path), use_container_width=True)
            else:
                st.caption(f"Missing: {path.name}")

    with tab4:
        fd = output_dir / "factors_daily.csv"
        fc = output_dir / "factor_correlations.csv"
        if fd.is_file():
            dfd = pd.read_csv(fd)
            st.markdown("**Daily factor levels**")
            st.dataframe(dfd, use_container_width=True)
            topic_cols = [c for c in dfd.columns if c.startswith("topic_")]
            if topic_cols:
                st.line_chart(
                    dfd.set_index("date")[topic_cols] if "date" in dfd.columns else dfd[topic_cols]
                )
        else:
            st.info(
                "Missing `factors_daily.csv`. Run **Factors + report** or the **full pipeline**."
            )

        if fc.is_file():
            st.markdown("**Correlation matrix**")
            st.dataframe(pd.read_csv(fc), use_container_width=True)

    with tab5:
        for name in [
            "coherence_plot.png",
            "factors_monthly_plot.png",
            "factor_correlations_heatmap.png",
        ]:
            p = output_dir / name
            if p.is_file():
                st.image(str(p), caption=name, use_container_width=True)
        wc = output_dir / "wordclouds"
        if wc.is_dir():
            for img in sorted(wc.glob("*.png")):
                st.image(str(img), caption=img.name, use_container_width=True)


def main() -> None:
    """Render and run Streamlit interface."""
    configure_logging()
    st.set_page_config(page_title="Greek Climate Risk Pipeline", layout="wide")
    st.title("Greek climate risk — pipeline & analytical report")

    project_root = Path(__file__).resolve().parent
    config_path = project_root / "config.yaml"
    config = load_config(config_path)
    output_dir = project_root / config["paths"]["output_dir"]

    st.sidebar.header("Settings")
    config["global"]["start_date"] = st.sidebar.text_input(
        "Start date (YYYY-MM-DD)", value=config["global"]["start_date"]
    )
    config["scraping"]["min_delay_seconds"] = st.sidebar.slider(
        "Min request delay (seconds)",
        1.0,
        5.0,
        float(config["scraping"]["min_delay_seconds"]),
        0.1,
    )
    config["scraping"]["max_delay_seconds"] = st.sidebar.slider(
        "Max request delay (seconds)",
        1.0,
        5.0,
        float(config["scraping"]["max_delay_seconds"]),
        0.1,
    )
    config["lda"]["passes"] = st.sidebar.slider(
        "LDA passes", 5, 50, int(config["lda"]["passes"]), 1
    )

    st.sidebar.caption(
        "Scrape sources (HKKS — Hardouvelis et al.): " + ", ".join(s.name for s in SOURCES)
    )
    seeds_raw = st.sidebar.text_area(
        "Seed search queries (one per line; FMS §3.1 bigrams + Greek equivalents)",
        value="\n".join(config["scraping"]["seed_queries"]),
    )
    config["scraping"]["seed_queries"] = [
        line.strip() for line in seeds_raw.splitlines() if line.strip()
    ]

    keywords_raw = st.sidebar.text_area(
        "Filter keywords (one per line; must match seed rule)",
        value="\n".join(config["preprocessing"]["filter_keywords"]),
    )
    config["preprocessing"]["filter_keywords"] = [
        line.strip() for line in keywords_raw.splitlines() if line.strip()
    ]

    if st.sidebar.button("Save config"):
        save_config(config_path, config)
        st.sidebar.success("Saved `config.yaml`.")

    st.markdown(
        "Run individual stages or the full pipeline. After LDA, use **Factors + report** "
        "to generate the PDF, JSON, text export, and tables."
    )
    db_path = project_root / config["paths"]["database"]
    output_dir.mkdir(parents=True, exist_ok=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        run_scrape = st.button("Scraping")
    with c2:
        run_pre_lda = st.button("Preprocess + LDA")
    with c3:
        run_factors_report = st.button("Factors + report")
    with c4:
        run_all = st.button("Full pipeline")
    with c5:
        regen_report = st.button("Report only (from disk)")

    if run_scrape:
        try:
            with st.spinner("Scraping..."):
                run_scraping_pipeline(config=config, db_path=db_path)
            st.success("Scraping finished.")
        except Exception as exc:  # pylint: disable=broad-except
            st.error(f"Scraping failed: {exc}")

    if run_pre_lda:
        try:
            with st.spinner("Preprocessing and LDA..."):
                corpus_artifact = run_preprocessing(config=config, db_path=db_path, output_dir=output_dir)
                run_lda_pipeline(config=config, corpus_artifact=corpus_artifact, output_dir=output_dir)
            st.success("Preprocessing and LDA finished.")
        except Exception as exc:  # pylint: disable=broad-except
            st.error(f"Preprocessing / LDA failed: {exc}")
            st.info("For a small corpus, lower `min_document_frequency` in `config.yaml`.")

    if run_factors_report:
        try:
            with st.spinner("Building factors and analytical report..."):
                path = finalize_pipeline_after_lda(config=config, project_root=project_root)
            if path:
                st.success(f"Done. PDF: {path}")
            else:
                st.error("Missing LDA/corpus files. Run **Preprocess + LDA** first.")
        except Exception as exc:  # pylint: disable=broad-except
            st.error(f"Failed: {exc}")

    if run_all:
        try:
            with st.spinner("Full pipeline (may take several minutes)..."):
                run_full_pipeline(config=config, project_root=project_root)
            st.success("Full pipeline finished.")
        except Exception as exc:  # pylint: disable=broad-except
            st.error(f"Failed: {exc}")

    if regen_report:
        try:
            with st.spinner("Rebuilding report..."):
                path = rebuild_report_from_disk(config=config, project_root=project_root)
            if path:
                st.success(f"Report updated: {path}")
            else:
                st.error("Missing files (corpus, LDA, or factors).")
        except Exception as exc:  # pylint: disable=broad-except
            st.error(f"Failed: {exc}")

    render_full_dashboard(project_root, output_dir, db_path)


if __name__ == "__main__":
    main()
