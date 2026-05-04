"""Entry point for the Greek climate risk measurement pipeline."""

from __future__ import annotations

import logging
from pathlib import Path

from greek_climate_risk.config import load_config
from greek_climate_risk.factors.construct import build_climate_factors
from greek_climate_risk.lda.modeling import run_lda_pipeline
from greek_climate_risk.output.report import build_pdf_report
from greek_climate_risk.preprocessing.pipeline import run_preprocessing
from greek_climate_risk.scrapers.pipeline import run_scraping_pipeline


def configure_logging() -> None:
    """Configure root logger for the full pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def main() -> None:
    """Run all processing phases from scraping to report generation."""
    configure_logging()
    project_root = Path(__file__).resolve().parent
    config = load_config(project_root / "config.yaml")

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


if __name__ == "__main__":
    main()
