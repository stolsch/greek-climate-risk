"""Construct daily climate risk factors from article-topic shares."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from greek_climate_risk.lda.modeling import LdaArtifact

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class FactorArtifact:
    """Outputs from the factor construction stage."""

    factors_daily_path: Path
    monthly_plot_path: Path
    correlations_path: Path
    correlation_heatmap_path: Path


def build_climate_factors(
    config: dict[str, Any],
    db_path: Path,
    lda_artifact: LdaArtifact,
    output_dir: Path,
) -> FactorArtifact:
    """Build level factors as daily sums of article topic shares."""
    _ = db_path
    topic_df = pd.read_csv(lda_artifact.article_topic_path)
    topic_cols = [col for col in topic_df.columns if col.startswith("topic_")]
    if topic_df.empty:
        raise ValueError("No article-topic rows found; cannot build factors.")

    topic_df["date"] = pd.to_datetime(topic_df["date"])
    daily = topic_df.groupby("date", as_index=False)[topic_cols].sum()
    factors_daily_path = output_dir / "factors_daily.csv"
    daily.to_csv(factors_daily_path, index=False)

    monthly = daily.set_index("date").resample("MS").mean().reset_index()
    monthly_plot_path = output_dir / "factors_monthly_plot.png"
    plt.figure(figsize=(12, 6))
    for col in topic_cols:
        plt.plot(monthly["date"], monthly[col], linewidth=1.5, label=col)
    plt.title("Monthly Averages of Climate Risk Topic Factors")
    plt.xlabel("Date")
    plt.ylabel("Average Factor Level")
    plt.legend(loc="upper left", ncol=2, fontsize=8)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(monthly_plot_path, dpi=300)
    plt.close()

    correlation_df = daily[topic_cols].corr()
    correlations_path = output_dir / "factor_correlations.csv"
    correlation_df.to_csv(correlations_path)
    heatmap_path = output_dir / "factor_correlations_heatmap.png"
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_df, cmap="coolwarm", center=0.0, annot=False, square=True)
    plt.title("Pairwise Correlations Across Topic Factors")
    plt.tight_layout()
    plt.savefig(heatmap_path, dpi=300)
    plt.close()

    LOGGER.info("Constructed factor levels and correlation table.")
    return FactorArtifact(
        factors_daily_path=factors_daily_path,
        monthly_plot_path=monthly_plot_path,
        correlations_path=correlations_path,
        correlation_heatmap_path=heatmap_path,
    )
