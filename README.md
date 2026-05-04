# Greek Climate Risk Pipeline

Reproducible Python pipeline to replicate the Faccini, Matin & Skiadopoulos (2023) climate risk measurement workflow, adapted to Greek newspaper articles.

## Methodology alignment

- **Media (Hardouvelis et al., HKKS):** Scraping is restricted to the **four** outlets used in the HKKS Greek news framework: **Kathimerini**, **To Vima**, **Ta Nea**, **Naftemporiki** (`greek_climate_risk/scrapers/sources.py`). No other sites are included.
- **Article filter (FMS 2023, §3.1):** As in the original Reuters study, an article enters the LDA corpus only if its text contains at least one of the bigrams **`climate change`** or **`global warming`**, or the Greek equivalents **`κλιματική αλλαγή`** and **`παγκόσμια υπερθέρμανση`** (`config.yaml`: `seed_queries` and `filter_keywords` match this rule).
- **Calendar start:** `global.start_date` defaults to **`2000-01-01`**, consistent with the FMS sample start. Deep historical coverage still depends on each outlet’s public search/archive; for print-scale history, use licensed archives or library digitisation.

## Project Structure

```text
greek_climate_risk/
├── greek_climate_risk/
│   ├── scrapers/
│   ├── preprocessing/
│   ├── lda/
│   ├── factors/
│   └── output/
├── config.yaml
├── main.py
├── requirements.txt
└── README.md
```

## Setup

1. Create Python 3.11+ virtual environment.
2. Install dependencies:
   - `pip install -r requirements.txt`  
     This includes the **`el_core_news_sm`** pipeline wheel (Greek), so preprocessing works on fresh machines and on Streamlit Cloud without a separate download step.
3. (Optional) If you use a minimal install without that wheel, install the model manually:
   - `python -m spacy download el_core_news_sm`

## Run

From project root:

`python main.py`

## Streamlit GUI

To use the graphical interface:

`streamlit run app.py`

The GUI lets you:

- Tune key config values from the sidebar
- Run scraping only, preprocessing + LDA, or the full pipeline
- Preview generated figures and confirm report creation

## Streamlit Community Cloud

LDA uses **scikit-learn** (wheel-only), so installs succeed on **Python 3.14** as well as 3.11–3.13.

**Streamlit Cloud does not read `runtime.txt`.** If you need a specific interpreter, choose it under **Advanced settings** when you deploy.

See [STREAMLIT_DEPLOY.md](STREAMLIT_DEPLOY.md) for notes on redeploying and Python selection.

Dependency versions are pinned in `requirements.txt` for reproducible installs.

## Outputs

The pipeline writes all artifacts under `output/`:

- `corpus.pkl`
- `coherence_plot.png`
- `lda_model/model.joblib` (fitted `LatentDirichletAllocation` plus vocabulary)
- `topic_word_distributions.csv`
- `article_topic_shares.csv`
- `topic_labels.json`
- `wordclouds/topic_*.png`
- `factors_daily.csv`
- `factors_monthly_plot.png`
- `factor_correlations.csv`
- `factor_correlations_heatmap.png`
- `summary_report.pdf` (πολυσέλιδη αναλυτική αναφορά με πίνακες και ενσωματωμένα γραφήματα)
- `analytical_findings.json` / `analytical_findings.txt` (δομημένα και αναγνώσιμα ευρήματα για παράδοση)
- `coherence_scores.csv`

## Notes

- Scrapers respect `robots.txt`, use retry logic, and wait 1-2 seconds between requests.
- The PDF report and `analytical_findings.json` snapshot document HKKS outlets, FMS-style keywords, and `start_date` for reproducibility.
- Factors are implemented as **levels** (daily sums of article-level topic shares), matching the paper's construction principle.
- You can tune keywords, date range, and topic search grid in `config.yaml`.
