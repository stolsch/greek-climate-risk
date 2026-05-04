# Deploying on Streamlit Community Cloud

## Dependencies and Python version

The pipeline uses **scikit-learn** for LDA (no `gensim` C extensions). `requirements.txt` pins **matplotlib**, **Pillow**, **NumPy**, **SciPy**, **pandas**, **spaCy**, **wordcloud**, and **Streamlit** to versions that publish **cp314** wheels where needed, so Community Cloud does not fall back to compiling **Pillow** (which fails without system `zlib` headers).

`runtime.txt` in this repo is for platforms that honor it (e.g. some PaaS). **Streamlit Community Cloud does not use `runtime.txt`** to pick the Python version; use the deploy UI if you need a specific version.

## Redeploying after Git changes

Push to GitHub, then open the app on Streamlit Cloud and trigger a rebuild, or use **Manage app** → reboot as needed.

For Python version selection and limitations on changing it in place, see the official docs: [Upgrade your app’s Python version on Community Cloud](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app/upgrade-python).
