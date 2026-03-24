# CLO Equity Monthly Return Estimation Model

A Streamlit application that estimates CLO equity monthly returns using a multi-factor model and calibrates against Flat Rock quarterly actuals.

## Output Series

- **Series 1 — Estimated Monthly Return**: Pure model output (carry + levered capital return with smoothing)
- **Series 2 — Actual-Extended Monthly Return**: True-up applied where Flat Rock quarterly returns are available; estimated thereafter

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deployment

Deploy directly to [Streamlit Community Cloud](https://streamlit.io/cloud) by connecting your GitHub repo.

## Input File

Upload an Excel file with an "Inputs" sheet containing:

| Column | Content |
|--------|---------|
| A | Month-end dates |
| B | 1m SOFR (decimal) |
| C | Loan spread proxy (decimal) |
| D | Loan price level |
| E | Flat Rock quarterly return (quarter-ends only) |

## Features

- Interactive parameter tuning via sidebar sliders
- Automated parameter optimisation (differential evolution)
- Proportional and end-loaded true-up methods
- Plotly charts with cumulative return index
- Excel export for monthly and quarterly tables
