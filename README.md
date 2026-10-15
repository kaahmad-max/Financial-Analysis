# Quarterly Financial Analysis

This project
applies three methods to published quarterly filings from a real
listed electronics manufacturer, Apple Inc. The goal is to collect the numbers, model them, and present the
results as figures.

The three methods from the paper are:

1. A review of historical quarterly performance with year-over-year growth.
2. Revenue forecasting with a seasonal ARIMA model.
3. Segmentation with k-means.

## Data

The script reads data from the SEC EDGAR XBRL API, which publishes the figures
companies report in their 10-Q and 10-K filings. Three series are collected for
the window 2018 Q1 to 2023 Q4 (24 quarters):

- Revenue
- Net income
- Operating cash flow

Companies report income-statement and cash-flow items as amounts accumulated
since the start of the fiscal year. A second-quarter filing, for example, holds
a six-month total. The script recovers the figure for each single quarter by
grouping the facts that belong to one fiscal year and subtracting each
cumulative value from the previous one.

The raw API responses are saved in `data_cache/`, so the script runs offline
after the first download.

## Requirements

Python 3.9 or later, with the following packages:

```bash
pip install numpy pandas matplotlib statsmodels scikit-learn
```

## Running the script

The SEC asks API users to identify themselves. Open `financial_analysis.py` and
set the `USER_AGENT` constant near the top to a contact address you control.

Then run:

```bash
python3 financial_analysis.py
```

The script writes three PNG files to `figures/` and prints a text summary of the
numbers behind each one.

## Output

| File | Contents |
| --- | --- |
| `figures/01_quarterly_performance.png` | Bars for revenue, net income, and operating cash flow each quarter, with a line for year-over-year revenue growth. |
| `figures/02_revenue_forecast.png` | Actual revenue, the model's back-tested forecast, a forward projection with an 80% interval, and a panel of per-quarter forecast errors. |
| `figures/03_operating_regimes.png` | A scatter of quarterly revenue against net margin, coloured by the k-means group, with each group's centre marked. |

## How the code works

The script is organised into sections, read from top to bottom.

### Configuration

A `Company` record holds the name, ticker, and SEC CIK number. Constants set the
date window, the number of quarters held back for testing, and the number of
quarters to project forward. File paths and the list of XBRL tags for each
metric are defined here.

### Visual style

`set_style` applies a consistent look to every figure: a sans-serif font, a
light horizontal grid behind the data, and thin axes. `style_axes` removes the
top and right borders from a given plot. `titled` adds a left-aligned title and
subtitle. `billions` formats large values in units of a billion.

### Data acquisition

- `fetch_concept` downloads one metric for one company from EDGAR and caches the
  response on disk.
- `quarterly_series` turns the cumulative year-to-date facts into a series of
  single-quarter values using the differencing method described above. It
  anchors each quarter's calendar label on the period midpoint, which keeps the
  labels correct when a company's fiscal quarter ends drift across month
  boundaries.
- `load_dataset` combines the three metrics into one table, trims it to the
  chosen window, and adds net margin and year-over-year growth columns.

### Forecasting

- `select_order` searches a small set of candidate ARIMA orders and returns the
  one with the lowest AIC score.
- `forecast_revenue` works on the logarithm of revenue. It trains on the data up
  to the test window, forecasts the held-out quarters, and compares the forecast
  with the actual figures to measure error. It then refits on the full history
  and projects the next four quarters with an 80% interval.

### Segmentation

`segment_quarters` scales revenue and net margin to a common range, groups the
quarters into three clusters with k-means, and relabels the clusters by their
average revenue so the ordering from lower to higher is stable across runs.

### Figures and reporting

`plot_performance`, `plot_forecast`, and `plot_segments` each build one figure.
`summarise` prints the key numbers to the terminal. `main` runs the whole
sequence: load the data, fit the model, run the clustering, draw the figures,
and print the summary.

## Changing the company

To analyse a different filer, edit the `COMPANY` line at the top of the script
with its name, ticker, and CIK number. The CIK is listed on the company's page
at sec.gov. You can also adjust `START` and `END` to change the date window.

## Notes

- Forecasting real revenue carries more error than the smooth sample data in the
  paper. The back-test error on Apple revenue comes to about 2% on average.
- Public filings do not contain customer-level detail. The segmentation step
  groups the company's quarters by scale and profitability, which is the closest
  match to the paper's customer segmentation that the available data supports.
- The script suppresses model-fitting warnings so the terminal output stays
  readable.
