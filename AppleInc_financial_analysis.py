"""Quarterly financial analysis of a listed electronics manufacturer.

Pulls real quarterly fundamentals (revenue, net income, operating cash flow)
from the SEC EDGAR XBRL API, then runs three analyses:

    1. Historical performance and year-over-year growth.
    2. Revenue forecasting with a seasonal ARIMA model (back-test + forward view).
    3. Segmentation of quarters into operating regimes with k-means.

Each analysis is rendered as a standalone figure. The company is Apple Inc.
by default; change COMPANY below to analyse any other SEC filer.

Requires: numpy, pandas, matplotlib, statsmodels, scikit-learn.
"""

from __future__ import annotations

import json
import urllib.request
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Company:
    name: str
    ticker: str
    cik: int


COMPANY = Company(name="Apple Inc.", ticker="AAPL", cik=320193)

START, END = pd.Period("2018Q1", "Q"), pd.Period("2023Q4", "Q")
TEST_QUARTERS = 4          # most recent quarters held out to back-test the model
FORECAST_QUARTERS = 4      # quarters projected beyond the last observation

HERE = Path(__file__).resolve().parent
CACHE_DIR = HERE / "data_cache"
FIG_DIR = HERE / "figures"

# SEC asks API users to identify themselves; set your own contact address here.
USER_AGENT = "financial-analysis-research contact@example.com"

REVENUE_TAGS = (
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "Revenues",
    "SalesRevenueNet",
)
NET_INCOME_TAGS = ("NetIncomeLoss",)
CASH_FLOW_TAGS = ("NetCashProvidedByUsedInOperatingActivities",)

# --------------------------------------------------------------------------- #
# Visual style
# --------------------------------------------------------------------------- #

INK = "#1A1D24"
MUTED = "#6B7280"
GRID = "#E8EAED"
REVENUE_C = "#2F4858"
INCOME_C = "#2A9D8F"
CASHFLOW_C = "#E4B363"
ACCENT = "#E05B49"
CLUSTER_C = ["#3D5A80", "#9DB4C0", "#EE6C4D"]


def set_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 200,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.bbox": "tight",
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 11,
        "text.color": INK,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": MUTED,
        "axes.titlecolor": INK,
        "axes.linewidth": 0.9,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": GRID,
        "grid.linewidth": 0.9,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.frameon": False,
        "legend.fontsize": 9.5,
    })


def style_axes(ax) -> None:
    """Strip chart junk: no top/right spines, grid behind the data."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def titled(ax, title: str, subtitle: str = "") -> None:
    ax.set_title(title, loc="left", fontsize=15, fontweight="bold", pad=18)
    if subtitle:
        ax.text(0, 1.02, subtitle, transform=ax.transAxes, ha="left", va="bottom",
                fontsize=10, color=MUTED)


def billions(x, _=None) -> str:
    return f"{x / 1e9:,.0f}"


# --------------------------------------------------------------------------- #
# Data acquisition (SEC EDGAR XBRL)
# --------------------------------------------------------------------------- #

def fetch_concept(cik: int, tag: str) -> dict | None:
    """Return a company-concept document, caching the raw JSON locally."""
    CACHE_DIR.mkdir(exist_ok=True)
    cache = CACHE_DIR / f"{cik:010d}_{tag}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    url = f"https://data.sec.gov/api/xbrl/companyconcept/CIK{cik:010d}/us-gaap/{tag}.json"
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = response.read()
    except Exception:
        return None
    cache.write_bytes(payload)
    return json.loads(payload)


def quarterly_series(cik: int, tags: tuple[str, ...]) -> pd.Series:
    """Build a per-quarter series from cumulative year-to-date XBRL facts.

    Filers report income-statement and cash-flow items as amounts accumulated
    since the start of the fiscal year. Facts belonging to one fiscal year share
    the same period start, so grouping on the start date and differencing the
    sorted cumulative values recovers the discrete quarterly figure.
    """
    facts: dict[tuple, tuple] = {}
    for tag in tags:
        document = fetch_concept(cik, tag)
        if not document:
            continue
        for fact in document.get("units", {}).get("USD", []):
            if "start" not in fact or "end" not in fact:
                continue
            start = pd.Timestamp(fact["start"]).date()
            end = pd.Timestamp(fact["end"]).date()
            months = round((end - start).days / 30.4)
            if months not in (3, 6, 9, 12):
                continue
            key = (start, end)
            if key not in facts or fact["filed"] > facts[key][1]:
                facts[key] = (fact["val"], fact["filed"], months)

    groups: dict = {}
    for (start, end), (value, _, months) in facts.items():
        groups.setdefault(start, []).append((end, months, value))

    records = []
    for cumulative in groups.values():
        cumulative.sort(key=lambda item: item[0])
        is_full_year = any(months >= 11 for _, months, _ in cumulative)
        if not (is_full_year or len(cumulative) >= 2):
            continue  # standalone single-quarter fact: duplicated by the YTD chain
        previous = 0.0
        for end, _, value in cumulative:
            # Fiscal quarter-ends drift (e.g. Apple closes near 1 Apr / 1 Jul), so
            # anchor the calendar quarter on the period midpoint, not the end date.
            midpoint = pd.Timestamp(end) - pd.Timedelta(days=46)
            records.append((midpoint.to_period("Q"), value - previous))
            previous = value

    series = (pd.Series(dict(records))
              .groupby(level=0).last()
              .sort_index())
    return series


def load_dataset(company: Company) -> pd.DataFrame:
    frame = pd.DataFrame({
        "revenue": quarterly_series(company.cik, REVENUE_TAGS),
        "net_income": quarterly_series(company.cik, NET_INCOME_TAGS),
        "operating_cash_flow": quarterly_series(company.cik, CASH_FLOW_TAGS),
    })
    frame = frame.loc[START:END].dropna()
    if frame.empty:
        raise RuntimeError("No quarterly data returned; check connectivity or the CIK.")
    frame["net_margin"] = frame["net_income"] / frame["revenue"]
    for column in ("revenue", "net_income", "operating_cash_flow"):
        frame[f"{column}_yoy"] = frame[column].pct_change(4) * 100
    return frame


# --------------------------------------------------------------------------- #
# Modelling
# --------------------------------------------------------------------------- #

def select_order(endog: np.ndarray) -> tuple:
    """Pick the seasonal ARIMA order minimising AIC over a small candidate grid."""
    best_order, best_aic = None, np.inf
    for p in range(3):
        for q in range(3):
            for seasonal_p in range(2):
                for seasonal_d in range(2):
                    for seasonal_q in range(2):
                        order = (p, 1, q)
                        seasonal = (seasonal_p, seasonal_d, seasonal_q, 4)
                        try:
                            model = SARIMAX(endog, order=order, seasonal_order=seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)
                            aic = model.fit(disp=False).aic
                        except Exception:
                            continue
                        if np.isfinite(aic) and aic < best_aic:
                            best_order, best_aic = (order, seasonal), aic
    return best_order or ((1, 1, 1), (0, 1, 1, 4))


def forecast_revenue(revenue: pd.Series) -> dict:
    """Back-test on held-out quarters and project the near-term trajectory."""
    log_revenue = np.log(revenue.values)
    order, seasonal = select_order(log_revenue)

    train = log_revenue[:-TEST_QUARTERS]
    fitted = SARIMAX(train, order=order, seasonal_order=seasonal,
                     enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    prediction = fitted.get_forecast(TEST_QUARTERS)
    predicted = np.exp(prediction.predicted_mean)
    actual = revenue.values[-TEST_QUARTERS:]

    full = SARIMAX(log_revenue, order=order, seasonal_order=seasonal,
                   enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    projection = full.get_forecast(FORECAST_QUARTERS)
    future_mean = np.exp(projection.predicted_mean)
    confidence = np.exp(projection.conf_int(alpha=0.2))

    future_index = pd.period_range(revenue.index[-1] + 1, periods=FORECAST_QUARTERS, freq="Q")
    relative_error = np.abs(predicted - actual) / actual * 100

    return {
        "order": order,
        "seasonal": seasonal,
        "aic": full.aic,
        "test_index": revenue.index[-TEST_QUARTERS:],
        "actual": actual,
        "predicted": predicted,
        "relative_error": relative_error,
        "future_index": future_index,
        "future_mean": future_mean,
        "future_lower": confidence[:, 0],
        "future_upper": confidence[:, 1],
    }


def segment_quarters(frame: pd.DataFrame, n_clusters: int = 3) -> pd.DataFrame:
    """Group quarters into operating regimes by scale and profitability.

    Public filings carry no customer-level detail, so segmentation is applied to
    the quarterly observations themselves: revenue measures scale, net margin
    measures profitability. Clusters are relabelled by mean revenue so that the
    ordering (lower / mid / peak) is stable across runs.
    """
    features = StandardScaler().fit_transform(frame[["revenue", "net_margin"]])
    raw = KMeans(n_clusters=n_clusters, n_init=10, random_state=0).fit_predict(features)

    order = frame.groupby(raw)["revenue"].mean().sort_values().index
    remap = {old: new for new, old in enumerate(order)}
    result = frame.copy()
    result["regime"] = [remap[label] for label in raw]
    return result


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #

def quarter_labels(index: pd.PeriodIndex) -> list[str]:
    """Quarter on the first line; the year appears only under Q1 to avoid clutter."""
    return [f"Q{p.quarter}\n{p.year}" if p.quarter == 1 else f"Q{p.quarter}" for p in index]


def plot_performance(frame: pd.DataFrame, company: Company) -> Path:
    labels = quarter_labels(frame.index)
    x = np.arange(len(frame))
    width = 0.27

    fig, ax = plt.subplots(figsize=(13, 6.2))
    style_axes(ax)
    ax.bar(x - width, frame["revenue"], width, color=REVENUE_C, label="Revenue")
    ax.bar(x, frame["net_income"], width, color=INCOME_C, label="Net income")
    ax.bar(x + width, frame["operating_cash_flow"], width, color=CASHFLOW_C,
           label="Operating cash flow")
    ax.yaxis.set_major_formatter(FuncFormatter(billions))
    ax.set_ylabel("USD billions")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.margins(x=0.01)

    growth = ax.twinx()
    growth.spines["top"].set_visible(False)
    growth.plot(x, frame["revenue_yoy"], color=ACCENT, lw=2, marker="o",
                markersize=4, label="Revenue YoY growth")
    growth.axhline(0, color=MUTED, lw=0.8, ls=(0, (4, 4)))
    growth.set_ylabel("Year-over-year growth (%)", color=ACCENT)
    growth.tick_params(axis="y", colors=ACCENT, length=0)
    growth.grid(False)

    bars, bar_labels = ax.get_legend_handles_labels()
    lines, line_labels = growth.get_legend_handles_labels()
    ax.legend(bars + lines, bar_labels + line_labels, loc="upper left",
              ncol=4, bbox_to_anchor=(0, 1.0))

    first, last = frame.index[0], frame.index[-1]
    titled(ax, "Quarterly performance and year-over-year growth",
           f"{company.name} ({company.ticker}) · "
           f"{first.year} Q{first.quarter} – {last.year} Q{last.quarter}")
    fig.tight_layout()
    path = FIG_DIR / "01_quarterly_performance.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_forecast(revenue: pd.Series, result: dict, company: Company) -> Path:
    history_idx = np.arange(len(revenue))
    test_positions = history_idx[-TEST_QUARTERS:]
    future_positions = np.arange(len(revenue), len(revenue) + FORECAST_QUARTERS)

    fig, (ax, err) = plt.subplots(
        2, 1, figsize=(12, 7.4), height_ratios=[3, 1], sharex=True,
        gridspec_kw={"hspace": 0.12})

    style_axes(ax)
    ax.plot(history_idx, revenue.values, color=REVENUE_C, lw=2, marker="o",
            markersize=4, label="Actual revenue")
    ax.plot(test_positions, result["predicted"], color=ACCENT, lw=2, ls="--",
            marker="s", markersize=5, label="Back-tested forecast")
    ax.plot(future_positions, result["future_mean"], color=ACCENT, lw=2, ls=":",
            marker="D", markersize=5, label="Forward projection")
    ax.fill_between(future_positions, result["future_lower"], result["future_upper"],
                    color=ACCENT, alpha=0.12, label="80% interval")
    ax.axvline(test_positions[0] - 0.5, color=MUTED, lw=0.9, ls=(0, (4, 4)))
    ax.text(test_positions[0] - 0.4, ax.get_ylim()[1], " back-test", color=MUTED,
            fontsize=9, va="top")
    ax.yaxis.set_major_formatter(FuncFormatter(billions))
    ax.set_ylabel("Revenue (USD billions)")
    ax.legend(loc="upper left", ncol=2)
    p, d, q = result["order"]
    sp, sd, sq, s = result["seasonal"]
    titled(ax, "Revenue forecast with a seasonal ARIMA model",
           f"{company.name} ({company.ticker}) · ARIMA({p},{d},{q})("
           f"{sp},{sd},{sq})[{s}] · mean back-test error "
           f"{result['relative_error'].mean():.1f}%")

    style_axes(err)
    bars = err.bar(test_positions, result["relative_error"], width=0.6, color=INCOME_C)
    err.bar_label(bars, labels=[f"{v:.1f}%" for v in result["relative_error"]],
                  padding=3, fontsize=9, color=INK)
    err.set_ylabel("Abs. error (%)")
    err.set_ylim(0, max(result["relative_error"].max() * 1.35, 1))

    all_idx = np.concatenate([history_idx, future_positions])
    labels = quarter_labels(revenue.index.append(result["future_index"]))
    err.set_xticks(all_idx)
    err.set_xticklabels(labels, rotation=0)
    err.margins(x=0.01)

    path = FIG_DIR / "02_revenue_forecast.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_segments(frame: pd.DataFrame, company: Company) -> Path:
    names = ["Lower-volume quarters", "Mid-volume quarters", "Peak (holiday) quarters"]
    fig, ax = plt.subplots(figsize=(10.5, 7))
    style_axes(ax)

    for regime, group in frame.groupby("regime"):
        ax.scatter(group["revenue"], group["net_margin"] * 100,
                   s=130, color=CLUSTER_C[regime], edgecolor="white", linewidth=1.2,
                   alpha=0.9, label=names[regime], zorder=3)
        centroid = (group["revenue"].mean(), group["net_margin"].mean() * 100)
        ax.scatter(*centroid, s=360, color=CLUSTER_C[regime], marker="X",
                   edgecolor=INK, linewidth=1.1, zorder=4)

    for period, row in frame.iterrows():
        if period.quarter == 4:  # label the holiday quarters for orientation
            ax.annotate(f"{period.year} Q4", (row["revenue"], row["net_margin"] * 100),
                        textcoords="offset points", xytext=(8, 6),
                        fontsize=8.5, color=MUTED)

    ax.xaxis.set_major_formatter(FuncFormatter(billions))
    ax.set_xlabel("Quarterly revenue (USD billions)")
    ax.set_ylabel("Net margin (%)")
    ax.legend(loc="lower right", title="Operating regime", title_fontsize=10)
    titled(ax, "Operating regimes from k-means segmentation",
           f"{company.name} ({company.ticker}) · quarters grouped by scale and profitability "
           "(X = cluster centre)")
    fig.tight_layout()
    path = FIG_DIR / "03_operating_regimes.png"
    fig.savefig(path)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #

def summarise(frame: pd.DataFrame, result: dict, segmented: pd.DataFrame,
              company: Company) -> None:
    line = "-" * 68
    print(f"\n{company.name} ({company.ticker}) — quarterly analysis")
    print(line)
    print(f"Window           : {frame.index[0]} to {frame.index[-1]} "
          f"({len(frame)} quarters)")
    print(f"Revenue          : {frame['revenue'].iloc[0]/1e9:,.1f}B -> "
          f"{frame['revenue'].iloc[-1]/1e9:,.1f}B")
    print(f"Avg net margin   : {frame['net_margin'].mean()*100:,.1f}%")
    print(f"Latest YoY growth: revenue {frame['revenue_yoy'].iloc[-1]:,.1f}%, "
          f"net income {frame['net_income_yoy'].iloc[-1]:,.1f}%")
    print(line)

    p, d, q = result["order"]
    sp, sd, sq, s = result["seasonal"]
    print(f"Forecast model   : ARIMA({p},{d},{q})({sp},{sd},{sq})[{s}]  AIC {result['aic']:.1f}")
    print(f"Back-test error  : mean {result['relative_error'].mean():.2f}%, "
          f"max {result['relative_error'].max():.2f}%")
    for period, actual, predicted, error in zip(
            result["test_index"], result["actual"], result["predicted"],
            result["relative_error"]):
        print(f"  {period}: actual {actual/1e9:6.1f}B  "
              f"forecast {predicted/1e9:6.1f}B  error {error:4.1f}%")
    print("Forward view     :")
    for period, mean in zip(result["future_index"], result["future_mean"]):
        print(f"  {period}: {mean/1e9:6.1f}B")
    print(line)

    names = ["Lower-volume", "Mid-volume", "Peak (holiday)"]
    print("Operating regimes:")
    for regime, group in segmented.groupby("regime"):
        print(f"  {names[regime]:15s}: {len(group)} quarters, "
              f"avg revenue {group['revenue'].mean()/1e9:5.1f}B, "
              f"avg margin {group['net_margin'].mean()*100:4.1f}%")
    print(line)


def main() -> None:
    set_style()
    FIG_DIR.mkdir(exist_ok=True)

    frame = load_dataset(COMPANY)
    result = forecast_revenue(frame["revenue"])
    segmented = segment_quarters(frame)

    paths = [
        plot_performance(frame, COMPANY),
        plot_forecast(frame["revenue"], result, COMPANY),
        plot_segments(segmented, COMPANY),
    ]

    summarise(frame, result, segmented, COMPANY)
    print("Figures written:")
    for path in paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
