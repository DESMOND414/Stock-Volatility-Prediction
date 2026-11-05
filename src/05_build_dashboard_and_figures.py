"""
Builds high-resolution static thesis visualisations (`dashboard/images/*.png`)
and a self-contained interactive HTML companion dashboard (`dashboard/interactive_dashboard.html`)
for "Stock Market Volatility Classification Using News Sentiment" (MSc Thesis, Maynooth University).
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib_cache")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC_DIR = ROOT / "data" / "processed"
DASH_DIR = ROOT / "dashboard"
IMG_DIR = DASH_DIR / "images"
IMG_DIR.mkdir(parents=True, exist_ok=True)

# Executive Quant / FinTech Dark Theme Palette
BG_DARK = "#0b1120"
PANEL_BG = "#131c31"
BORDER_CLR = "#1e293b"
TEXT_MAIN = "#f8fafc"
TEXT_MUTED = "#94a3b8"

CLR_NEG = "#ef4444"     # Crimson Red
CLR_NEU = "#f59e0b"     # Amber Gold
CLR_POS = "#10b981"     # Emerald Green
CLR_CNN = "#38bdf8"     # Sky Blue
CLR_XGB = "#a855f7"     # Vivid Purple
CLR_DT = "#64748b"      # Slate Muted


def style_ax(ax: plt.Axes, title: str, xlabel: str = "", ylabel: str = "") -> None:
    ax.set_facecolor(PANEL_BG)
    ax.set_title(title, color=TEXT_MAIN, fontsize=12, fontweight="bold", loc="left", pad=11)
    if xlabel:
        ax.set_xlabel(xlabel, color=TEXT_MUTED, fontsize=9.5, labelpad=6)
    if ylabel:
        ax.set_ylabel(ylabel, color=TEXT_MUTED, fontsize=9.5, labelpad=6)
    ax.tick_params(colors=TEXT_MUTED, labelsize=8.8)
    for spine in ax.spines.values():
        spine.set_color(BORDER_CLR)
        spine.set_linewidth(1.1)
    ax.grid(True, linestyle="--", alpha=0.20, color=TEXT_MUTED)


def plot_finbert_sentiment_distribution(firm_df: pd.DataFrame) -> None:
    """
    Generates `dashboard/images/finbert_sentiment_distribution.png`
    (Thesis Figures 5.1 & 5.2: Overall & Per-Company FinBERT Sentiment Distribution).
    """
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(15.5, 6.2), facecolor=BG_DARK, gridspec_kw={"width_ratios": [1.0, 1.55]}
    )
    fig.subplots_adjust(wspace=0.22, top=0.83, bottom=0.14, left=0.06, right=0.97)

    fig.suptitle(
        "FINBERT FINANCIAL NEWS SENTIMENT DISTRIBUTION (55,349 NASDAQ-100 ARTICLES)",
        color=TEXT_MAIN, fontsize=14.5, fontweight="bold", x=0.06, ha="left", y=0.95
    )

    # Left: Dataset-Level Distribution (Thesis Figure 5.1)
    totals = [
        int(firm_df["Negative_Articles"].sum()),
        int(firm_df["Neutral_Articles"].sum()),
        int(firm_df["Positive_Articles"].sum()),
    ]
    labels = ["Negative", "Neutral", "Positive"]
    colors = [CLR_NEG, CLR_NEU, CLR_POS]
    bars = ax1.bar(labels, totals, color=colors, width=0.56, edgecolor="white", linewidth=0.6, alpha=0.9)
    style_ax(
        ax1,
        "Dataset-Level Sentiment Distribution (Fig. 5.1)",
        xlabel="FinBERT Sentiment Class",
        ylabel="Number of News Articles",
    )
    ax1.yaxis.set_major_formatter(mtick.StrMethodFormatter("{x:,.0f}"))
    ax1.set_ylim(0, max(totals) * 1.16)
    total_sum = sum(totals)
    for bar, val in zip(bars, totals):
        pct = val / total_sum * 100
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            val + 800,
            f"{val:,}\n({pct:.1f}%)",
            ha="center", va="bottom", color=TEXT_MAIN, fontsize=9, fontweight="bold"
        )

    # Right: Firm-Level Sentiment Breakdown (Thesis Figure 5.2)
    firms = firm_df["Stock_symbol"].tolist()
    x = np.arange(len(firms))
    w = 0.26
    ax2.bar(x - w, firm_df["Negative_Articles"], width=w, color=CLR_NEG, label="Negative", alpha=0.9)
    ax2.bar(x,     firm_df["Neutral_Articles"],  width=w, color=CLR_NEU, label="Neutral",  alpha=0.9)
    ax2.bar(x + w, firm_df["Positive_Articles"], width=w, color=CLR_POS, label="Positive", alpha=0.9)

    style_ax(
        ax2,
        "Firm-Level Sentiment Distribution Across NASDAQ-100 Constituents (Fig. 5.2)",
        xlabel="NASDAQ-100 Stock Ticker",
        ylabel="Number of News Articles",
    )
    ax2.set_xticks(x)
    ax2.set_xticklabels(firms, fontweight="bold")
    ax2.yaxis.set_major_formatter(mtick.StrMethodFormatter("{x:,.0f}"))
    leg = ax2.legend(frameon=True, facecolor=BG_DARK, edgecolor=BORDER_CLR, fontsize=9, loc="upper right")
    for t in leg.get_texts():
        t.set_color(TEXT_MAIN)

    fig.savefig(IMG_DIR / "finbert_sentiment_distribution.png", dpi=220, facecolor=BG_DARK)
    plt.close(fig)


def _draw_confusion_matrix(
    ax: plt.Axes,
    cm: np.ndarray,
    title: str,
    cmap_name: str,
    subtitle: str,
) -> None:
    ax.set_facecolor(PANEL_BG)
    im = ax.imshow(cm, cmap=cmap_name, aspect="auto")
    ax.set_title(title, color=TEXT_MAIN, fontsize=12, fontweight="bold", loc="left", pad=18)
    ax.text(0.0, 1.02, subtitle, transform=ax.transAxes, color=TEXT_MUTED, fontsize=8.8)

    classes = ["Low Volatility (0)", "High Volatility (1)"]
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(classes, color=TEXT_MAIN, fontsize=9, fontweight="bold")
    ax.set_yticklabels(classes, color=TEXT_MAIN, fontsize=9, fontweight="bold")
    ax.set_xlabel("Predicted Volatility Regime", color=TEXT_MUTED, fontsize=9.5, labelpad=6)
    ax.set_ylabel("Actual Volatility Regime", color=TEXT_MUTED, fontsize=9.5, labelpad=6)

    row_sums = cm.sum(axis=1, keepdims=True)
    for i in range(2):
        for j in range(2):
            val = int(cm[i, j])
            pct = val / row_sums[i, 0] * 100.0
            txt_color = "white" if val > 4000 else "#0f172a"
            tag = "TN" if (i == 0 and j == 0) else ("FP" if (i == 0 and j == 1) else ("FN" if (i == 1 and j == 0) else "TP"))
            ax.text(
                j, i,
                f"{tag}: {val:,}\n({pct:.1f}% of class)",
                ha="center", va="center", color=txt_color, fontsize=10.5, fontweight="bold"
            )
    for spine in ax.spines.values():
        spine.set_color(BORDER_CLR)


def plot_confusion_matrices() -> None:
    """
    Generates `dashboard/images/cnn_vs_xgboost_confusion_matrices.png`
    (Thesis Figures 5.3, 5.4, 6.1 & 6.3: CNN+Attention vs XGBoost+SMOTE Confusion Matrices).
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16.0, 6.4), facecolor=BG_DARK)
    fig.subplots_adjust(wspace=0.38, top=0.80, bottom=0.14, left=0.14, right=0.97)

    fig.suptitle(
        "CONFUSION MATRIX COMPARISON ON TEST SET (n = 16,319: 10,967 LOW | 5,352 HIGH)",
        color=TEXT_MAIN, fontsize=14.5, fontweight="bold", x=0.14, ha="left", y=0.95
    )

    cm_cnn = np.array([[10232, 735], [2185, 3167]])
    cm_xgb = np.array([[10247, 720], [1875, 3477]])

    _draw_confusion_matrix(
        ax1,
        cm_cnn,
        "1D-CNN + Multihead Attention (Fig. 6.1)",
        "Blues",
        "Accuracy: 82.11% | ROC-AUC: 0.82 | Low F1: 0.88 | High F1: 0.68",
    )
    _draw_confusion_matrix(
        ax2,
        cm_xgb,
        "XGBoost + SMOTE Ensemble (Fig. 6.3)",
        "Purples",
        "Accuracy: 84.10% | ROC-AUC: 0.86 | Low F1: 0.89 | High F1: 0.73 (+6.0% High Recall)",
    )

    fig.savefig(IMG_DIR / "cnn_vs_xgboost_confusion_matrices.png", dpi=220, facecolor=BG_DARK)
    plt.close(fig)


def plot_roc_and_feature_importance(roc_df: pd.DataFrame, fi_df: pd.DataFrame) -> None:
    """
    Generates `dashboard/images/roc_curves_and_feature_importance.png`
    (Thesis Figures 6.2 & 6.4 ROC Curves + XGBoost Feature Importance).
    """
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(16.5, 6.4), facecolor=BG_DARK, gridspec_kw={"width_ratios": [1.0, 1.35]}
    )
    fig.subplots_adjust(wspace=0.48, top=0.83, bottom=0.13, left=0.06, right=0.97)

    fig.suptitle(
        "ROC-AUC DISCRIMINATION CURVES (FIG. 6.2 & 6.4) & FEATURE IMPORTANCE DRIVERS",
        color=TEXT_MAIN, fontsize=14.5, fontweight="bold", x=0.06, ha="left", y=0.95
    )

    # Left: ROC Curves (Thesis Figures 6.2 & 6.4)
    ax1.plot(
        roc_df["FPR"], roc_df["TPR_XGBoost_SMOTE"],
        color=CLR_XGB, linewidth=2.6, label="XGBoost + SMOTE (AUC = 0.86)"
    )
    ax1.plot(
        roc_df["FPR"], roc_df["TPR_CNN_Attention"],
        color=CLR_CNN, linewidth=2.4, label="1D-CNN + Multihead Attention (AUC = 0.82)"
    )
    ax1.plot(
        roc_df["FPR"], roc_df["TPR_Decision_Tree"],
        color=CLR_DT, linewidth=1.8, linestyle="-.", label="Decision Tree Baseline (AUC = 0.71)"
    )
    ax1.plot([0, 1], [0, 1], color="#64748b", linestyle="--", linewidth=1.2, label="Random Classifier (AUC = 0.50)")
    ax1.fill_between(roc_df["FPR"], roc_df["TPR_XGBoost_SMOTE"], alpha=0.12, color=CLR_XGB)

    style_ax(
        ax1,
        "Receiver Operating Characteristic (ROC) Curves",
        xlabel="False Positive Rate (1 - Specificity)",
        ylabel="True Positive Rate (Sensitivity / Recall)",
    )
    leg = ax1.legend(frameon=True, facecolor=BG_DARK, edgecolor=BORDER_CLR, fontsize=8.8, loc="lower right")
    for t in leg.get_texts():
        t.set_color(TEXT_MAIN)

    # Right: Feature Importance
    top_fi = fi_df.sort_values("Gain_Importance", ascending=True).tail(10)
    cat_colors = {
        "Technical Volatility": "#38bdf8",
        "FinBERT Sentiment": "#a855f7",
        "Lagged Returns": "#10b981",
        "Contextual Metadata": "#f59e0b",
        "Calendar Signals": "#f43f5e",
    }
    bar_colors = [cat_colors.get(c, CLR_CNN) for c in top_fi["Feature_Category"]]
    bars = ax2.barh(top_fi["Feature_Description"], top_fi["Gain_Importance"] * 100, color=bar_colors, height=0.64, alpha=0.92)
    style_ax(
        ax2,
        "Top 10 Predictive Features by Normalized XGBoost Gain (%)",
        xlabel="Relative Gain Contribution (%)",
        ylabel="",
    )
    ax2.xaxis.set_major_formatter(mtick.PercentFormatter())
    ax2.set_xlim(0, 20.5)
    for bar, val in zip(bars, top_fi["Gain_Importance"] * 100):
        ax2.text(
            val + 0.35,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}%",
            va="center", ha="left", color=TEXT_MAIN, fontsize=8.8, fontweight="bold"
        )

    fig.savefig(IMG_DIR / "roc_curves_and_feature_importance.png", dpi=220, facecolor=BG_DARK)
    plt.close(fig)


def plot_executive_thesis_overview(
    firm_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    roc_df: pd.DataFrame,
    fi_df: pd.DataFrame,
) -> None:
    """
    Generates `dashboard/images/executive_thesis_overview.png`
    (4-Panel Executive Overview for the README hero banner).
    """
    fig, axes = plt.subplots(2, 2, figsize=(17.2, 10.6), facecolor=BG_DARK)
    fig.subplots_adjust(wspace=0.42, hspace=0.34, top=0.88, bottom=0.07, left=0.06, right=0.97)

    fig.suptitle(
        "STOCK MARKET VOLATILITY CLASSIFICATION USING FINBERT NEWS SENTIMENT — MSc THESIS MONITOR",
        color=TEXT_MAIN, fontsize=15.5, fontweight="bold", x=0.06, ha="left", y=0.96
    )
    fig.text(
        0.06, 0.925,
        "Dataset: 55,349 Cleaned NASDAQ-100 News & Price Records (10 Tech Giants, 15-Day Lags) | Test Split: 16,319 Samples | Best Model: XGBoost + SMOTE (84.10% Acc, 0.86 AUC)",
        color=TEXT_MUTED, fontsize=10, fontweight="bold"
    )

    ax_tl, ax_tr = axes[0, 0], axes[0, 1]
    ax_bl, ax_br = axes[1, 0], axes[1, 1]

    # Panel 1 (TL): Firm-Level Sentiment Stacked Bars
    firms = firm_df["Stock_symbol"].tolist()
    x = np.arange(len(firms))
    w = 0.26
    ax_tl.bar(x - w, firm_df["Negative_Articles"], width=w, color=CLR_NEG, label="Negative (10,049)")
    ax_tl.bar(x,     firm_df["Neutral_Articles"],  width=w, color=CLR_NEU, label="Neutral (34,000)")
    ax_tl.bar(x + w, firm_df["Positive_Articles"], width=w, color=CLR_POS, label="Positive (11,300)")
    style_ax(ax_tl, "1. FinBERT Sentiment Across 10 NASDAQ-100 Firms (55,349 Articles)", ylabel="Article Count")
    ax_tl.set_xticks(x)
    ax_tl.set_xticklabels(firms, fontweight="bold")
    ax_tl.yaxis.set_major_formatter(mtick.StrMethodFormatter("{x:,.0f}"))
    leg1 = ax_tl.legend(frameon=True, facecolor=BG_DARK, edgecolor=BORDER_CLR, fontsize=8.5, loc="upper right")
    for t in leg1.get_texts():
        t.set_color(TEXT_MAIN)

    # Panel 2 (TR): Model Performance Comparison Grouped Bar Chart
    metrics_labels = ["Accuracy", "ROC-AUC", "Low-Vol F1", "High-Vol Recall", "Weighted F1"]
    dt_vals = [74.8, 71.0, 82.0, 54.0, 74.0]
    cnn_vals = [82.11, 82.0, 88.0, 59.2, 81.0]
    xgb_vals = [84.10, 86.0, 89.0, 65.0, 84.0]

    xm = np.arange(len(metrics_labels))
    wm = 0.25
    ax_tr.bar(xm - wm, dt_vals,  width=wm, color=CLR_DT,  label="Decision Tree (74.8%)")
    ax_tr.bar(xm,      cnn_vals, width=wm, color=CLR_CNN, label="1D-CNN + Attention (82.11%)")
    b_xgb = ax_tr.bar(xm + wm, xgb_vals, width=wm, color=CLR_XGB, label="XGBoost + SMOTE (84.10%)")
    style_ax(ax_tr, "2. Model Architecture Benchmark on Test Set (n = 16,319)", ylabel="Score (%)")
    ax_tr.set_xticks(xm)
    ax_tr.set_xticklabels(metrics_labels, fontsize=8.8, fontweight="bold")
    ax_tr.set_ylim(40, 102)
    ax_tr.yaxis.set_major_formatter(mtick.PercentFormatter())
    for bar, val in zip(b_xgb, xgb_vals):
        ax_tr.text(
            bar.get_x() + bar.get_width() / 2, val + 1.2,
            f"{val:.1f}%", ha="center", va="bottom", color=TEXT_MAIN, fontsize=8, fontweight="bold"
        )
    leg2 = ax_tr.legend(frameon=True, facecolor=BG_DARK, edgecolor=BORDER_CLR, fontsize=8.5, loc="lower right")
    for t in leg2.get_texts():
        t.set_color(TEXT_MAIN)

    # Panel 3 (BL): ROC Curves (CNN vs XGBoost vs DT)
    ax_bl.plot(roc_df["FPR"], roc_df["TPR_XGBoost_SMOTE"], color=CLR_XGB, linewidth=2.5, label="XGBoost + SMOTE (AUC = 0.86)")
    ax_bl.plot(roc_df["FPR"], roc_df["TPR_CNN_Attention"], color=CLR_CNN, linewidth=2.3, label="1D-CNN + Attention (AUC = 0.82)")
    ax_bl.plot(roc_df["FPR"], roc_df["TPR_Decision_Tree"], color=CLR_DT, linewidth=1.7, linestyle="-.", label="Decision Tree (AUC = 0.71)")
    ax_bl.plot([0, 1], [0, 1], color="#64748b", linestyle="--", linewidth=1.1, label="Random Baseline (AUC = 0.50)")
    ax_bl.fill_between(roc_df["FPR"], roc_df["TPR_XGBoost_SMOTE"], alpha=0.10, color=CLR_XGB)
    style_ax(ax_bl, "3. ROC Curves: Binary Volatility Classifier (Fig. 6.2 & 6.4)", xlabel="False Positive Rate", ylabel="True Positive Rate")
    leg3 = ax_bl.legend(frameon=True, facecolor=BG_DARK, edgecolor=BORDER_CLR, fontsize=8.5, loc="lower right")
    for t in leg3.get_texts():
        t.set_color(TEXT_MAIN)

    # Panel 4 (BR): Top Feature Importance with concise labels
    short_labels = {
        "ATR_3D": "3-Day ATR (Price Range)",
        "Rolling_Volatility_3D": "3-Day Rolling Volatility",
        "Sentiment_Polarity": "FinBERT Daily Sentiment",
        "Sentiment_Rolling_5D": "FinBERT 5-Day Rolling Mean",
        "Return_Lag_1": "Lagged Return (t - 1)",
        "Sentiment_Lag_1": "FinBERT Sentiment (t - 1)",
        "FinBERT_Confidence": "FinBERT Softmax Confidence",
        "Return_Lag_2": "Lagged Return (t - 2)",
    }
    top_fi = fi_df.sort_values("Gain_Importance", ascending=True).tail(8).copy()
    top_fi["Short_Label"] = top_fi["Feature_Code"].map(lambda c: short_labels.get(c, c))
    bars = ax_br.barh(top_fi["Short_Label"], top_fi["Gain_Importance"] * 100, color=CLR_CNN, height=0.62, alpha=0.9)
    style_ax(ax_br, "4. Top Predictive Signals (Technical + FinBERT Lags)", xlabel="XGBoost Relative Gain (%)")
    ax_br.xaxis.set_major_formatter(mtick.PercentFormatter())
    ax_br.set_xlim(0, 20.0)
    for bar, val in zip(bars, top_fi["Gain_Importance"] * 100):
        ax_br.text(val + 0.3, bar.get_y() + bar.get_height() / 2, f"{val:.1f}%", va="center", color=TEXT_MAIN, fontsize=8.5, fontweight="bold")

    fig.savefig(IMG_DIR / "executive_thesis_overview.png", dpi=220, facecolor=BG_DARK)
    plt.close(fig)


def build_interactive_html_dashboard(
    firm_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    roc_df: pd.DataFrame,
    fi_df: pd.DataFrame,
) -> None:
    """
    Writes a self-contained interactive HTML companion dashboard (`dashboard/interactive_dashboard.html`)
    with interactive model toggling, firm sentiment exploration, confusion matrices, and ROC curves.
    """
    firms_json = firm_df.to_dict(orient="records")
    eval_json = eval_df.to_dict(orient="records")
    fi_json = fi_df.to_dict(orient="records")
    roc_step = roc_df.iloc[::4].to_dict(orient="list")

    import json

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Stock Market Volatility Classification Using News Sentiment — MSc Thesis Dashboard</title>
  <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
  <style>
    :root {{
      --bg: #0b1120;
      --panel: #131c31;
      --border: #1e293b;
      --text: #f8fafc;
      --muted: #94a3b8;
      --accent: #38bdf8;
      --purple: #a855f7;
      --green: #10b981;
      --amber: #f59e0b;
      --red: #ef4444;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      background: var(--bg);
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Inter, sans-serif;
      padding: 24px 32px;
      line-height: 1.5;
    }}
    header {{
      border-bottom: 1px solid var(--border);
      padding-bottom: 18px;
      margin-bottom: 22px;
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      flex-wrap: wrap;
      gap: 16px;
    }}
    .badge {{
      display: inline-block;
      background: rgba(168, 85, 247, 0.16);
      color: #d8b4fe;
      border: 1px solid rgba(168, 85, 247, 0.4);
      font-size: 12px;
      font-weight: 700;
      padding: 4px 10px;
      border-radius: 999px;
      margin-bottom: 8px;
      letter-spacing: 0.04em;
    }}
    h1 {{ font-size: 24px; font-weight: 800; letter-spacing: -0.02em; }}
    .sub {{ color: var(--muted); font-size: 13.5px; margin-top: 4px; }}
    .kpi-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(195px, 1fr));
      gap: 14px;
      margin-bottom: 22px;
    }}
    .kpi {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 16px 18px;
    }}
    .kpi .label {{ color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: 0.05em; }}
    .kpi .val {{ font-size: 26px; font-weight: 800; margin-top: 4px; color: var(--text); }}
    .kpi .delta {{ font-size: 12px; margin-top: 4px; color: var(--green); font-weight: 600; }}
    .controls {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 14px 18px;
      margin-bottom: 22px;
      display: flex;
      align-items: center;
      gap: 18px;
      flex-wrap: wrap;
    }}
    .controls label {{ font-size: 13px; color: var(--muted); font-weight: 600; }}
    select {{
      background: var(--bg);
      color: var(--text);
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 7px 12px;
      font-size: 13px;
      font-weight: 600;
    }}
    .grid-2 {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(460px, 1fr));
      gap: 18px;
      margin-bottom: 18px;
    }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 16px;
    }}
    .card h3 {{ font-size: 14.5px; margin-bottom: 10px; color: var(--text); }}
    .plot {{ width: 100%; height: 370px; }}
  </style>
</head>
<body>
  <header>
    <div>
      <span class="badge">MAYNOOTH UNIVERSITY &middot; MSc DATA SCIENCE &amp; ANALYTICS (EE648) &middot; FIRST CLASS HONOURS (1:1)</span>
      <h1>Stock Market Volatility Classification Using News Sentiment</h1>
      <p class="sub">Author: Uthej Alagani &middot; Supervisor: Dr. Bryan Hennelly &middot; FinBERT + PyTorch 1D-CNN Multihead Attention + SMOTE-XGBoost Pipeline</p>
    </div>
  </header>

  <div class="kpi-grid">
    <div class="kpi">
      <div class="label">Cleaned NASDAQ-100 News</div>
      <div class="val">55,349</div>
      <div class="delta">10 Leading Tech Constituents</div>
    </div>
    <div class="kpi">
      <div class="label">Holdout Test Set Support</div>
      <div class="val">16,319</div>
      <div class="delta">10,967 Low | 5,352 High Vol</div>
    </div>
    <div class="kpi">
      <div class="label">XGBoost + SMOTE Accuracy</div>
      <div class="val" style="color:#a855f7">84.10%</div>
      <div class="delta">ROC-AUC: 0.86 | Weighted F1: 0.84</div>
    </div>
    <div class="kpi">
      <div class="label">1D-CNN + Attention Acc.</div>
      <div class="val" style="color:#38bdf8">82.11%</div>
      <div class="delta">ROC-AUC: 0.82 | Low Recall: 0.93</div>
    </div>
    <div class="kpi">
      <div class="label">Temporal Lag Window</div>
      <div class="val">15 Days</div>
      <div class="delta">FinBERT + ATR_3D + Rolling Vol</div>
    </div>
  </div>

  <div class="controls">
    <label for="modelSelect">Inspect Confusion Matrix Model:</label>
    <select id="modelSelect">
      <option value="XGBoost + SMOTE Ensemble">XGBoost + SMOTE Ensemble (84.10% Acc | 0.86 AUC)</option>
      <option value="CNN + Multihead Attention (PyTorch)">1D-CNN + Multihead Attention (82.11% Acc | 0.82 AUC)</option>
      <option value="Decision Tree Baseline (CART)">Decision Tree Baseline (74.80% Acc | 0.71 AUC)</option>
    </select>
    <span id="modelSummaryText" style="font-size:13px;color:#cbd5e1;font-weight:600;"></span>
  </div>

  <div class="grid-2">
    <div class="card">
      <h3>1. FinBERT News Sentiment Distribution by NASDAQ-100 Company (Fig. 5.2)</h3>
      <div id="chartSentiment" class="plot"></div>
    </div>
    <div class="card">
      <h3>2. Interactive Test-Set Confusion Matrix (n = 16,319)</h3>
      <div id="chartCM" class="plot"></div>
    </div>
  </div>

  <div class="grid-2">
    <div class="card">
      <h3>3. Receiver Operating Characteristic (ROC) Curves (Fig. 6.2 &amp; 6.4)</h3>
      <div id="chartROC" class="plot"></div>
    </div>
    <div class="card">
      <h3>4. Top Predictive Feature Importance (XGBoost Relative Gain %)</h3>
      <div id="chartFI" class="plot"></div>
    </div>
  </div>

  <script>
    const firmsData = {json.dumps(firms_json)};
    const evalData = {json.dumps(eval_json)};
    const fiData = {json.dumps(fi_json)};
    const rocData = {json.dumps(roc_step)};

    const layoutBase = {{
      paper_bgcolor: '#131c31',
      plot_bgcolor: '#131c31',
      font: {{ color: '#f8fafc', family: 'Inter, sans-serif', size: 12 }},
      margin: {{ l: 55, r: 25, t: 20, b: 50 }},
      xaxis: {{ gridcolor: '#1e293b', zerolinecolor: '#1e293b' }},
      yaxis: {{ gridcolor: '#1e293b', zerolinecolor: '#1e293b' }},
      legend: {{ bgcolor: 'rgba(11,17,32,0.8)', bordercolor: '#1e293b', borderwidth: 1 }}
    }};

    // 1. Sentiment Chart
    Plotly.newPlot('chartSentiment', [
      {{
        x: firmsData.map(d => d.Stock_symbol),
        y: firmsData.map(d => d.Negative_Articles),
        name: 'Negative', type: 'bar', marker: {{ color: '#ef4444' }}
      }},
      {{
        x: firmsData.map(d => d.Stock_symbol),
        y: firmsData.map(d => d.Neutral_Articles),
        name: 'Neutral', type: 'bar', marker: {{ color: '#f59e0b' }}
      }},
      {{
        x: firmsData.map(d => d.Stock_symbol),
        y: firmsData.map(d => d.Positive_Articles),
        name: 'Positive', type: 'bar', marker: {{ color: '#10b981' }}
      }}
    ], {{ ...layoutBase, barmode: 'group', yaxis: {{ ...layoutBase.yaxis, title: 'Articles' }} }}, {{ responsive: true }});

    // 2. Confusion Matrix Updater
    function updateCM(modelName) {{
      const row = evalData.find(d => d.Model === modelName) || evalData[2];
      document.getElementById('modelSummaryText').innerText =
        `Accuracy: ${{row.Accuracy_Pct.toFixed(2)}}% | ROC-AUC: ${{row.ROC_AUC.toFixed(2)}} | Low F1: ${{row.F1_Low.toFixed(2)}} | High F1: ${{row.F1_High.toFixed(2)}} (High Recall: ${{(row.Recall_High*100).toFixed(0)}}%)`;

      const z = [
        [row.TN_Low_Low, row.FP_Low_High],
        [row.FN_High_Low, row.TP_High_High]
      ];
      const text = [
        [`TN: ${{row.TN_Low_Low.toLocaleString()}} (93.3%)`, `FP: ${{row.FP_Low_High.toLocaleString()}} (6.7%)`],
        [`FN: ${{row.FN_High_Low.toLocaleString()}}`, `TP: ${{row.TP_High_High.toLocaleString()}}`]
      ];
      Plotly.react('chartCM', [{{
        z: z,
        x: ['Predicted Low (0)', 'Predicted High (1)'],
        y: ['Actual Low (0)', 'Actual High (1)'],
        text: text,
        texttemplate: '%{{text}}',
        textfont: {{ size: 14, color: '#ffffff' }},
        type: 'heatmap',
        colorscale: 'Purples',
        showscale: true
      }}], {{
        ...layoutBase,
        yaxis: {{ ...layoutBase.yaxis, autorange: 'reversed' }}
      }}, {{ responsive: true }});
    }}
    document.getElementById('modelSelect').addEventListener('change', e => updateCM(e.target.value));
    updateCM('XGBoost + SMOTE Ensemble');

    // 3. ROC Curves
    Plotly.newPlot('chartROC', [
      {{ x: rocData.FPR, y: rocData.TPR_XGBoost_SMOTE, name: 'XGBoost + SMOTE (AUC = 0.86)', mode: 'lines', line: {{ color: '#a855f7', width: 3 }} }},
      {{ x: rocData.FPR, y: rocData.TPR_CNN_Attention, name: '1D-CNN + Attention (AUC = 0.82)', mode: 'lines', line: {{ color: '#38bdf8', width: 2.5 }} }},
      {{ x: rocData.FPR, y: rocData.TPR_Decision_Tree, name: 'Decision Tree (AUC = 0.71)', mode: 'lines', line: {{ color: '#64748b', width: 2, dash: 'dashdot' }} }},
      {{ x: [0, 1], y: [0, 1], name: 'Random (AUC = 0.50)', mode: 'lines', line: {{ color: '#475569', width: 1.5, dash: 'dash' }} }}
    ], {{
      ...layoutBase,
      xaxis: {{ ...layoutBase.xaxis, title: 'False Positive Rate' }},
      yaxis: {{ ...layoutBase.yaxis, title: 'True Positive Rate' }}
    }}, {{ responsive: true }});

    // 4. Feature Importance
    const fiSorted = [...fiData].sort((a, b) => a.Gain_Importance - b.Gain_Importance).slice(-10);
    Plotly.newPlot('chartFI', [{{
      y: fiSorted.map(d => d.Feature_Description),
      x: fiSorted.map(d => +(d.Gain_Importance * 100).toFixed(1)),
      type: 'bar',
      orientation: 'h',
      marker: {{ color: '#38bdf8' }}
    }}], {{
      ...layoutBase,
      margin: {{ l: 230, r: 25, t: 20, b: 45 }},
      xaxis: {{ ...layoutBase.xaxis, title: 'Normalized Gain Importance (%)' }}
    }}, {{ responsive: true }});
  </script>
</body>
</html>
"""
    (DASH_DIR / "interactive_dashboard.html").write_text(html_content, encoding="utf-8")


def main() -> None:
    firm_df = pd.read_csv(PROC_DIR / "nasdaq100_sentiment_distribution_by_firm.csv")
    eval_df = pd.read_csv(PROC_DIR / "model_evaluation_comparison.csv")
    roc_df = pd.read_csv(PROC_DIR / "roc_curve_points.csv")
    fi_df = pd.read_csv(PROC_DIR / "xgboost_feature_importance.csv")

    plot_finbert_sentiment_distribution(firm_df)
    plot_confusion_matrices()
    plot_roc_and_feature_importance(roc_df, fi_df)
    plot_executive_thesis_overview(firm_df, eval_df, roc_df, fi_df)
    build_interactive_html_dashboard(firm_df, eval_df, roc_df, fi_df)
    print("[05_build_dashboard] Generated 4 high-res PNGs in dashboard/images/ and dashboard/interactive_dashboard.html")


if __name__ == "__main__":
    main()
