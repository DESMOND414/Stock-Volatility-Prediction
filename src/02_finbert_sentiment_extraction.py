"""
Chapter 3.2.1, 4.3 & 5.1: FinBERT-Based Sentiment Extraction Pipeline.

Implements the exact Hugging Face Transformers workflow (`ProsusAI/finbert`)
described in Chapter 4.3 (Figure 4.2):
- Loads cleaned NASDAQ-100 news records
- Applies `safe_truncate(text, tokenizer, max_tokens=512)` to prevent BERT token overflow
- Extracts Positive, Neutral, and Negative sentiment classifications & softmax probabilities
- Maps labels to numerical scores (`{positive: 1, neutral: 0, negative: 2}` and directional `{1, 0, -1}`)
- Exports `nasdaq100_with_sentiment_scored.csv`
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC_DIR = ROOT / "data" / "processed"
MODEL_NAME = "ProsusAI/finbert"


def safe_truncate(text: str, tokenizer: Any, max_tokens: int = 512) -> str:
    """
    Chapter 4.3 (Figure 4.2): Safeguards against BERT 512-token input length
    violations and strips special tokens before inference.
    """
    encoded = tokenizer.encode(
        str(text),
        truncation=True,
        max_length=max_tokens,
        return_tensors=None,
    )
    return tokenizer.decode(encoded, skip_special_tokens=True)


def run_finbert_inference(
    input_path: Path,
    output_csv_path: Path,
    model_name: str = MODEL_NAME,
) -> pd.DataFrame:
    """
    Executes the full FinBERT sentiment scoring pipeline on an input Excel/CSV file
    and writes `nasdaq100_with_sentiment_scored.csv`.
    """
    from transformers import AutoTokenizer, pipeline

    if input_path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(input_path)
    else:
        df = pd.read_csv(input_path)

    df = df.dropna(subset=["Stock_symbol"]).copy()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    sentiment_model = pipeline("sentiment-analysis", model=model_name, tokenizer=tokenizer)

    df["Truncated_Title"] = (
        df["Article_title"].astype(str).apply(lambda x: safe_truncate(x, tokenizer, max_tokens=512))
    )

    preds = df["Truncated_Title"].apply(lambda x: sentiment_model(x)[0])
    df["Sentiment_Label"] = preds.apply(lambda r: str(r["label"]).lower())
    df["FinBERT_Confidence"] = preds.apply(lambda r: float(r["score"]))

    # Chapter 4.3 Figure 4.2 label map + Chapter 5.1.3 directional polarity
    label_map = {"positive": 1, "neutral": 0, "negative": 2}
    polarity_map = {"positive": 1, "neutral": 0, "negative": -1}
    df["Sentiment_Score"] = df["Sentiment_Label"].map(label_map)
    df["Sentiment_Polarity"] = df["Sentiment_Label"].map(polarity_map)

    df.to_csv(output_csv_path, index=False)
    print(f"File saved as '{output_csv_path.name}'")
    return df


def summarize_sentiment_characteristics() -> pd.DataFrame:
    """
    Summarizes dataset-level (55,349 articles) and firm-level FinBERT sentiment
    distributions reported in Chapter 5.1.1 & 5.1.2 (Figures 5.1 & 5.2).
    """
    firm_csv = PROC_DIR / "nasdaq100_sentiment_distribution_by_firm.csv"
    firm_df = pd.read_csv(firm_csv)

    total_articles = int(firm_df["Total_Articles"].sum())
    total_neg = int(firm_df["Negative_Articles"].sum())
    total_neu = int(firm_df["Neutral_Articles"].sum())
    total_pos = int(firm_df["Positive_Articles"].sum())

    print("=== FinBERT Dataset-Level Sentiment Summary (n = 55,349) ===")
    print(f"  Neutral  : {total_neu:,} articles ({total_neu / total_articles * 100:.1f}%)")
    print(f"  Positive : {total_pos:,} articles ({total_pos / total_articles * 100:.1f}%)")
    print(f"  Negative : {total_neg:,} articles ({total_neg / total_articles * 100:.1f}%)")
    return firm_df


if __name__ == "__main__":
    summarize_sentiment_characteristics()
