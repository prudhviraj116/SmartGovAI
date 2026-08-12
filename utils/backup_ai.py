import pandas as pd


def _top_value(series: pd.Series):
    cleaned = series.dropna()
    if cleaned.empty:
        return None
    return cleaned.mode().iloc[0]


def generate_backup_summary(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "No data available to summarize."

    segments = []
    if "category" in df.columns:
        top_category = _top_value(df["category"])
        if top_category:
            segments.append(f"The dataset is dominated by issues in the '{top_category}' category.")

    if "region" in df.columns:
        top_region = _top_value(df["region"])
        if top_region:
            segments.append(f"Most records originate from the region '{top_region}'.")

    if "severity" in df.columns:
        severity = df["severity"].astype(str)
        high_count = severity.str.contains("high", case=False, na=False).sum()
        if high_count > 0:
            segments.append(f"{high_count} records are marked as high severity.")

    if "date" in df.columns:
        dates = pd.to_datetime(df["date"], errors="coerce").dropna()
        if not dates.empty:
            segments.append(
                f"The dataset contains {len(dates)} dated records from {dates.min().date()} to {dates.max().date()}."
            )

    if "description_anonymized" in df.columns:
        samples = df["description_anonymized"].dropna().astype(str).head(3).tolist()
        if samples:
            segments.append("Example issues include: " + " / ".join(samples))

    if not segments:
        return (
            "A backup summary is available, but the dataset lacks standard summarization columns. "
            "Please include category, region, or date fields for better insights."
        )

    return " ".join(segments)
