
import pandas as pd
import numpy as np

def basic_clean(df):
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    if 'timestamp' in df.columns and 'date' not in df.columns:
        df['date'] = pd.to_datetime(df['timestamp'], errors='coerce')
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    if 'category' in df.columns:
        df['category'] = df['category'].fillna('Unknown').astype(str)
    else:
        df['category'] = 'Unknown'
    if 'region' in df.columns:
        df['region'] = df['region'].fillna('Unknown').astype(str)
    else:
        df['region'] = 'Unknown'
    if 'description' in df.columns:
        df['description'] = df['description'].fillna('').astype(str)
    else:
        df['description'] = ''
    df['description_anonymized'] = df['description'].replace(to_replace=r'\b\d{10}\b', value='[PHONE]', regex=True)
    df['description_anonymized'] = df['description_anonymized'].replace(to_replace=r'\b\d+\b', value='[NUM]', regex=True)
    df['description_anonymized'] = df['description_anonymized'].str.lower()
    return df

def aggregate_counts(df, freq='W', date_col='date'):
    if date_col not in df.columns or df[date_col].isna().all():
        raise ValueError("Date column missing or all NA — please provide a 'date' column in data.")
    df = df.copy()
    df['period_start'] = df[date_col].dt.to_period(freq).dt.to_timestamp()
    agg = df.groupby(['period_start','region','category']).size().reset_index(name='count')
    return agg
