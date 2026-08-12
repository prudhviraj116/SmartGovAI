
import pandas as pd
import numpy as np

def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans dataset by standardizing column names, filling missing values,
    and parsing date formats.
    """
    df = df.copy()
    
    # 1. Normalize column headers (strip spaces & convert to lowercase)
    df.columns = [str(col).strip().lower() for col in df.columns]

    # 2. Flexible Column Name Mapping
    COLUMN_MAP = {
        # Date variations
        'created_at': 'date', 'created_date': 'date', 'timestamp': 'date', 
        'incident_date': 'date', 'complaint_date': 'date', 'time': 'date',
        
        # Category variations
        'department': 'category', 'dept': 'category', 'issue_type': 'category', 
        'complaint_type': 'category', 'type': 'category', 'service': 'category',
        
        # Region variations
        'district': 'region', 'city': 'region', 'location': 'region', 
        'zone': 'region', 'area': 'region', 'ward': 'region',
        
        # Description variations
        'details': 'description', 'summary': 'description', 'comment': 'description', 
        'text': 'description', 'issue': 'description', 'remark': 'description',
        
        # Severity variations
        'priority': 'severity', 'urgency': 'severity', 'status_priority': 'severity'
    }

    # Rename matched columns
    df = df.rename(columns=COLUMN_MAP)

    # 3. Ensure required columns exist (Assign 'Unknown' if completely missing)
    required_defaults = {
        'category': 'Unknown',
        'region': 'Unknown',
        'description': 'No description provided',
        'severity': 'Medium'
    }

    for col, default_val in required_defaults.items():
        if col not in df.columns:
            df[col] = default_val
        else:
            # Fill NaN / empty values inside existing columns
            df[col] = df[col].fillna(default_val).replace('', default_val)

    # 4. Handle Date Parsing safely
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        # Drop rows where date couldn't be parsed
        df = df.dropna(subset=['date'])
    
    # 5. Anonymize/Clean description column if needed
    df['description_anonymized'] = df['description'].astype(str)

    return df

def aggregate_counts(df, freq='W', date_col='date'):
    if date_col not in df.columns or df[date_col].isna().all():
        raise ValueError("Date column missing or all NA — please provide a 'date' column in data.")
    df = df.copy()
    df['period_start'] = df[date_col].dt.to_period(freq).dt.to_timestamp()
    agg = df.groupby(['period_start','region','category']).size().reset_index(name='count')
    return agg
