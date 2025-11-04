
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

class SimpleTrendPredictor:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=50, random_state=42)
        self.le_region = LabelEncoder()
        self.le_category = LabelEncoder()
        self.fitted = False
        self.mean_per_group = {}

    def prepare_features(self, agg_df):
        df = agg_df.copy().sort_values(['region','category','period_start'])
        df['lag_1'] = df.groupby(['region','category'])['count'].shift(1).fillna(0)
        df['lag_2'] = df.groupby(['region','category'])['count'].shift(2).fillna(0)
        df['region_enc'] = self.le_region.fit_transform(df['region'])
        df['category_enc'] = self.le_category.fit_transform(df['category'])
        X = df[['region_enc','category_enc','lag_1','lag_2']].astype(float)
        y = df['count'].astype(float)
        return X, y, df

    def fit(self, agg_df):
        X, y, df = self.prepare_features(agg_df)
        if len(X) < 10:
            self.model = None
            self.mean_per_group = agg_df.groupby(['region','category'])['count'].mean().to_dict()
            self.fitted = True
            return
        self.model.fit(X, y)
        self.fitted = True

    def predict_next_period(self, agg_df):
        if not self.fitted:
            raise RuntimeError("Model not fitted.")
        last = agg_df.sort_values('period_start').groupby(['region','category']).tail(1).copy()
        last['lag_1'] = last['count']
        last['lag_2'] = 0
        try:
            last['region_enc'] = self.le_region.transform(last['region'])
        except Exception:
            last['region_enc'] = 0
        try:
            last['category_enc'] = self.le_category.transform(last['category'])
        except Exception:
            last['category_enc'] = 0
        X_pred = last[['region_enc','category_enc','lag_1','lag_2']].astype(float)
        if self.model is None:
            preds = []
            for r,c in zip(last['region'], last['category']):
                preds.append(self.mean_per_group.get((r,c), last['count'].mean()))
            last['predicted_count'] = np.round(preds,0).astype(int)
        else:
            raw = self.model.predict(X_pred)
            last['predicted_count'] = np.round(np.clip(raw,0,None),0).astype(int)
        return last[['region','category','predicted_count']]
