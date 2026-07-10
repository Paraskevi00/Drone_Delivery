import re
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class DroneFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, drop_raw: bool = True):
        self.drop_raw = drop_raw
        self.raw_cols_ = ["wind_token", "route_turns", "alt_profile"]

    def fit(self, X, y=None):
        return self

    @staticmethod
    def _parse_wind_token(token):
        if pd.isna(token):
            return pd.Series([np.nan, np.nan, np.nan])
        match = re.match(r"(?P<direction>[A-Z]+)-(?P<avg>\d+)/(?P<gust>\d+)", str(token))
        if match is None:
            return pd.Series([np.nan, np.nan, np.nan])
        return pd.Series([match.group("direction"), int(match.group("avg")), int(match.group("gust"))])

    @staticmethod
    def _count_direction_changes(route):
        route = str(route)
        if len(route) <= 1:
            return 0
        return sum(route[i] != route[i - 1] for i in range(1, len(route)))

    @staticmethod
    def _parse_altitude(profile):
        try:
            return list(map(int, str(profile).split("-")))
        except Exception:
            return []

    @staticmethod
    def _count_altitude_changes(levels):
        if len(levels) <= 1:
            return 0
        return sum(levels[i] != levels[i - 1] for i in range(1, len(levels)))

    def transform(self, X):
        X = X.copy()

        if "wind_token" in X.columns:
            wind = X["wind_token"].apply(self._parse_wind_token)
            wind.columns = ["wind_direction", "wind_avg", "wind_gust"]
            X = pd.concat([X, wind], axis=1)

        if "route_turns" in X.columns:
            X["route_length"] = X["route_turns"].astype(str).str.len()
            X["direction_changes"] = X["route_turns"].apply(self._count_direction_changes)

        if "alt_profile" in X.columns:
            altitudes = X["alt_profile"].apply(self._parse_altitude)
            X["mean_altitude"] = altitudes.apply(lambda x: np.mean(x) if len(x) else np.nan)
            X["altitude_changes"] = altitudes.apply(self._count_altitude_changes)

        if self.drop_raw:
            X = X.drop(columns=[c for c in self.raw_cols_ if c in X.columns])

        return X
