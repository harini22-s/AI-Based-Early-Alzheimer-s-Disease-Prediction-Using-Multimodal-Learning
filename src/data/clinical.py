"""
Clinical / demographic tabular preprocessing:
  missing-value handling -> encoding -> normalization -> feature selection

Fits on the training split only and is then applied (transform-only) to
val/test, to avoid leakage — the fitted parameters are saved to disk so
inference-time preprocessing matches training exactly.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path

import json
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif


@dataclass
class ClinicalPreprocessor:
    numeric_features: list
    categorical_features: list
    impute_values_: dict = field(default_factory=dict)
    category_maps_: dict = field(default_factory=dict)
    norm_mean_: dict = field(default_factory=dict)
    norm_std_: dict = field(default_factory=dict)
    selected_features_: list = field(default_factory=list)

    @property
    def all_features(self):
        return self.numeric_features + self.categorical_features

    def fit(self, df: pd.DataFrame, labels: np.ndarray, top_k: int | None = None):
        df = df.copy()

        # 1) missing value handling: median for numeric, mode for categorical
        for col in self.numeric_features:
            self.impute_values_[col] = float(df[col].median())
        for col in self.categorical_features:
            self.impute_values_[col] = df[col].mode(dropna=True).iloc[0] if df[col].notna().any() else "Unknown"

        df = self._impute(df)

        # 2) encoding: simple ordinal/label maps for categoricals (kept small & interpretable)
        for col in self.categorical_features:
            cats = sorted(df[col].astype(str).unique().tolist())
            self.category_maps_[col] = {c: i for i, c in enumerate(cats)}

        encoded = self._encode(df)

        # 3) normalization (z-score) fit on numeric + encoded categorical columns
        for col in self.all_features:
            self.norm_mean_[col] = float(encoded[col].mean())
            self.norm_std_[col] = float(encoded[col].std() + 1e-8)

        normed = self._normalize(encoded)

        # 4) feature selection via mutual information with the diagnosis label
        if top_k is not None and top_k < len(self.all_features):
            mi = mutual_info_classif(normed[self.all_features].values, labels, random_state=0)
            order = np.argsort(mi)[::-1]
            self.selected_features_ = [self.all_features[i] for i in order[:top_k]]
        else:
            self.selected_features_ = list(self.all_features)

        return self

    def _impute(self, df):
        df = df.copy()
        for col, val in self.impute_values_.items():
            if col in df.columns:
                df[col] = df[col].fillna(val)
        return df

    def _encode(self, df):
        df = df.copy()
        for col, mapping in self.category_maps_.items():
            df[col] = df[col].astype(str).map(mapping).fillna(-1).astype(float)
        return df

    def _normalize(self, df):
        df = df.copy()
        for col in self.all_features:
            df[col] = (df[col] - self.norm_mean_[col]) / self.norm_std_[col]
        return df

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        df = self._impute(df)
        df = self._encode(df)
        df = self._normalize(df)
        return df[self.selected_features_].values.astype(np.float32)

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(dict(
                numeric_features=self.numeric_features,
                categorical_features=self.categorical_features,
                impute_values_=self.impute_values_,
                category_maps_=self.category_maps_,
                norm_mean_=self.norm_mean_,
                norm_std_=self.norm_std_,
                selected_features_=self.selected_features_,
            ), f, indent=2)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            d = json.load(f)
        obj = cls(d["numeric_features"], d["categorical_features"])
        obj.impute_values_ = d["impute_values_"]
        obj.category_maps_ = d["category_maps_"]
        obj.norm_mean_ = d["norm_mean_"]
        obj.norm_std_ = d["norm_std_"]
        obj.selected_features_ = d["selected_features_"]
        return obj
