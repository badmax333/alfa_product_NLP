"""Скоринг склонности: клиент × продукт → score, топ-N продуктов."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml

from feature_rules import PRODUCT_IDS

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ONBOARDING = ROOT / "data" / "alfa_onboarding_dataset_5000.csv"
DEFAULT_MODEL = ROOT / "models" / "propensity_lgbm.pkl"
DEFAULT_CONFIG = ROOT / "models" / "feature_config.json"
DEFAULT_PRODUCTS = ROOT / "config" / "products.yaml"


@dataclass
class ProductScore:
    product_id: str
    product_name: str
    propensity_score: float
    rank: int


def load_products_config(path: Path = DEFAULT_PRODUCTS) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)["products"]


class PropensityScorer:
    def __init__(
        self,
        model_path: Path = DEFAULT_MODEL,
        config_path: Path = DEFAULT_CONFIG,
        products_path: Path = DEFAULT_PRODUCTS,
    ) -> None:
        self.pipe = joblib.load(model_path)
        with open(config_path, encoding="utf-8") as f:
            self.config = json.load(f)
        self.cat_features = self.config["cat_features"]
        self.num_features = self.config["num_features"]
        self.feature_cols = self.cat_features + self.num_features
        self.products = load_products_config(products_path)

    def _client_to_matrix(self, client: pd.Series, product_id: str) -> pd.DataFrame:
        row = client.copy()
        if "target" in row.index:
            row["priority_segment"] = row["target"]
        row["product_id"] = product_id
        data = {c: row[c] if c in row.index else np.nan for c in self.feature_cols}
        X = pd.DataFrame([data])
        for c in self.cat_features:
            X[c] = X[c].astype(str)
        return X

    def score_product(self, client: pd.Series, product_id: str) -> float:
        X = self._client_to_matrix(client, product_id)
        return float(self.pipe.predict_proba(X)[0, 1])

    def score_client(self, client: pd.Series, top_k: int = 3) -> list[ProductScore]:
        scores: list[tuple[str, float]] = []
        for pid in PRODUCT_IDS:
            scores.append((pid, self.score_product(client, pid)))
        scores.sort(key=lambda x: x[1], reverse=True)

        out: list[ProductScore] = []
        for rank, (pid, sc) in enumerate(scores[:top_k], start=1):
            out.append(
                ProductScore(
                    product_id=pid,
                    product_name=self.products[pid]["name_ru"],
                    propensity_score=round(sc, 6),
                    rank=rank,
                )
            )
        return out

    def score_all_products(self, client: pd.Series) -> list[ProductScore]:
        return self.score_client(client, top_k=len(PRODUCT_IDS))

    def score_dataframe(
        self,
        clients: pd.DataFrame,
        *,
        top_k: int = 3,
        client_id_col: str | None = None,
    ) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for idx, client in clients.iterrows():
            cid = int(client[client_id_col]) if client_id_col and client_id_col in client else int(idx)
            for ps in self.score_client(client, top_k=top_k):
                rows.append(
                    {
                        "client_id": cid,
                        "priority_segment": client.get("target", client.get("priority_segment")),
                        **asdict(ps),
                    }
                )
        return pd.DataFrame(rows)


def load_onboarding_clients(path: Path | None = None) -> pd.DataFrame:
    path = path or DEFAULT_ONBOARDING
    return pd.read_csv(path).reset_index(drop=True)
