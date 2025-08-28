from typing import Sequence
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

def make_preprocessor(numeric: Sequence[str], categorical: Sequence[str]) -> ColumnTransformer:
    return ColumnTransformer([
        ("num", StandardScaler(), list(numeric)),
        ("cat", OneHotEncoder(handle_unknown="ignore"), list(categorical)),
    ])

def make_pipeline(numeric: Sequence[str], categorical: Sequence[str]) -> Pipeline:
    return Pipeline([("pre", make_preprocessor(numeric, categorical)),
                     ("model", LinearRegression())])

def regression_metrics(y_true, y_pred) -> dict:
    mse = float(mean_squared_error(y_true, y_pred))
    return {"mse": mse, "rmse": float(np.sqrt(mse)), "r2": float(r2_score(y_true, y_pred))}
