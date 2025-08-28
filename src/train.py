from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import joblib
from .data import load_csv
from .preprocess import ColumnSpec, split_xy
from .model import make_pipeline, regression_metrics

def parity_plot(y_true, y_pred, path: Path, title="Actual vs Predicted"):
    plt.figure()
    plt.plot(y_true, y_pred, "x", markersize=5)
    lo, hi = float(min(np.min(y_true), np.min(y_pred))), float(max(np.max(y_true), np.max(y_pred)))
    line = np.linspace(lo, hi, 100)
    plt.plot(line, line, linewidth=1)
    plt.xlabel("Actual"); plt.ylabel("Predicted"); plt.title(title)
    plt.tight_layout(); path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=180); plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--numeric", nargs="+", required=True)
    ap.add_argument("--categorical", nargs="*", default=[])
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--model-path", default="results/pipeline.joblib")
    args = ap.parse_args()

    df = load_csv(args.csv)
    spec = ColumnSpec(target=args.target, numeric=args.numeric, categorical=args.categorical)
    X, y = split_xy(df, spec)

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=args.test_size, random_state=42)
    pipe = make_pipeline(spec.numeric, spec.categorical)
    pipe.fit(Xtr, ytr)

    ytr_hat, yte_hat = pipe.predict(Xtr), pipe.predict(Xte)
    metrics = {"train": regression_metrics(ytr, ytr_hat),
               "test":  regression_metrics(yte, yte_hat),
               "features": {"numeric": list(spec.numeric), "categorical": list(spec.categorical)}}

    resdir = Path(args.results_dir); resdir.mkdir(parents=True, exist_ok=True)
    (resdir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    parity_plot(ytr, ytr_hat, resdir / "parity_train.png")
    parity_plot(yte, yte_hat, resdir / "parity_test.png")

    joblib.dump(pipe, args.model_path)
    print(json.dumps(metrics, indent=2))
    print(f"Saved trained pipeline → {args.model_path}")

if __name__ == "__main__":
    main()
