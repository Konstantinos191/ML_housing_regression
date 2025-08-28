from __future__ import annotations
import argparse
from .data import load_csv, save_csv
import joblib

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--model-path", default="results/pipeline.joblib")
    ap.add_argument("--out", default="predictions.csv")
    ap.add_argument("--numeric", nargs="+", required=True)
    ap.add_argument("--categorical", nargs="*", default=[])
    ap.add_argument("--target", default=None)  # if present in CSV, we’ll ignore it
    args = ap.parse_args()

    df = load_csv(args.csv)
    if args.target and args.target in df.columns:
        X = df[args.numeric + args.categorical].copy()
    else:
        X = df[args.numeric + args.categorical].copy()

    pipe = joblib.load(args.model_path)
    df_out = df.copy()
    df_out["prediction"] = pipe.predict(X)
    save_csv(df_out, args.out)
    print(f"Saved predictions → {args.out}")

if __name__ == "__main__":
    main()
