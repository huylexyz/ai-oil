"""Reproduce TV2's Brent data cleaning from the repository's raw CSV."""

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "DCOILBRENTEU_2015-2026.csv"
OUT_DIR = ROOT / "data" / "processed"
REPORT = ROOT / "docs" / "cleaning_report_tv2.txt"
START, END = "2015-01-01", "2026-09-30"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=RAW)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--report", type=Path, default=REPORT)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    lines = []

    def log(message: str) -> None:
        print(message)
        lines.append(message)

    df = pd.read_csv(args.input)
    log(f"1. Doc du lieu tho: {len(df)} dong, cot = {list(df.columns)}")

    # Preserve TV2's parsing and filtering rules.
    df["observation_date"] = pd.to_datetime(df["observation_date"], errors="raise")
    df["DCOILBRENTEU"] = pd.to_numeric(df["DCOILBRENTEU"], errors="coerce")
    df = df[["observation_date", "DCOILBRENTEU"]]

    n0 = len(df)
    df = df[(df["observation_date"] >= START) & (df["observation_date"] <= END)]
    log(f"2. Gioi han {START} -> {END}: loai {n0 - len(df)} dong ngoai khoang")

    duplicate = df["observation_date"].duplicated(keep="last")
    log(f"3. Ngay trung lap: {int(duplicate.sum())} (giu ban ghi cuoi neu co)")
    df = df[~duplicate]
    df = df.sort_values("observation_date").reset_index(drop=True)

    missing = df[df["DCOILBRENTEU"].isna()].copy()
    missing["reason"] = "Gia thieu (FRED/EIA: ngay nghi le, khong co phien)"
    missing.to_csv(args.output_dir / "removed_rows.csv", index=False)
    log(f"4. Gia tri thieu: {len(missing)} dong -> loai bo, ghi tai removed_rows.csv")
    df = df.dropna(subset=["DCOILBRENTEU"]).reset_index(drop=True)

    nonpositive = int((df["DCOILBRENTEU"] <= 0).sum())
    log(f"5. Gia <= 0: {nonpositive} dong")
    lowest = df.loc[df["DCOILBRENTEU"].idxmin()]
    highest = df.loc[df["DCOILBRENTEU"].idxmax()]
    log(
        f"   Min {lowest.DCOILBRENTEU:.2f} ({lowest.observation_date.date()}), "
        f"Max {highest.DCOILBRENTEU:.2f} ({highest.observation_date.date()}) "
        "-> la bien dong thi truong that, giu nguyen"
    )

    df["DCOILBRENTEU"] = df["DCOILBRENTEU"].round(2)
    df.to_csv(args.output_dir / "brent_clean.csv", index=False, date_format="%Y-%m-%d")
    log(
        f"6. Xuat brent_clean.csv: {len(df)} dong, "
        f"{df.observation_date.min().date()} -> {df.observation_date.max().date()}"
    )
    args.report.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
