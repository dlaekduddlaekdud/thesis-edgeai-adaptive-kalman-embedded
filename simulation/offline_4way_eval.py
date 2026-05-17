#!/usr/bin/env python3
"""
Offline 4-Way KF Evaluation
===========================

Run Raw, Fixed KF, CM-AKF, and optional TinyML-AKF on a real experiment CSV
or an 18-column experiment CSV.

Examples:
    python simulation/offline_4way_eval.py path/to/input.csv
    python simulation/offline_4way_eval.py path/to/input.csv --output-dir results
    python simulation/offline_4way_eval.py path/to/input.csv --enable-tinyml
    python simulation/offline_4way_eval.py path/to/input.csv --enable-tinyml --tinyml-model tools/tinyml/tinyml_akf_3feat_int8.tflite

Required input columns:
    tof_distance_mm
    encoder_speed_mms

Optional input columns used when available:
    timestamp_ms, encoder_distance_mm, gt_distance_mm, tof_signal_rate,
    tof_range_status, us_distance_mm, sensor_disagree

Notes:
    - If gt_distance_mm is present, RMSE and MAE are computed.
    - If gt_distance_mm is absent, metrics are skipped and only time-series
      CSV/plots are saved.
    - TinyML-AKF is optional. It is skipped unless --enable-tinyml is set,
      a model file exists, and a TFLite interpreter is importable.
    - Existing repository files are not modified by this script.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Keep Matplotlib/font caches inside writable temp locations when the script is
# executed in a sandboxed environment.
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


KF_Q = 1.0
KF_R_FIXED = 400.0
KF_R_INIT = 400.0
KF_WINDOW_SIZE = 20
KF_R_MIN = 1.0
KF_R_MAX = 10000.0
DEFAULT_DT_S = 0.005

REQUIRED_COLUMNS = ["tof_distance_mm", "encoder_speed_mms"]
OPTIONAL_COLUMNS = [
    "timestamp_ms",
    "encoder_distance_mm",
    "gt_distance_mm",
    "tof_signal_rate",
    "tof_range_status",
    "us_distance_mm",
    "sensor_disagree",
]


@dataclass
class FilterResult:
    name: str
    estimate: np.ndarray
    residual: np.ndarray
    kalman_gain: np.ndarray
    innovation_cov: np.ndarray
    r_values: np.ndarray
    p_values: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline Raw / Fixed KF / CM-AKF / optional TinyML-AKF evaluator"
    )
    parser.add_argument("csv_path", help="Input real experiment or 18-column CSV")
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Base output directory. A per-input subdirectory is created inside it.",
    )
    parser.add_argument(
        "--enable-tinyml",
        action="store_true",
        help="Attempt TinyML-AKF using a TFLite model. Skips with warning if unavailable.",
    )
    parser.add_argument(
        "--tinyml-model",
        default="tools/tinyml/tinyml_akf_3feat_int8.tflite",
        help="Path to INT8 TFLite model for optional TinyML-AKF.",
    )
    parser.add_argument(
        "--default-dt",
        type=float,
        default=DEFAULT_DT_S,
        help="Fallback dt in seconds when timestamp_ms is absent or invalid.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display plots interactively in addition to saving them.",
    )
    return parser.parse_args()


def load_input_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input CSV not found: {path}")

    df = pd.read_csv(path)
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing)
            + ". Required minimum is tof_distance_mm and encoder_speed_mms."
        )

    for col in REQUIRED_COLUMNS + OPTIONAL_COLUMNS:
        if col in df.columns and col != "scenario_id":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    valid = ~(df["tof_distance_mm"].isna() | df["encoder_speed_mms"].isna())
    dropped = int((~valid).sum())
    if dropped:
        print(f"[warn] Dropping {dropped} rows with NaN required fields.")
    df = df.loc[valid].reset_index(drop=True)

    if df.empty:
        raise ValueError("No valid rows remain after required-column filtering.")
    return df


def make_output_dir(base_dir: Path, input_path: Path) -> Path:
    out_dir = base_dir / input_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def build_time_and_dt(df: pd.DataFrame, default_dt: float) -> tuple[np.ndarray, np.ndarray]:
    n = len(df)
    if "timestamp_ms" in df.columns and df["timestamp_ms"].notna().sum() >= 2:
        ts_ms = df["timestamp_ms"].to_numpy(dtype=float)
        time_s = ts_ms / 1000.0
        dt = np.diff(time_s, prepend=np.nan)
        finite_positive = dt[np.isfinite(dt) & (dt > 0)]
        fallback = float(np.median(finite_positive)) if len(finite_positive) else default_dt
        dt[0] = fallback
        bad = ~np.isfinite(dt) | (dt <= 0)
        dt[bad] = fallback
        return time_s, dt

    time_s = np.arange(n, dtype=float) * default_dt
    dt = np.full(n, default_dt, dtype=float)
    return time_s, dt


def compute_encoder_distance(speed: np.ndarray, dt_s: np.ndarray, df: pd.DataFrame) -> np.ndarray:
    if "encoder_distance_mm" in df.columns and df["encoder_distance_mm"].notna().any():
        enc = df["encoder_distance_mm"].to_numpy(dtype=float)
        if np.isnan(enc).any():
            computed = np.cumsum(speed * dt_s)
            enc = np.where(np.isnan(enc), computed, enc)
        return enc
    return np.cumsum(speed * dt_s)


def run_kf(
    z: np.ndarray,
    u: np.ndarray,
    dt_s: np.ndarray,
    mode: str,
    tinyml_r: Optional[np.ndarray] = None,
) -> FilterResult:
    n = len(z)
    x = np.zeros(n, dtype=float)
    p = np.zeros(n, dtype=float)
    residual = np.zeros(n, dtype=float)
    gain = np.zeros(n, dtype=float)
    innov_cov = np.zeros(n, dtype=float)
    r_values = np.zeros(n, dtype=float)

    x[0] = z[0]
    p[0] = KF_R_INIT
    r_current = KF_R_INIT
    r_values[0] = r_current

    res_buf = np.zeros(KF_WINDOW_SIZE, dtype=float)
    buf_idx = 0
    buf_count = 0
    sq_sum = 0.0

    for k in range(1, n):
        x_pred = x[k - 1] + dt_s[k] * u[k]
        p_pred = p[k - 1] + KF_Q
        if p_pred < 1e-6:
            p_pred = 1e-6

        residual[k] = z[k] - x_pred

        if mode == "fixed":
            r_current = KF_R_FIXED
        elif mode == "cm_akf":
            if buf_count >= KF_WINDOW_SIZE:
                old = res_buf[buf_idx]
                sq_sum -= old * old

            res_buf[buf_idx] = residual[k]
            sq_sum += residual[k] * residual[k]
            buf_idx = (buf_idx + 1) % KF_WINDOW_SIZE
            buf_count = min(buf_count + 1, KF_WINDOW_SIZE)

            if buf_count >= KF_WINDOW_SIZE:
                res_sq_mean = sq_sum / float(KF_WINDOW_SIZE)
                r_current = float(np.clip(res_sq_mean - p_pred, KF_R_MIN, KF_R_MAX))
        elif mode == "tinyml_akf":
            if tinyml_r is None:
                raise ValueError("tinyml_r is required for tinyml_akf mode")
            r_current = float(np.clip(tinyml_r[k], KF_R_MIN, KF_R_MAX))
        else:
            raise ValueError(f"Unknown KF mode: {mode}")

        r_values[k] = r_current
        denom = max(p_pred + r_current, 1e-6)
        innov_cov[k] = denom
        gain[k] = p_pred / denom
        x[k] = x_pred + gain[k] * residual[k]
        p[k] = max((1.0 - gain[k]) * p_pred, 1e-6)

    if mode == "fixed":
        label = "Fixed KF"
    elif mode == "cm_akf":
        label = "CM-AKF"
    else:
        label = "TinyML-AKF"

    return FilterResult(
        name=label,
        estimate=x,
        residual=residual,
        kalman_gain=gain,
        innovation_cov=innov_cov,
        r_values=r_values,
        p_values=p,
    )


def residual_window_stats(residual: np.ndarray, window: int = KF_WINDOW_SIZE) -> tuple[np.ndarray, np.ndarray]:
    n = len(residual)
    var = np.full(n, np.nan, dtype=float)
    mean = np.full(n, np.nan, dtype=float)
    for k in range(window, n):
        win = residual[k - window + 1 : k + 1]
        var[k] = float(np.var(win))
        mean[k] = float(np.mean(win))
    return var, mean


def try_import_tflite_interpreter():
    try:
        from tflite_runtime.interpreter import Interpreter

        return Interpreter, "tflite_runtime"
    except Exception:
        pass

    try:
        import tensorflow as tf

        return tf.lite.Interpreter, "tensorflow"
    except Exception:
        return None, None


def predict_tinyml_r_stub(
    df: pd.DataFrame,
    cm_result: FilterResult,
    model_path: Path,
    enable_tinyml: bool,
) -> Optional[np.ndarray]:
    """
    Optional TinyML R prediction hook.

    The checked-in TinyML training script prints normalization constants but
    does not store a scaler metadata file. Without those constants, real-data
    inference would not be reproducible. This function therefore only verifies
    that a TFLite interpreter and model are available, then skips with a clear
    warning until model metadata is added.
    """
    if not enable_tinyml:
        return None

    if not model_path.exists():
        print(f"[warn] TinyML-AKF skipped: model file not found: {model_path}")
        return None

    Interpreter, provider = try_import_tflite_interpreter()
    if Interpreter is None:
        print("[warn] TinyML-AKF skipped: no TFLite interpreter importable.")
        print("[warn] Install tflite_runtime or tensorflow, then provide scaler metadata.")
        return None

    # Instantiate once to catch corrupt model files, but do not run inference
    # because normalization metadata is not checked into the repository.
    try:
        interpreter = Interpreter(model_path=str(model_path))
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()[0]
        print(
            "[warn] TinyML-AKF skipped: model is loadable via "
            f"{provider}, input shape={input_details.get('shape')}, "
            "but scaler metadata is not available."
        )
    except Exception as exc:
        print(f"[warn] TinyML-AKF skipped: failed to load TFLite model: {exc}")
    return None


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.nanmean((a - b) ** 2)))


def mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.nanmean(np.abs(a - b)))


def compute_metrics(
    z: np.ndarray,
    gt: Optional[np.ndarray],
    results: list[FilterResult],
) -> Optional[pd.DataFrame]:
    if gt is None:
        return None

    rows = [
        {
            "method": "Raw",
            "rmse_mm": rmse(z, gt),
            "mae_mm": mae(z, gt),
            "max_abs_error_mm": float(np.nanmax(np.abs(z - gt))),
        }
    ]
    for result in results:
        err = result.estimate - gt
        rows.append(
            {
                "method": result.name,
                "rmse_mm": rmse(result.estimate, gt),
                "mae_mm": mae(result.estimate, gt),
                "max_abs_error_mm": float(np.nanmax(np.abs(err))),
            }
        )
    return pd.DataFrame(rows)


def build_output_dataframe(
    df: pd.DataFrame,
    time_s: np.ndarray,
    dt_s: np.ndarray,
    encoder_distance: np.ndarray,
    results: list[FilterResult],
) -> pd.DataFrame:
    out = pd.DataFrame()
    if "timestamp_ms" in df.columns:
        out["timestamp_ms"] = df["timestamp_ms"].to_numpy()
    else:
        out["timestamp_ms"] = np.round(time_s * 1000.0).astype(int)

    out["time_s"] = time_s
    out["dt_s"] = dt_s
    out["tof_distance_mm"] = df["tof_distance_mm"].to_numpy(dtype=float)
    out["encoder_speed_mms"] = df["encoder_speed_mms"].to_numpy(dtype=float)
    out["encoder_distance_mm"] = encoder_distance

    for col in OPTIONAL_COLUMNS:
        if col in df.columns and col not in out.columns:
            out[col] = df[col].to_numpy()

    for result in results:
        prefix = method_prefix(result.name)
        res_var, res_mean = residual_window_stats(result.residual)
        out[f"{prefix}_estimate_mm"] = result.estimate
        out[f"{prefix}_residual_mm"] = result.residual
        out[f"{prefix}_residual_var"] = res_var
        out[f"{prefix}_residual_mean"] = res_mean
        out[f"{prefix}_kalman_gain"] = result.kalman_gain
        out[f"{prefix}_innovation_cov"] = result.innovation_cov
        out[f"{prefix}_R"] = result.r_values
    return out


def method_prefix(name: str) -> str:
    return (
        name.lower()
        .replace("-", "_")
        .replace(" ", "_")
        .replace("__", "_")
    )


def save_plot(
    out_path: Path,
    time_s: np.ndarray,
    z: np.ndarray,
    gt: Optional[np.ndarray],
    results: list[FilterResult],
    show: bool,
) -> None:
    has_gt = gt is not None
    fig, axes = plt.subplots(
        3 if has_gt else 2,
        1,
        figsize=(11, 8 if has_gt else 6),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1.2, 1.2] if has_gt else [3, 1.2]},
    )
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    ax = axes[0]
    ax.plot(time_s, z, color="#d4735e", lw=0.8, alpha=0.7, label="Raw ToF")
    if gt is not None:
        ax.plot(time_s, gt, color="#1a1a1a", lw=1.3, label="Ground truth")
    colors = {
        "Fixed KF": "#2b5ea7",
        "CM-AKF": "#2d8c4e",
        "TinyML-AKF": "#7b3f98",
    }
    for result in results:
        ax.plot(
            time_s,
            result.estimate,
            lw=1.0,
            alpha=0.9,
            color=colors.get(result.name, None),
            label=result.name,
        )
    ax.set_ylabel("Distance (mm)")
    ax.set_title("Raw / Fixed KF / CM-AKF / TinyML-AKF Offline Comparison")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)

    ax = axes[1]
    for result in results:
        ax.plot(
            time_s,
            result.r_values,
            lw=0.9,
            alpha=0.9,
            color=colors.get(result.name, None),
            label=f"{result.name} R",
        )
    ax.set_ylabel("R (mm^2)")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)

    if has_gt:
        ax = axes[2]
        ax.plot(time_s, np.abs(z - gt), color="#d4735e", lw=0.7, alpha=0.6, label="Raw |error|")
        for result in results:
            ax.plot(
                time_s,
                np.abs(result.estimate - gt),
                lw=0.8,
                alpha=0.8,
                color=colors.get(result.name, None),
                label=f"{result.name} |error|",
            )
        ax.set_ylabel("|Error| (mm)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)

    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"[saved] {out_path}")
    if show:
        plt.show()
    plt.close(fig)


def main() -> int:
    args = parse_args()
    input_path = Path(args.csv_path)
    output_base = Path(args.output_dir)
    out_dir = make_output_dir(output_base, input_path)

    print(f"[info] Reading {input_path}")
    df = load_input_csv(input_path)
    time_s, dt_s = build_time_and_dt(df, args.default_dt)
    z = df["tof_distance_mm"].to_numpy(dtype=float)
    u = df["encoder_speed_mms"].to_numpy(dtype=float)
    encoder_distance = compute_encoder_distance(u, dt_s, df)

    gt = None
    if "gt_distance_mm" in df.columns and df["gt_distance_mm"].notna().any():
        gt = df["gt_distance_mm"].to_numpy(dtype=float)
        print("[info] gt_distance_mm found: RMSE/MAE will be computed.")
    else:
        print("[warn] gt_distance_mm not found: RMSE/MAE will be skipped.")

    fixed = run_kf(z, u, dt_s, mode="fixed")
    cm_akf = run_kf(z, u, dt_s, mode="cm_akf")
    results = [fixed, cm_akf]

    tinyml_r = predict_tinyml_r_stub(
        df=df,
        cm_result=cm_akf,
        model_path=Path(args.tinyml_model),
        enable_tinyml=args.enable_tinyml,
    )
    if tinyml_r is not None:
        results.append(run_kf(z, u, dt_s, mode="tinyml_akf", tinyml_r=tinyml_r))

    output_df = build_output_dataframe(df, time_s, dt_s, encoder_distance, results)
    series_path = out_dir / f"{input_path.stem}_offline_4way_timeseries.csv"
    output_df.to_csv(series_path, index=False)
    print(f"[saved] {series_path}")

    metrics_df = compute_metrics(z, gt, results)
    if metrics_df is not None:
        metrics_path = out_dir / "metrics_summary.csv"
        metrics_df.to_csv(metrics_path, index=False)
        print(f"[saved] {metrics_path}")
        print(metrics_df.to_string(index=False, float_format="{:.4f}".format))

    plot_path = out_dir / f"{input_path.stem}_offline_4way_comparison.png"
    save_plot(plot_path, time_s, z, gt, results, show=args.show)

    print("[done] Offline evaluation complete.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        raise SystemExit(1)
