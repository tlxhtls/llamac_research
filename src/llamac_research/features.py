"""Feature extraction for LLaMAC trial-wise biosignal modeling.

The all-channel feature set intentionally mirrors the official Figshare notebook:
per-trial EEG, GSR, PPG, SKT, and respiration CSV files are summarized into a
single tabular row and then merged with answer.csv labels. The PPG-only mode uses
only PPG-derived columns from band_*.csv so it can share the same metric harness
without leaking questionnaire features.
"""

from __future__ import annotations

import csv
import math
import re
import warnings
from collections.abc import Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
import polars as pl

from .labels import ANSWER_COLUMNS, add_target_columns

try:  # SciPy is a declared dependency, but keep import-time failure readable.
    from scipy.signal import butter, filtfilt, find_peaks, welch

    SCIPY_OK = True
except Exception:  # pragma: no cover - only hit in broken environments
    SCIPY_OK = False

FeatureMode = Literal["all", "ppg", "ppg_rich"]

PPG_MIN_DIST_SEC = 0.35
RESP_MIN_DIST_SEC = 1.0
GSR_MIN_DIST_SEC = 1.0


@dataclass(frozen=True)
class FeatureBuildSummary:
    """Summary emitted after building a feature table."""

    rows: int
    columns: int
    participants: int
    trials: int
    feature_mode: str
    output_path: str | None

    def to_dict(self) -> dict[str, int | str | None]:
        return {
            "rows": self.rows,
            "columns": self.columns,
            "participants": self.participants,
            "trials": self.trials,
            "feature_mode": self.feature_mode,
            "output_path": self.output_path,
        }


def sanitize_name(name: str) -> str:
    """Normalize CSV column names to stable identifier-like names."""
    name = name.strip()
    name = re.sub(r"[^\w]+", "_", name)
    name = re.sub(r"_+", "_", name)
    return name.strip("_")


def extract_trial_number(path: str | Path) -> int | None:
    """Return the numeric trial suffix from band_12.csv style names."""
    m = re.search(r"_(\d+)\.csv$", Path(path).name)
    return int(m.group(1)) if m else None


def natural_key(path: str | Path) -> list[int | str]:
    """Natural-sort key for participant/trial filenames."""
    text = str(path)
    return [int(tok) if tok.isdigit() else tok.lower() for tok in re.split(r"(\d+)", text)]


def _read_csv(path: str | Path) -> pl.DataFrame:
    """Read CSV through Polars with encoding fallbacks."""
    p = Path(path)
    errors: list[str] = []
    for enc in ("utf8", "utf8-lossy", "cp949", "euc-kr"):
        try:
            df = pl.read_csv(p, encoding=enc, infer_schema_length=256, ignore_errors=True)
            return df.rename({c: sanitize_name(c) for c in df.columns})
        except Exception as exc:
            errors.append(f"{enc}: {exc}")
    raise ValueError(f"failed to read CSV {p}: {'; '.join(errors[:2])}")


def _as_float_array(values: Any) -> np.ndarray:
    """Convert a Polars series/list into a float array, coercing invalid values."""
    if isinstance(values, pl.Series):
        series = values.cast(pl.Float64, strict=False)
        return series.to_numpy().astype(float, copy=False)
    arr = np.asarray(values)
    if arr.dtype.kind in {"f", "i", "u", "b"}:
        return arr.astype(float, copy=False)
    out = np.empty(arr.shape[0], dtype=float)
    for i, item in enumerate(arr):
        try:
            out[i] = float(item)
        except (TypeError, ValueError):
            out[i] = np.nan
    return out


def robust_stats(values: Any, prefix: str) -> dict[str, float | int]:
    """Basic robust statistics used by the official baseline notebook."""
    x = _as_float_array(values)
    x = x[np.isfinite(x)]
    keys = (
        "count",
        "min",
        "max",
        "mean",
        "var",
        "std",
        "median",
        "iqr",
        "q10",
        "q90",
        "skew",
        "kurt",
        "cv",
        "rms",
        "energy",
    )
    if x.size == 0:
        return {f"{prefix}_{k}": math.nan for k in keys}

    q10, q25, q50, q75, q90 = np.quantile(x, [0.10, 0.25, 0.50, 0.75, 0.90])
    mean = float(np.mean(x))
    var = float(np.var(x, ddof=1)) if x.size > 1 else 0.0
    std = float(np.sqrt(var))
    centered = x - mean
    if std > 0 and x.size > 2:
        skew = float(np.mean((centered / std) ** 3))
        kurt = float(np.mean((centered / std) ** 4) - 3.0)
    else:
        skew = math.nan
        kurt = math.nan
    cv = float(std / mean) if mean != 0.0 else math.nan
    rms = float(np.sqrt(np.mean(x**2)))
    return {
        f"{prefix}_count": int(x.size),
        f"{prefix}_min": float(np.min(x)),
        f"{prefix}_max": float(np.max(x)),
        f"{prefix}_mean": mean,
        f"{prefix}_var": var,
        f"{prefix}_std": std,
        f"{prefix}_median": float(q50),
        f"{prefix}_iqr": float(q75 - q25),
        f"{prefix}_q10": float(q10),
        f"{prefix}_q90": float(q90),
        f"{prefix}_skew": skew,
        f"{prefix}_kurt": kurt,
        f"{prefix}_cv": cv,
        f"{prefix}_rms": rms,
        f"{prefix}_energy": float(np.sum(x**2) / x.size),
    }


def parse_time_seconds(values: Any) -> np.ndarray:
    """Convert numeric or timestamp-like columns into seconds."""
    if isinstance(values, pl.Series):
        raw = values.to_list()
    else:
        raw = list(values)
    numeric = _as_float_array(raw)
    if np.isfinite(numeric).sum() >= 2:
        return numeric

    out = np.empty(len(raw), dtype=float)
    out[:] = np.nan
    for idx, item in enumerate(raw):
        if item is None:
            continue
        text = str(item).strip()
        if not text:
            continue
        # Python handles both 'YYYY-mm-dd HH:MM:SS.sss' and ISO forms here.
        try:
            out[idx] = datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()
        except ValueError:
            try:
                out[idx] = np.datetime64(text).astype("datetime64[ns]").astype("int64") / 1e9
            except Exception:
                out[idx] = np.nan
    if np.isfinite(out).sum() >= 2:
        return out
    return np.arange(len(raw), dtype=float)


def duration_and_fs(time_values: Any, n_values: int, prefix: str) -> dict[str, float]:
    """Estimate duration and sampling rate from a time column."""
    t = parse_time_seconds(time_values)
    t = t[np.isfinite(t)]
    if t.size >= 2:
        duration = float(np.max(t) - np.min(t))
        fs = float((n_values - 1) / duration) if duration > 0 and n_values > 1 else math.nan
    else:
        duration = math.nan
        fs = math.nan
    return {f"{prefix}_duration_s": duration, f"{prefix}_fs_hz": fs}


def interpolate_nans(values: np.ndarray) -> np.ndarray:
    """Linearly interpolate NaNs, edge-filling as needed."""
    y = np.asarray(values, dtype=float).copy()
    if y.size == 0:
        return y
    mask = np.isfinite(y)
    if mask.all():
        return y
    if not mask.any():
        y[:] = 0.0
        return y
    idx = np.arange(y.size)
    first = np.flatnonzero(mask)[0]
    last = np.flatnonzero(mask)[-1]
    y[:first] = y[first]
    y[last + 1 :] = y[last]
    y[~mask] = np.interp(idx[~mask], idx[mask], y[mask])
    return y


def safe_trapezoid(y: np.ndarray | None, x: np.ndarray | None) -> float:
    """Numerical integration with finite-value guards."""
    if y is None or x is None:
        return math.nan
    y_arr = np.asarray(y, dtype=float)
    x_arr = np.asarray(x, dtype=float)
    mask = np.isfinite(y_arr) & np.isfinite(x_arr)
    if mask.sum() < 2:
        return math.nan
    return float(np.trapezoid(y_arr[mask], x_arr[mask]))


def linear_slope(time_values: Any, signal_values: Any) -> float:
    """Least-squares linear slope for finite samples."""
    x = parse_time_seconds(time_values)
    y = _as_float_array(signal_values)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return math.nan
    x = x[mask]
    y = y[mask]
    x = x - np.mean(x)
    denom = float(np.dot(x, x))
    if denom == 0.0:
        return math.nan
    return float(np.dot(x, y - np.mean(y)) / denom)


def zcr(signal_values: Any) -> float:
    """Zero-crossing rate after mean-centering."""
    y = _as_float_array(signal_values)
    y = y[np.isfinite(y)]
    if y.size < 2:
        return math.nan
    centered = y - np.mean(y)
    return float(np.mean(np.diff(np.signbit(centered)) != 0))


def bandpass(values: np.ndarray, fs: float, low: float, high: float) -> np.ndarray:
    """Band-pass filter with graceful fallback when SciPy/fs are unavailable."""
    y = interpolate_nans(values)
    if not SCIPY_OK or not np.isfinite(fs) or fs <= 0 or y.size < 8:
        return y
    nyq = fs / 2.0
    high = min(high, nyq * 0.95)
    if low <= 0 or high <= low:
        return y
    try:
        b, a = butter(2, [low / nyq, high / nyq], btype="bandpass")
        return filtfilt(b, a, y)
    except Exception:
        return y


def detect_peaks_adaptive(
    values: np.ndarray,
    fs: float | None,
    min_dist_sec: float,
    prom_frac: float = 0.08,
) -> np.ndarray:
    """Adaptive peak detection used for PPG/GSR/RESP derived features."""
    y = interpolate_nans(values)
    if y.size < 3:
        return np.array([], dtype=int)
    if SCIPY_OK:
        distance = 1
        if fs is not None and np.isfinite(fs) and fs > 0:
            distance = max(1, int(round(fs * min_dist_sec)))
        spread = float(np.nanpercentile(y, 95) - np.nanpercentile(y, 5))
        prominence = spread * prom_frac if spread > 0 else None
        peaks, _ = find_peaks(y, distance=distance, prominence=prominence)
        return peaks.astype(int)

    # Small fallback: strict local maxima with minimum index spacing.
    peaks: list[int] = []
    min_dist = max(1, int(round((fs or 1.0) * min_dist_sec)))
    for i in range(1, y.size - 1):
        if y[i] > y[i - 1] and y[i] >= y[i + 1]:
            if not peaks or i - peaks[-1] >= min_dist:
                peaks.append(i)
    return np.asarray(peaks, dtype=int)


def _welch_psd(values: np.ndarray, fs: float | None) -> tuple[np.ndarray | None, np.ndarray | None]:
    y = interpolate_nans(values)
    if not SCIPY_OK or y.size < 8:
        return None, None
    fs_val = float(fs) if fs is not None and np.isfinite(fs) and fs > 0 else 256.0
    nperseg = min(256, y.size)
    try:
        freq, pxx = welch(y, fs=fs_val, nperseg=nperseg)
    except Exception:
        return None, None
    return freq, pxx


def band_power(freq: np.ndarray | None, pxx: np.ndarray | None, low: float, high: float) -> float:
    if freq is None or pxx is None:
        return math.nan
    mask = (freq >= low) & (freq < high)
    if not np.any(mask):
        return math.nan
    return safe_trapezoid(pxx[mask], freq[mask])


def spectral_entropy(freq: np.ndarray | None, pxx: np.ndarray | None) -> float:
    if freq is None or pxx is None:
        return math.nan
    power = np.asarray(pxx, dtype=float)
    power = power[np.isfinite(power) & (power > 0)]
    total = float(np.sum(power))
    if total <= 0 or power.size == 0:
        return math.nan
    p = power / total
    return float(-np.sum(p * np.log2(p)) / np.log2(p.size)) if p.size > 1 else 0.0


def hjorth_params(values: Any) -> tuple[float, float, float]:
    x = _as_float_array(values)
    x = x[np.isfinite(x)]
    if x.size < 3:
        return math.nan, math.nan, math.nan
    dx = np.diff(x)
    ddx = np.diff(dx)
    var0 = float(np.var(x))
    var1 = float(np.var(dx))
    var2 = float(np.var(ddx))
    activity = var0
    mobility = math.sqrt(var1 / var0) if var0 > 0 else math.nan
    complexity = math.sqrt(var2 / var1) / mobility if var1 > 0 and np.isfinite(mobility) and mobility > 0 else math.nan
    return activity, float(mobility), float(complexity)


def line_length(values: Any) -> float:
    x = _as_float_array(values)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return math.nan
    return float(np.sum(np.abs(np.diff(x))))


def _ppg_features(df: pl.DataFrame, info: dict[str, Any], *, rich: bool = False) -> None:
    if "PPG" not in df.columns:
        return
    ppg = df["PPG"]
    if "PPG_Time" in df.columns:
        info.update(duration_and_fs(df["PPG_Time"], df.height, "Band_PPG"))
        fs = info.get("Band_PPG_fs_hz")
        t_sec = parse_time_seconds(df["PPG_Time"])
    else:
        fs = math.nan
        t_sec = np.arange(df.height, dtype=float)
    y0 = _as_float_array(ppg)
    finite_ratio = float(np.isfinite(y0).mean()) if y0.size else 0.0
    info.update(robust_stats(ppg, "Band_PPG"))

    if finite_ratio >= 0.5:
        fs_val = float(fs) if fs is not None and np.isfinite(fs) and fs > 0 else 30.0
        y = y0 - np.nanmedian(y0)
        y_f = bandpass(y, fs_val, low=0.5, high=8.0)
        peaks = detect_peaks_adaptive(y_f, fs_val, PPG_MIN_DIST_SEC, prom_frac=0.08)
        if peaks.size >= 2 and t_sec.size == y_f.size:
            ibis = np.diff(t_sec[peaks])
            ibis = ibis[np.isfinite(ibis) & (ibis > 0)]
        else:
            ibis = np.array([], dtype=float)
    else:
        ibis = np.array([], dtype=float)

    if ibis.size:
        info["Band_PPG_hr_bpm"] = float(60.0 / np.mean(ibis))
        info["Band_PPG_ibi_mean_s"] = float(np.mean(ibis))
        info["Band_PPG_ibi_sd_s"] = float(np.std(ibis, ddof=1)) if ibis.size > 1 else 0.0
        diff_ibis = np.diff(ibis)
        info["Band_PPG_rmssd_s"] = float(np.sqrt(np.mean(diff_ibis**2))) if diff_ibis.size else math.nan
        info["Band_PPG_pnn50"] = float(np.mean(np.abs(diff_ibis) > 0.05)) if diff_ibis.size else math.nan
    else:
        info["Band_PPG_hr_bpm"] = math.nan
        info["Band_PPG_ibi_mean_s"] = math.nan
        info["Band_PPG_ibi_sd_s"] = math.nan
        info["Band_PPG_rmssd_s"] = math.nan
        info["Band_PPG_pnn50"] = math.nan

    info["Band_PPG_slope"] = linear_slope(t_sec, ppg)
    info["Band_PPG_zcr"] = zcr(ppg)

    if not rich:
        return

    # PPG-only extension: morphology, derivatives, frequency, and coarse temporal dynamics.
    # These are excluded from `mode=all` so the all-channel feature table remains close to
    # the official Figshare notebook baseline.
    y_finite = y0[np.isfinite(y0)]
    if y_finite.size >= 2:
        info.update(robust_stats(np.diff(y_finite), "Band_PPG_diff"))
    else:
        info.update(robust_stats([], "Band_PPG_diff"))
    if y_finite.size >= 3:
        info.update(robust_stats(np.diff(y_finite, n=2), "Band_PPG_diff2"))
    else:
        info.update(robust_stats([], "Band_PPG_diff2"))

    fs_val = float(fs) if fs is not None and np.isfinite(fs) and fs > 0 else 30.0
    y_centered = interpolate_nans(y0 - np.nanmedian(y0)) if y0.size else y0
    peaks = detect_peaks_adaptive(bandpass(y_centered, fs_val, low=0.5, high=8.0), fs_val, PPG_MIN_DIST_SEC, prom_frac=0.08)
    info["Band_PPG_peak_count"] = int(peaks.size)
    duration_s = info.get("Band_PPG_duration_s", math.nan)
    info["Band_PPG_peak_rate_per_min"] = float(peaks.size * 60.0 / duration_s) if np.isfinite(duration_s) and duration_s > 0 else math.nan
    if peaks.size and y0.size:
        peak_values = y0[peaks]
        info.update(robust_stats(peak_values, "Band_PPG_peak_value"))
    else:
        info.update(robust_stats([], "Band_PPG_peak_value"))

    freq, pxx = _welch_psd(y_centered, fs_val)
    total_power = safe_trapezoid(pxx, freq) if freq is not None and pxx is not None else math.nan
    info["Band_PPG_total_power"] = float(total_power) if np.isfinite(total_power) else math.nan
    for band_name, low, high in (
        ("vlf", 0.04, 0.15),
        ("lf", 0.15, 0.40),
        ("resp", 0.40, 0.80),
        ("cardiac_low", 0.80, 2.00),
        ("cardiac_high", 2.00, 4.00),
        ("motion", 4.00, 8.00),
    ):
        bp = band_power(freq, pxx, low, high)
        info[f"Band_PPG_{band_name}_power_abs"] = bp
        info[f"Band_PPG_{band_name}_power_rel"] = bp / total_power if np.isfinite(bp) and np.isfinite(total_power) and total_power > 0 else math.nan
    if freq is not None and pxx is not None and pxx.size:
        valid = np.isfinite(freq) & np.isfinite(pxx)
        info["Band_PPG_dominant_hz"] = float(freq[valid][np.argmax(pxx[valid])]) if np.any(valid) else math.nan
        info["Band_PPG_spec_entropy"] = spectral_entropy(freq, pxx)
    else:
        info["Band_PPG_dominant_hz"] = math.nan
        info["Band_PPG_spec_entropy"] = math.nan

    if y0.size >= 8:
        edges = np.linspace(0, y0.size, 5, dtype=int)
        for idx in range(4):
            lo, hi = edges[idx], edges[idx + 1]
            seg = y0[lo:hi]
            seg_t = t_sec[lo:hi] if t_sec.size == y0.size else np.arange(seg.size, dtype=float)
            finite_seg = seg[np.isfinite(seg)]
            info[f"Band_PPG_seg{idx + 1}_mean"] = float(np.mean(finite_seg)) if finite_seg.size else math.nan
            info[f"Band_PPG_seg{idx + 1}_std"] = float(np.std(finite_seg, ddof=1)) if finite_seg.size > 1 else (0.0 if finite_seg.size else math.nan)
            info[f"Band_PPG_seg{idx + 1}_slope"] = linear_slope(seg_t, seg)
    else:
        for idx in range(4):
            info[f"Band_PPG_seg{idx + 1}_mean"] = math.nan
            info[f"Band_PPG_seg{idx + 1}_std"] = math.nan
            info[f"Band_PPG_seg{idx + 1}_slope"] = math.nan


def summarize_band_csv(path: str | Path, mode: FeatureMode = "all") -> dict[str, Any]:
    """Summarize one band_*.csv file."""
    df = _read_csv(path)
    info: dict[str, Any] = {}

    if mode == "all" and "GSR" in df.columns:
        info.update(robust_stats(df["GSR"], "Band_GSR"))
        gsr_tcol = "GSR_Time" if "GSR_Time" in df.columns else None
        if gsr_tcol:
            info.update(duration_and_fs(df[gsr_tcol], df.height, "Band_GSR"))
            fs_gsr = info.get("Band_GSR_fs_hz")
        else:
            fs_gsr = math.nan
        gsr = _as_float_array(df["GSR"])
        if np.isfinite(gsr).sum() >= 3:
            gsr_i = interpolate_nans(gsr)
            peaks = detect_peaks_adaptive(gsr_i, fs_gsr, GSR_MIN_DIST_SEC)
            if peaks.size:
                base = np.nanpercentile(gsr_i, 5)
                top = np.nanpercentile(gsr_i, 95)
                amp = gsr_i[peaks] - base
                threshold = (top - base) * 0.1
                info["Band_GSR_scr_count"] = int(np.sum(amp > threshold))
                info["Band_GSR_peak_amp_mean"] = float(np.nanmean(amp)) if amp.size else math.nan
                info["Band_GSR_peak_amp_max"] = float(np.nanmax(amp)) if amp.size else math.nan
            else:
                info["Band_GSR_scr_count"] = 0
                info["Band_GSR_peak_amp_mean"] = math.nan
                info["Band_GSR_peak_amp_max"] = math.nan
        else:
            info["Band_GSR_scr_count"] = 0
            info["Band_GSR_peak_amp_mean"] = math.nan
            info["Band_GSR_peak_amp_max"] = math.nan
        info["Band_GSR_slope"] = linear_slope(df[gsr_tcol] if gsr_tcol else np.arange(df.height), df["GSR"])
        info["Band_GSR_zcr"] = zcr(df["GSR"])

    _ppg_features(df, info, rich=(mode == "ppg_rich"))

    if mode == "all" and "SKT" in df.columns:
        info.update(robust_stats(df["SKT"], "Band_SKT"))
        skt_tcol = "SKT_Time" if "SKT_Time" in df.columns else None
        if skt_tcol:
            info.update(duration_and_fs(df[skt_tcol], df.height, "Band_SKT"))
        info["Band_SKT_slope"] = linear_slope(df[skt_tcol] if skt_tcol else np.arange(df.height), df["SKT"])
    return info


def summarize_resp_csv(path: str | Path) -> dict[str, Any]:
    """Summarize one respiration_*.csv file."""
    df = _read_csv(path)
    info: dict[str, Any] = {}
    time_col = next((c for c in ("Time", "Timestamp") if c in df.columns), None)
    force_col = next((c for c in df.columns if c.lower().startswith("force")), None)
    if force_col is None:
        return info

    info.update(robust_stats(df[force_col], "Resp_Force"))
    if time_col:
        info.update(duration_and_fs(df[time_col], df.height, "Resp"))
        t_sec = parse_time_seconds(df[time_col])
        fs = info.get("Resp_fs_hz")
    else:
        t_sec = np.arange(df.height, dtype=float)
        fs = math.nan

    y0 = _as_float_array(df[force_col])
    if np.isfinite(y0).sum() >= 3:
        peaks = detect_peaks_adaptive(interpolate_nans(y0), fs, RESP_MIN_DIST_SEC, prom_frac=0.08)
        if peaks.size >= 2 and t_sec.size == y0.size:
            ibis = np.diff(t_sec[peaks])
            ibis = ibis[np.isfinite(ibis) & (ibis > 0)]
        else:
            ibis = np.array([], dtype=float)
    else:
        ibis = np.array([], dtype=float)

    if ibis.size:
        info["Resp_breath_bpm"] = float(60.0 / np.mean(ibis))
        info["Resp_ibi_mean_s"] = float(np.mean(ibis))
        info["Resp_ibi_sd_s"] = float(np.std(ibis, ddof=1)) if ibis.size > 1 else 0.0
    else:
        info["Resp_breath_bpm"] = math.nan
        info["Resp_ibi_mean_s"] = math.nan
        info["Resp_ibi_sd_s"] = math.nan
    info["Resp_slope"] = linear_slope(t_sec, df[force_col])
    info["Resp_zcr"] = zcr(df[force_col])
    return info


def summarize_eeg_csv(path: str | Path) -> dict[str, Any]:
    """Summarize one eeg_*.csv file."""
    df = _read_csv(path)
    info: dict[str, Any] = {}
    tcol = next((c for c in ("Timestamp", "Time", "TS") if c in df.columns), None)
    if tcol:
        info.update(duration_and_fs(df[tcol], df.height, "EEG"))
        fs = info.get("EEG_fs_hz")
        t_sec = parse_time_seconds(df[tcol])
    else:
        fs = math.nan
        t_sec = np.arange(df.height, dtype=float)

    bands = {
        "delta": (1, 4),
        "theta": (4, 7),
        "alpha": (8, 13),
        "beta": (13, 30),
        "gamma": (30, 45),
    }
    for col in df.columns:
        if col == tcol:
            continue
        values = _as_float_array(df[col])
        prefix = f"EEG_{col}"
        info.update(robust_stats(df[col], prefix))
        activity, mobility, complexity = hjorth_params(values)
        info[f"{prefix}_hjorth_activity"] = activity
        info[f"{prefix}_hjorth_mobility"] = mobility
        info[f"{prefix}_hjorth_complexity"] = complexity
        info[f"{prefix}_line_length"] = line_length(values)
        info[f"{prefix}_zcr"] = zcr(values)
        info[f"{prefix}_slope"] = linear_slope(t_sec, values)

        freq, pxx = _welch_psd(values, fs)
        total_power = safe_trapezoid(pxx, freq) if freq is not None and pxx is not None else math.nan
        info[f"{prefix}_total_power"] = float(total_power) if np.isfinite(total_power) else math.nan
        for band_name, (low, high) in bands.items():
            bp = band_power(freq, pxx, low, high)
            info[f"{prefix}_{band_name}_abs"] = bp
            info[f"{prefix}_{band_name}_rel"] = bp / total_power if np.isfinite(bp) and np.isfinite(total_power) and total_power > 0 else math.nan
        if freq is not None and pxx is not None:
            mask = (freq >= 8) & (freq <= 13)
            info[f"{prefix}_alpha_peak_hz"] = float(freq[mask][np.argmax(pxx[mask])]) if np.any(mask) else math.nan
            info[f"{prefix}_spec_entropy"] = spectral_entropy(freq, pxx)
        else:
            info[f"{prefix}_alpha_peak_hz"] = math.nan
            info[f"{prefix}_spec_entropy"] = math.nan
    return info


def read_answer_csv(subject_dir: str | Path, subject_id: str | int | None = None) -> pl.DataFrame:
    """Read one participant answer.csv and add SubjectID if absent."""
    sdir = Path(subject_dir)
    answer_path = next((sdir / name for name in ("answer.csv", "Answer.csv", "ANSWER.csv") if (sdir / name).is_file()), None)
    if answer_path is None:
        raise FileNotFoundError(f"answer.csv not found under {sdir}")
    frame = _read_csv(answer_path)
    if "Trial" not in frame.columns:
        raise ValueError(f"Trial column missing in {answer_path}")
    if "SubjectID" not in frame.columns:
        frame = frame.with_columns(pl.lit(str(subject_id or sdir.name)).alias("SubjectID"))
    frame = frame.select([c for c in ANSWER_COLUMNS if c in frame.columns])
    return add_target_columns(frame)


def process_subject(subject_dir: str | Path, mode: FeatureMode = "all") -> pl.DataFrame:
    """Build merged label+feature rows for one participant directory."""
    sdir = Path(subject_dir)
    subject_id = sdir.name
    answer = read_answer_csv(sdir, subject_id=subject_id)
    trial_map: dict[int, dict[str, list[Path]]] = {}
    patterns = ["band_*.csv"] if mode in {"ppg", "ppg_rich"} else ["band_*.csv", "eeg_*.csv", "respiration_*.csv"]
    for pattern in patterns:
        for path in sorted(sdir.glob(pattern), key=natural_key):
            trial = extract_trial_number(path)
            if trial is None:
                continue
            bucket = trial_map.setdefault(trial, {"band": [], "eeg": [], "resp": []})
            lower = path.name.lower()
            if lower.startswith("band_"):
                bucket["band"].append(path)
            elif lower.startswith("eeg_"):
                bucket["eeg"].append(path)
            elif lower.startswith("respiration_"):
                bucket["resp"].append(path)

    rows: list[dict[str, Any]] = []
    trials = answer["Trial"].cast(pl.Int64, strict=False).drop_nulls().to_list()
    for trial in sorted(int(t) for t in trials):
        feat: dict[str, Any] = {"Trial": int(trial)}
        paths = trial_map.get(int(trial), {"band": [], "eeg": [], "resp": []})
        for path in paths["band"]:
            feat.update(summarize_band_csv(path, mode=mode))
        if mode == "all":
            for path in paths["eeg"]:
                feat.update(summarize_eeg_csv(path))
            for path in paths["resp"]:
                feat.update(summarize_resp_csv(path))
        rows.append(feat)
    feature_frame = pl.DataFrame(rows) if rows else pl.DataFrame({"Trial": []})
    feature_frame = feature_frame.with_columns(pl.col("Trial").cast(pl.Int64, strict=False))
    answer = answer.with_columns(pl.col("Trial").cast(pl.Int64, strict=False))
    return answer.join(feature_frame, on="Trial", how="left")


def _process_subject_worker(args: tuple[str, str]) -> pl.DataFrame:
    subject_dir, mode = args
    return process_subject(subject_dir, mode=mode)  # type: ignore[arg-type]


def discover_subject_dirs(data_root: str | Path, limit_subjects: int | None = None) -> list[Path]:
    """Discover participant folders, including archives that extracted one level deep.

    Most LLaMAC zip files extract directly to `data/extracted/<id>/answer.csv`,
    but at least one archive can appear as `data/extracted/<id>/<id>/answer.csv`.
    The feature builder treats the directory containing `answer.csv` as the
    participant directory and de-duplicates by resolved path.
    """
    root = Path(data_root)
    subject_dirs: list[Path] = []
    seen: set[Path] = set()
    for answer_path in root.glob("*/answer.csv"):
        parent = answer_path.parent.resolve()
        if parent not in seen:
            seen.add(parent)
            subject_dirs.append(answer_path.parent)
    for answer_path in root.glob("*/*/answer.csv"):
        parent = answer_path.parent.resolve()
        if parent not in seen:
            seen.add(parent)
            subject_dirs.append(answer_path.parent)
    subject_dirs = sorted(subject_dirs, key=natural_key)
    if limit_subjects is not None:
        subject_dirs = subject_dirs[:limit_subjects]
    return subject_dirs


def build_feature_table(
    data_root: str | Path = "data/extracted",
    *,
    mode: FeatureMode = "all",
    limit_subjects: int | None = None,
    workers: int = 1,
    output_path: str | Path | None = None,
) -> tuple[pl.DataFrame, FeatureBuildSummary]:
    """Build a trial-wise feature table from extracted LLaMAC participant folders."""
    subject_dirs = discover_subject_dirs(data_root, limit_subjects=limit_subjects)
    if not subject_dirs:
        raise FileNotFoundError(f"No participant folders with answer.csv under {data_root}")

    frames: list[pl.DataFrame] = []
    if workers <= 1:
        for idx, subject_dir in enumerate(subject_dirs, start=1):
            print(f"[{idx}/{len(subject_dirs)}] subject={subject_dir.name} mode={mode}", flush=True)
            frames.append(process_subject(subject_dir, mode=mode))
    else:
        args = [(str(path), mode) for path in subject_dirs]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_process_subject_worker, arg): Path(arg[0]).name for arg in args}
            for idx, future in enumerate(as_completed(futures), start=1):
                sid = futures[future]
                try:
                    frames.append(future.result())
                    print(f"[{idx}/{len(subject_dirs)}] subject={sid} done", flush=True)
                except Exception as exc:
                    warnings.warn(f"subject {sid} failed: {exc}")

    if not frames:
        raise RuntimeError("No feature frames were produced")
    merged = pl.concat(frames, how="diagonal_relaxed")
    first_cols = [c for c in [*ANSWER_COLUMNS, "IntendedType", "ReportedType"] if c in merged.columns]
    other_cols = [c for c in merged.columns if c not in first_cols]
    merged = merged.select(first_cols + sorted(other_cols))

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.suffix.lower() == ".parquet":
            merged.write_parquet(out)
        else:
            merged.write_csv(out)
    summary = FeatureBuildSummary(
        rows=merged.height,
        columns=merged.width,
        participants=merged.select(pl.col("SubjectID").n_unique()).item() if "SubjectID" in merged.columns else 0,
        trials=merged.select(pl.col("Trial").n_unique()).item() if "Trial" in merged.columns else 0,
        feature_mode=mode,
        output_path=str(output_path) if output_path is not None else None,
    )
    return merged, summary


def read_feature_table(path: str | Path) -> pl.DataFrame:
    """Read a generated feature table from CSV or parquet."""
    p = Path(path)
    if p.suffix.lower() == ".parquet":
        return pl.read_parquet(p)
    return pl.read_csv(p, infer_schema_length=1024, ignore_errors=True)


def write_summary_csv(rows: Sequence[dict[str, Any]], output_path: str | Path) -> None:
    """Write compact dict rows as CSV without requiring pandas."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
