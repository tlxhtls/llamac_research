from __future__ import annotations

from pathlib import Path

import numpy as np

from llamac_research.features import build_feature_table, summarize_band_csv


def _write_subject(root: Path, subject: int, trials: int = 5) -> None:
    sdir = root / str(subject)
    sdir.mkdir(parents=True)
    lines = ["Trial,NoVideo,Valence,Arousal,Dominance,Liking,EmotType,EmotStr,Seen"]
    for trial in range(1, trials + 1):
        lines.append(f"{trial},{trial},3,3,3,3,{trial},3,1")
        t = np.arange(40, dtype=float) * 0.04
        ppg = np.sin(2 * np.pi * 1.2 * t) + trial
        gsr = np.linspace(1.0, 1.5, t.size)
        skt = np.linspace(32.0, 32.2, t.size)
        with (sdir / f"band_{trial}.csv").open("w", encoding="utf-8") as f:
            f.write("GSR,GSR_Time,PPG,PPG_Time,SKT,SKT_Time\n")
            for i in range(t.size):
                f.write(f"{gsr[i]},{t[i]},{ppg[i]},{t[i]},{skt[i]},{t[i]}\n")
    (sdir / "answer.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_summarize_band_csv_has_ppg_features(tmp_path: Path) -> None:
    _write_subject(tmp_path, 1)
    features = summarize_band_csv(tmp_path / "1" / "band_1.csv", mode="ppg")
    assert "Band_PPG_mean" in features
    assert "Band_PPG_hr_bpm" in features
    assert "Band_GSR_mean" not in features


def test_build_ppg_feature_table(tmp_path: Path) -> None:
    _write_subject(tmp_path, 1)
    frame, summary = build_feature_table(tmp_path, mode="ppg", output_path=tmp_path / "features.parquet")
    assert summary.rows == 5
    assert summary.participants == 1
    assert "ReportedType" in frame.columns
    assert any(c.startswith("Band_PPG_") for c in frame.columns)
    assert not any(c.startswith("Band_GSR_") for c in frame.columns)


def test_discover_nested_extracted_subject(tmp_path: Path) -> None:
    # Some Figshare archives extract as data/extracted/<id>/<id>/answer.csv.
    nested_root = tmp_path / "outer"
    _write_subject(nested_root, 9)
    frame, summary = build_feature_table(tmp_path, mode="ppg")
    assert summary.participants == 1
    assert summary.rows == 5
    assert set(frame["SubjectID"].to_list()) == {"9"}


def test_build_rich_ppg_adds_ppg_only_extensions(tmp_path: Path) -> None:
    _write_subject(tmp_path, 1)
    base, _ = build_feature_table(tmp_path, mode="ppg")
    rich, _ = build_feature_table(tmp_path, mode="ppg_rich")
    assert rich.width > base.width
    assert "Band_PPG_diff_mean" in rich.columns
    assert "Band_PPG_total_power" in rich.columns
    assert not any(c.startswith("Band_GSR_") for c in rich.columns)
