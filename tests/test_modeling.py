from __future__ import annotations

import polars as pl

from llamac_research.modeling import load_feature_matrix, select_feature_columns


def test_feature_selection_excludes_labels_and_supports_ppg(tmp_path) -> None:
    path = tmp_path / "features.parquet"
    frame = pl.DataFrame(
        {
            "SubjectID": ["1", "1", "2", "2", "3"],
            "Trial": [1, 2, 1, 2, 1],
            "NoVideo": [1, 12, 23, 34, 45],
            "EmotType": [1, 2, 3, 4, 5],
            "ReportedType": [1, 2, 3, 4, 5],
            "IntendedType": [1, 2, 3, 4, 5],
            "Band_PPG_mean": [0.1, 0.2, 0.3, 0.4, 0.5],
            "Band_GSR_mean": [1.0, 1.1, 1.2, 1.3, 1.4],
        }
    )
    frame.write_parquet(path)
    assert select_feature_columns(frame, "ppg") == ["Band_PPG_mean"]
    all_cols = select_feature_columns(frame, "all")
    assert "Band_PPG_mean" in all_cols
    assert "Band_GSR_mean" in all_cols
    assert "EmotType" not in all_cols
    matrix = load_feature_matrix(path, feature_set="ppg", target="reported")
    assert matrix.x.shape == (5, 1)
    assert matrix.y.tolist() == [1, 2, 3, 4, 5]
