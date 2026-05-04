from __future__ import annotations

import polars as pl

from llamac_research.labels import add_target_columns, map_novideo_to_intended, validate_emotion_ids


def test_map_novideo_to_intended_boundaries() -> None:
    assert map_novideo_to_intended(1) == 1
    assert map_novideo_to_intended(10) == 1
    assert map_novideo_to_intended(11) == 2
    assert map_novideo_to_intended(20) == 2
    assert map_novideo_to_intended(21) == 3
    assert map_novideo_to_intended(30) == 3
    assert map_novideo_to_intended(31) == 4
    assert map_novideo_to_intended(40) == 4
    assert map_novideo_to_intended(41) == 5
    assert map_novideo_to_intended(50) == 5
    assert map_novideo_to_intended(51) is None


def test_add_target_columns() -> None:
    frame = pl.DataFrame({"NoVideo": [2, 14, 25, 33, 49], "EmotType": [1, 2, 3, 4, 5]})
    out = add_target_columns(frame)
    assert out["IntendedType"].to_list() == [1, 2, 3, 4, 5]
    assert out["ReportedType"].to_list() == [1, 2, 3, 4, 5]
    validate_emotion_ids(out["ReportedType"].to_list())
