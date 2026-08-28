from __future__ import annotations

import pandas as pd

from alphabench.data.repository import Repository


def test_repository_write_read_roundtrip(tmp_path, clean_panel):
    repo = Repository(tmp_path)

    assert not repo.exists("ohlcv")
    repo.write(clean_panel, "ohlcv")
    assert repo.exists("ohlcv")

    loaded = repo.read("ohlcv")
    pd.testing.assert_frame_equal(loaded.reset_index(drop=True), clean_panel.reset_index(drop=True))


def test_repository_creates_root_dir(tmp_path):
    nested = tmp_path / "a" / "b"
    Repository(nested)
    assert nested.exists()
