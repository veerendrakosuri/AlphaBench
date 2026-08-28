from __future__ import annotations

import tomllib

from alphabench.config import ROOT

BANNED = {
    "sklearn.model_selection.KFold",
    "sklearn.model_selection.train_test_split",
    "sklearn.model_selection.StratifiedKFold",
}


def test_banned_splitters_are_configured_and_enforced():
    """Config-only regression guard: `ruff.lint.select` must include "TID" or
    the `flake8-tidy-imports.banned-api` table below is silently inert — it was
    configured in Stage 0 without "TID" selected, so `ruff check` never
    actually caught a banned import despite the table existing. See also the
    one-time live verification performed in Stage 3 (added a throwaway
    `from sklearn.model_selection import KFold` import, confirmed `ruff check`
    failed with the TID251 message, then removed it)."""
    with open(ROOT / "pyproject.toml", "rb") as f:
        config = tomllib.load(f)

    lint = config["tool"]["ruff"]["lint"]
    assert "TID" in lint["select"], "TID must be selected or banned-api is never enforced"

    banned_api = lint["flake8-tidy-imports"]["banned-api"]
    assert banned_api.keys() >= BANNED
