"""Settings / path-resolution unit tests (no live network)."""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from config import Settings  # noqa: E402


class SqlitePathResolutionTests(unittest.TestCase):
    def test_four_slash_tmp_url_resolves_to_posix_tmp(self) -> None:
        settings = Settings(
            gemini_api_key="test-key",
            database_url="sqlite:////tmp/growthos.db",
        )
        resolved = settings.resolve_sqlite_path()
        self.assertEqual(resolved, Path("/tmp/growthos.db"))
        self.assertEqual(resolved.as_posix(), "/tmp/growthos.db")
        self.assertFalse(resolved.as_posix().startswith(BACKEND_ROOT.as_posix()))
        self.assertNotIn("var/task", resolved.as_posix())

    def test_relative_sqlite_url_stays_under_backend_root(self) -> None:
        settings = Settings(
            gemini_api_key="test-key",
            database_url="sqlite:///./growthos.db",
        )
        resolved = settings.resolve_sqlite_path()
        self.assertEqual(resolved, (BACKEND_ROOT / "growthos.db").resolve())

    def test_serverless_three_slash_tmp_promoted_to_absolute_tmp(self) -> None:
        settings = Settings(
            gemini_api_key="test-key",
            database_url="sqlite:///tmp/growthos.db",
        )
        with patch.dict(os.environ, {"VERCEL": "1"}, clear=False):
            resolved = settings.resolve_sqlite_path()
        self.assertEqual(resolved, Path("/tmp/growthos.db"))

    def test_faiss_posix_tmp_paths_not_prefixed(self) -> None:
        settings = Settings(
            gemini_api_key="test-key",
            faiss_index_path="/tmp/faiss_index/index.faiss",
            faiss_metadata_path="/tmp/faiss_index/metadata.json",
        )
        self.assertEqual(
            settings.resolve_faiss_index_path(),
            Path("/tmp/faiss_index/index.faiss"),
        )
        self.assertEqual(
            settings.resolve_faiss_metadata_path(),
            Path("/tmp/faiss_index/metadata.json"),
        )


if __name__ == "__main__":
    unittest.main()
