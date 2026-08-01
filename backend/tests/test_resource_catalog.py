"""Unit tests for the free-resource catalog loader and SQLite sync."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from exceptions import ResourceCatalogError  # noqa: E402
from services.database import (  # noqa: E402
    count_resources,
    get_resource_by_id,
    init_db,
    upsert_catalog_resources,
)
from services.resource_catalog import (  # noqa: E402
    catalog_item_to_metadata,
    load_resource_catalog,
)


def _valid_item(**overrides: object) -> dict:
    base = {
        "catalog_id": "mdn-js-test",
        "title": "JavaScript First Steps",
        "source": "MDN",
        "resource_type": "documentation",
        "url": "https://developer.mozilla.org/en-US/docs/Learn/JavaScript/First_steps",
        "description": "Official MDN introduction to JavaScript fundamentals.",
        "topics": ["programming", "javascript"],
        "skills": ["javascript basics"],
        "difficulty": "beginner",
        "estimated_duration_minutes": 45,
        "supported_formats": ["read", "article"],
        "language": "en",
        "is_free": True,
        "requires_account": False,
        "suitable_moods": ["focused", "curious"],
        "suitable_energy_levels": ["medium"],
        "metadata": {"area": "programming"},
    }
    base.update(overrides)
    return base


class ResourceCatalogTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory(prefix="growthos_catalog_")
        self.root = Path(self._tmpdir.name)

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def _write_catalog(self, items: list[dict]) -> Path:
        path = self.root / "catalog.json"
        path.write_text(json.dumps(items), encoding="utf-8")
        return path

    def test_valid_catalog_loads(self) -> None:
        items = load_resource_catalog(BACKEND_ROOT / "data" / "sample_resources.json")
        self.assertGreaterEqual(len(items), 40)
        self.assertTrue(all(item.is_free for item in items))

    def test_duplicate_catalog_id_rejected(self) -> None:
        path = self._write_catalog(
            [
                _valid_item(catalog_id="dup"),
                _valid_item(
                    catalog_id="dup",
                    url="https://developer.mozilla.org/en-US/docs/Learn/HTML",
                ),
            ]
        )
        with self.assertRaises(ResourceCatalogError):
            load_resource_catalog(path)

    def test_duplicate_url_rejected(self) -> None:
        path = self._write_catalog(
            [
                _valid_item(catalog_id="a"),
                _valid_item(catalog_id="b"),
            ]
        )
        with self.assertRaises(ResourceCatalogError):
            load_resource_catalog(path)

    def test_invalid_url_rejected(self) -> None:
        path = self._write_catalog([_valid_item(url="not-a-url")])
        with self.assertRaises(ResourceCatalogError):
            load_resource_catalog(path)

    def test_example_domain_rejected(self) -> None:
        path = self._write_catalog(
            [_valid_item(url="https://example.com/resource")]
        )
        with self.assertRaises(ResourceCatalogError):
            load_resource_catalog(path)

    def test_non_free_rejected(self) -> None:
        path = self._write_catalog([_valid_item(is_free=False)])
        with self.assertRaises(ResourceCatalogError):
            load_resource_catalog(path)

    def test_required_metadata_validated(self) -> None:
        path = self._write_catalog([_valid_item(skills=[])])
        with self.assertRaises(ResourceCatalogError):
            load_resource_catalog(path)

    def test_malformed_json_fails(self) -> None:
        path = self.root / "bad.json"
        path.write_text("{not-json", encoding="utf-8")
        with self.assertRaises(ResourceCatalogError):
            load_resource_catalog(path)

    def test_sync_idempotent_and_stable_ids(self) -> None:
        db_path = self.root / "test.db"
        init_db(db_path)
        items = load_resource_catalog(BACKEND_ROOT / "data" / "sample_resources.json")
        rows = [
            {
                "catalog_id": item.catalog_id,
                "title": item.title,
                "source": item.source,
                "resource_type": item.resource_type,
                "url": str(item.url),
                "description": item.description,
                "difficulty": item.difficulty.value,
                "estimated_duration_minutes": item.estimated_duration_minutes,
                "is_free": True,
                "metadata": catalog_item_to_metadata(item),
            }
            for item in items
        ]
        first = upsert_catalog_resources(rows, db_path=db_path)
        second = upsert_catalog_resources(rows, db_path=db_path)
        self.assertEqual(first, second)
        self.assertEqual(count_resources(db_path=db_path), len(items))
        sample_id = next(iter(first.values()))
        resource = get_resource_by_id(sample_id, db_path=db_path)
        assert resource is not None
        self.assertIn("catalog_id", resource["metadata"])


if __name__ == "__main__":
    unittest.main()
