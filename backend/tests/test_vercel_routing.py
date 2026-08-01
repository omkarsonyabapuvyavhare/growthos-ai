"""Validate root vercel.json Services routing / path-strip configuration."""

from __future__ import annotations

import json
import re
import sys
import unittest
from pathlib import Path
from urllib.request import urlopen

REPO_ROOT = Path(__file__).resolve().parents[2]
VERCEL_JSON = REPO_ROOT / "vercel.json"
OPENAPI_SCHEMA_URL = "https://openapi.vercel.sh/vercel.json"


def _load_vercel() -> dict:
    return json.loads(VERCEL_JSON.read_text(encoding="utf-8"))


def _path_to_regexp(pattern: str) -> re.Pattern[str]:
    """
    Minimal path-to-regexp subset for our vercel.json patterns.

    Supports:
    - /api/backend
    - /api/backend/:path*
    - /(.*)
    """
    if pattern == "/api/backend":
        return re.compile(r"^/api/backend$")
    if pattern == "/api/backend/:path*":
        return re.compile(r"^/api/backend(?:/(?P<path>.*))?$")
    if pattern == "/(.*)":
        return re.compile(r"^/(?P<_rest>.*)$")
    raise AssertionError(f"Unsupported test pattern: {pattern}")


def _apply_path_transform(public_path: str, source: str, args: str) -> str:
    match = _path_to_regexp(source).fullmatch(public_path)
    if match is None:
        raise AssertionError(f"{public_path!r} does not match {source!r}")
    if args == "/":
        return "/"
    if args == "/:path*":
        captured = match.groupdict().get("path") or ""
        return "/" + captured if captured else "/"
    raise AssertionError(f"Unsupported transform args: {args!r}")


def _first_matching_rewrite(config: dict, public_path: str) -> dict:
    for rule in config["rewrites"]:
        if _path_to_regexp(rule["source"]).fullmatch(public_path):
            return rule
    raise AssertionError(f"No rewrite matched {public_path!r}")


class VercelRoutingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = _load_vercel()

    def test_services_and_public_routing_present(self) -> None:
        self.assertEqual(set(self.config["services"]), {"frontend", "backend"})
        self.assertEqual(self.config["services"]["frontend"]["framework"], "nextjs")
        self.assertEqual(self.config["services"]["backend"]["framework"], "fastapi")
        self.assertEqual(self.config["services"]["backend"]["entrypoint"], "main:app")

        destinations = [
            rule["destination"].get("service")
            for rule in self.config["rewrites"]
            if isinstance(rule.get("destination"), dict)
        ]
        self.assertIn("backend", destinations)
        self.assertIn("frontend", destinations)

    def test_backend_prefix_stripped_for_fastapi(self) -> None:
        cases = [
            ("/api/backend/health", "/health"),
            ("/api/backend/onboarding", "/onboarding"),
            ("/api/backend/users/1/dashboard", "/users/1/dashboard"),
        ]
        for public_path, expected in cases:
            with self.subTest(public_path=public_path):
                rule = _first_matching_rewrite(self.config, public_path)
                self.assertEqual(rule["destination"]["service"], "backend")
                transforms = rule.get("transforms") or []
                self.assertTrue(transforms, "backend rewrite must set request.path")
                transform = transforms[0]
                self.assertEqual(transform["type"], "request.path")
                self.assertEqual(transform["op"], "set")
                observed = _apply_path_transform(
                    public_path, rule["source"], transform["args"]
                )
                self.assertEqual(observed, expected)

    def test_dashboard_routes_to_frontend(self) -> None:
        rule = _first_matching_rewrite(self.config, "/dashboard")
        self.assertEqual(rule["destination"]["service"], "frontend")

    def test_backend_service_rewrites_strip_prefix(self) -> None:
        backend_rewrites = self.config["services"]["backend"].get("rewrites") or []
        sources = {item["source"]: item["destination"] for item in backend_rewrites}
        self.assertEqual(sources.get("/api/backend"), "/")
        self.assertEqual(sources.get("/api/backend/:path*"), "/:path*")

    def test_vercel_json_matches_openapi_schema_shape(self) -> None:
        """
        Structural validation against https://openapi.vercel.sh/vercel.json.

        Avoids requiring jsonschema; checks required Services + rewrite transform keys.
        """
        try:
            with urlopen(OPENAPI_SCHEMA_URL, timeout=20) as response:
                schema = json.loads(response.read().decode("utf-8"))
        except Exception as exc:  # noqa: BLE001
            self.skipTest(f"OpenAPI schema unavailable: {type(exc).__name__}")

        root_props = schema.get("properties") or {}
        self.assertIn("services", root_props)
        self.assertIn("rewrites", root_props)

        rewrite_props = root_props["rewrites"]["items"]["properties"]
        self.assertIn("transforms", rewrite_props)
        transform_enum = rewrite_props["transforms"]["items"]["properties"]["type"]["enum"]
        self.assertIn("request.path", transform_enum)

        service_props = root_props["services"]["additionalProperties"]["properties"]
        self.assertIn("rewrites", service_props)
        self.assertIn("entrypoint", service_props)
        self.assertIn("framework", service_props)

        # Name pattern from schema
        name_pattern = root_props["services"]["propertyNames"]["pattern"]
        for name in self.config["services"]:
            self.assertRegex(name, name_pattern)

        # Transform args must satisfy schema pattern
        args_pattern = rewrite_props["transforms"]["items"]["properties"]["args"]["pattern"]
        for rule in self.config["rewrites"]:
            for transform in rule.get("transforms") or []:
                self.assertRegex(transform["args"], args_pattern)


if __name__ == "__main__":
    unittest.main()
