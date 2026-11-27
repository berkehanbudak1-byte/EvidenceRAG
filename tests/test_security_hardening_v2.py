"""Security hardening tests for DPAPI vault, per-IP auth rate limiting, and strict SHA-256 CSP."""

from __future__ import annotations

import asyncio
import base64
import json
from pathlib import Path
import re
import tempfile
from types import SimpleNamespace
import unittest

from app.access import (
    AccessMiddleware,
    AuthFailureRateLimiter,
    DEFAULT_STRICT_CSP,
    access_denial,
    build_strict_csp,
)
from app.dpapi_vault import DPAPI_PREFIX, seal_secret, unseal_secret


class TestEvidenceRagSecurityHardening(unittest.TestCase):
    def test_dpapi_seal_and_unseal_roundtrip(self) -> None:
        def fake_runner(action: str, b64_in: str) -> str:
            if action == "protect":
                return base64.b64encode(f"S:{b64_in}".encode("utf-8")).decode("ascii")
            decoded = base64.b64decode(b64_in).decode("utf-8")
            self.assertTrue(decoded.startswith("S:"))
            return decoded[2:]

        sealed = seal_secret("erag-secret-token", runner=fake_runner)
        self.assertTrue(sealed.startswith(DPAPI_PREFIX))
        self.assertNotIn("erag-secret-token", sealed)
        self.assertEqual(unseal_secret(sealed, runner=fake_runner), "erag-secret-token")

    def test_hosted_api_key_auth_enforces_per_ip_rate_limit(self) -> None:
        settings = SimpleNamespace(
            access_profile="hosted",
            public_hosts=("rag.example.com",),
            api_key="correct-secret-key",
        )
        limiter = AuthFailureRateLimiter(max_failures=3, window_seconds=60.0)
        bad_scope = {
            "type": "http",
            "method": "POST",
            "scheme": "https",
            "path": "/ask",
            "headers": [(b"host", b"rag.example.com"), (b"x-api-key", b"wrong")],
            "client": ("203.0.113.10", 44321),
        }
        for _ in range(3):
            denied = access_denial(bad_scope, settings, rate_limiter=limiter)
            self.assertEqual(denied[0], 401)

        # 4th attempt is blocked with 429 even if attacker now guesses the right key
        good_from_blocked_ip = {
            **bad_scope,
            "headers": [(b"host", b"rag.example.com"), (b"x-api-key", b"correct-secret-key")],
        }
        self.assertEqual(access_denial(good_from_blocked_ip, settings, rate_limiter=limiter)[0], 429)

        # Different IP still succeeds with valid key
        other_ip = {**good_from_blocked_ip, "client": ("203.0.113.11", 44322)}
        self.assertIsNone(access_denial(other_ip, settings, rate_limiter=limiter))

    def test_strict_csp_has_no_unsafe_inline_and_middleware_attaches_headers(self) -> None:
        self.assertIn("script-src 'self'", DEFAULT_STRICT_CSP)
        self.assertNotIn("unsafe-inline", DEFAULT_STRICT_CSP)
        self.assertNotIn("unsafe-eval", DEFAULT_STRICT_CSP)
        self.assertIn("object-src 'none'", DEFAULT_STRICT_CSP)
        self.assertIn("frame-ancestors 'none'", DEFAULT_STRICT_CSP)

        # Verify shipped static/index.html has zero inline script/style bodies
        index_html = (Path(__file__).resolve().parents[1] / "static" / "index.html").read_text(encoding="utf-8")
        for attrs, body in re.findall(r"<script\b([^>]*)>([\s\S]*?)</script\s*>", index_html, re.I):
            self.assertRegex(attrs, r"\bsrc\s*=", "All script tags in static/index.html must be external")
            self.assertEqual(body.strip(), "")
        self.assertNotRegex(index_html, r"<style\b", "static/index.html must not contain inline style blocks")

        # Verify dynamic sha256 computation when an inline script exists (cross-platform tmpdir)
        with tempfile.TemporaryDirectory(prefix="erag-csp-") as tmp_dir:
            tmp_html = Path(tmp_dir) / "sample.html"
            tmp_html.write_text("<html><script>console.log(42);</script></html>", encoding="utf-8")
            csp = build_strict_csp(tmp_html)
            self.assertIn("'sha256-", csp)
            self.assertNotIn("unsafe-inline", csp)

        # Verify AccessMiddleware injects headers on valid local GET
        async def dummy_app(scope, receive, send):
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"ok"})

        sent: list[dict] = []

        async def collect(msg):
            sent.append(msg)

        async def noop_receive():
            return {"type": "http.request", "body": b""}

        local_scope = {
            "type": "http",
            "method": "GET",
            "scheme": "http",
            "path": "/",
            "headers": [(b"host", b"127.0.0.1:8000")],
            "client": ("127.0.0.1", 50000),
        }
        asyncio.run(AccessMiddleware(dummy_app)(local_scope, noop_receive, collect))
        resp_headers = {k.lower(): v for k, v in sent[0]["headers"]}
        self.assertEqual(resp_headers[b"x-content-type-options"], b"nosniff")
        self.assertEqual(resp_headers[b"x-frame-options"], b"DENY")
        self.assertIn(b"script-src 'self'", resp_headers[b"content-security-policy"])

    def test_middleware_denial_attaches_security_headers_and_retry_after(self) -> None:
        settings = SimpleNamespace(
            access_profile="hosted",
            public_hosts=("rag.example.com",),
            api_key="correct-secret-key",
        )
        limiter = AuthFailureRateLimiter(max_failures=1, window_seconds=45.0)

        async def unreachable_app(scope, receive, send):
            raise AssertionError("Downstream app must not run on access denial")

        async def noop_receive():
            return {"type": "http.request", "body": b""}

        middleware = AccessMiddleware(unreachable_app, settings=settings, rate_limiter=limiter)
        bad_scope = {
            "type": "http",
            "method": "POST",
            "scheme": "https",
            "path": "/ask",
            "headers": [(b"host", b"rag.example.com"), (b"x-api-key", b"wrong")],
            "client": ("203.0.113.55", 51234),
        }

        sent_401: list[dict] = []

        async def collect_401(msg):
            sent_401.append(msg)

        asyncio.run(middleware(bad_scope, noop_receive, collect_401))
        self.assertEqual(sent_401[0]["status"], 401)
        headers_401 = {k.lower(): v for k, v in sent_401[0]["headers"]}
        self.assertEqual(headers_401[b"x-content-type-options"], b"nosniff")
        self.assertEqual(headers_401[b"x-frame-options"], b"DENY")
        self.assertEqual(headers_401[b"referrer-policy"], b"no-referrer")
        self.assertEqual(json.loads(sent_401[1]["body"].decode("utf-8"))["detail"], "API key required")

        sent_429: list[dict] = []

        async def collect_429(msg):
            sent_429.append(msg)

        asyncio.run(middleware(bad_scope, noop_receive, collect_429))
        self.assertEqual(sent_429[0]["status"], 429)
        headers_429 = {k.lower(): v for k, v in sent_429[0]["headers"]}
        self.assertEqual(headers_429[b"x-content-type-options"], b"nosniff")
        self.assertEqual(headers_429[b"retry-after"], b"45")


if __name__ == "__main__":
    unittest.main()
