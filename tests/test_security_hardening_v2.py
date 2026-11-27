"""Security hardening tests for DPAPI vault, per-IP auth rate limiting, and strict SHA-256 CSP."""

from __future__ import annotations

import asyncio
import base64
import contextlib
import io
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
import warnings

from app.access import (
    AccessMiddleware,
    AuthFailureRateLimiter,
    DEFAULT_STRICT_CSP,
    access_denial,
    build_strict_csp,
)
from app.dpapi_vault import DPAPI_PREFIX, seal_secret, unseal_secret
from app.rag import ocr
from scripts import dpapi_vault_cli

try:
    from PIL import Image as _PIL_IMAGE, UnidentifiedImageError as _PIL_UNIDENTIFIED_ERROR
except ImportError:
    _PIL_IMAGE = None
    _PIL_UNIDENTIFIED_ERROR = None

_HAS_PIL = _PIL_IMAGE is not None


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


class TestDpapiVaultCli(unittest.TestCase):
    @staticmethod
    def _fake_dpapi_runner(action: str, b64_in: str) -> str:
        if action == "protect":
            return base64.b64encode(f"VAULT:{b64_in}".encode("utf-8")).decode("ascii")
        if action == "unprotect":
            decoded = base64.b64decode(b64_in).decode("utf-8")
            if not decoded.startswith("VAULT:"):
                raise ValueError("Invalid ciphertext marker")
            return decoded[len("VAULT:") :]
        raise ValueError(f"Unsupported action: {action}")

    def test_status_reports_sealed_plaintext_and_unset_without_leaking_secrets(self) -> None:
        sealed_api = seal_secret("erag-top-secret-alpha", runner=self._fake_dpapi_runner)
        plain_xai = "xai-plain-secret-beta-999"
        plain_grok = "grok-env-secret-gamma-777"

        with tempfile.TemporaryDirectory(prefix="erag-vault-status-") as tmp_dir:
            env_path = Path(tmp_dir) / ".env"
            env_path.write_text(
                "\n".join(
                    [
                        "# EvidenceRAG local config",
                        "ACCESS_PROFILE=hosted",
                        f"EVIDENCERAG_API_KEY={sealed_api}",
                        f'XAI_API_KEY="{plain_xai}"  # inline note',
                        "GROK_API_KEY=",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            env_map = {"GROK_API_KEY": plain_grok}

            # JSON status check
            stdout_json = io.StringIO()
            stderr_json = io.StringIO()
            with contextlib.redirect_stdout(stdout_json), contextlib.redirect_stderr(stderr_json):
                rc_json = dpapi_vault_cli.main(
                    ["status", "--env-file", str(env_path), "--json"],
                    environ=env_map,
                )
            self.assertEqual(rc_json, 0)
            self.assertEqual(stderr_json.getvalue(), "")
            raw_json = stdout_json.getvalue()
            report = json.loads(raw_json)
            self.assertTrue(report["ok"])
            self.assertTrue(report["env_file_exists"])
            self.assertTrue(report["has_plaintext"])
            self.assertEqual(
                report["keys"]["EVIDENCERAG_API_KEY"],
                {"env": "unset", "dotenv": "sealed", "effective": "sealed", "source": "dotenv"},
            )
            self.assertEqual(
                report["keys"]["XAI_API_KEY"],
                {"env": "unset", "dotenv": "plaintext", "effective": "plaintext", "source": "dotenv"},
            )
            self.assertEqual(
                report["keys"]["GROK_API_KEY"],
                {"env": "plaintext", "dotenv": "unset", "effective": "plaintext", "source": "env"},
            )

            # Text status check
            stdout_txt = io.StringIO()
            stderr_txt = io.StringIO()
            with contextlib.redirect_stdout(stdout_txt), contextlib.redirect_stderr(stderr_txt):
                rc_txt = dpapi_vault_cli.main(
                    ["status", "--env-file", str(env_path)],
                    environ=env_map,
                )
            self.assertEqual(rc_txt, 0)
            self.assertEqual(stderr_txt.getvalue(), "")
            raw_txt = stdout_txt.getvalue()
            self.assertIn("EVIDENCERAG_API_KEY: effective=sealed", raw_txt)
            self.assertIn("XAI_API_KEY: effective=plaintext", raw_txt)
            self.assertIn("GROK_API_KEY: effective=plaintext", raw_txt)

            # Zero secret or ciphertext leakage in either output format
            for forbidden in ("erag-top-secret-alpha", plain_xai, plain_grok, sealed_api, DPAPI_PREFIX):
                self.assertNotIn(forbidden, raw_json)
                self.assertNotIn(forbidden, raw_txt)

            # Missing .env file still succeeds and reports env-only status
            missing_env = Path(tmp_dir) / "does-not-exist.env"
            stdout_missing = io.StringIO()
            with contextlib.redirect_stdout(stdout_missing):
                rc_missing = dpapi_vault_cli.main(
                    ["status", "--env-file", str(missing_env), "--json"],
                    environ={"EVIDENCERAG_API_KEY": sealed_api},
                )
            self.assertEqual(rc_missing, 0)
            missing_report = json.loads(stdout_missing.getvalue())
            self.assertFalse(missing_report["env_file_exists"])
            self.assertFalse(missing_report["has_plaintext"])
            self.assertEqual(missing_report["keys"]["EVIDENCERAG_API_KEY"]["effective"], "sealed")
            self.assertEqual(missing_report["keys"]["XAI_API_KEY"]["effective"], "unset")
            self.assertEqual(missing_report["keys"]["GROK_API_KEY"]["effective"], "unset")

    def test_seal_rewrites_plaintext_atomically_sets_0600_and_never_leaks(self) -> None:
        plain_api = "erag-plain-secret-001"
        plain_xai = "xai-plain#hash-secret-002"
        plain_grok = "grok-plain-secret-003"
        already_sealed = seal_secret("pre-existing-sealed-004", runner=self._fake_dpapi_runner)

        with tempfile.TemporaryDirectory(prefix="erag-vault-seal-") as tmp_dir:
            env_path = Path(tmp_dir) / ".env"
            env_path.write_text(
                "\n".join(
                    [
                        "# Header comment preserved",
                        "ACCESS_PROFILE=hosted",
                        f"EVIDENCERAG_API_KEY={plain_api}",
                        f'XAI_API_KEY="{plain_xai}"  # prod key',
                        f"export GROK_API_KEY='{plain_grok}'",
                        "LLM_PROVIDER=xai",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            os.chmod(env_path, 0o644)

            chmod_calls: list[tuple[Path, int]] = []
            real_chmod = os.chmod

            def spy_chmod(path: os.PathLike[str] | str, mode: int, *args: object, **kwargs: object) -> None:
                chmod_calls.append((Path(path), mode))
                real_chmod(path, mode, *args, **kwargs)

            stdout_seal = io.StringIO()
            stderr_seal = io.StringIO()
            with mock.patch("scripts.dpapi_vault_cli.os.chmod", side_effect=spy_chmod):
                with contextlib.redirect_stdout(stdout_seal), contextlib.redirect_stderr(stderr_seal):
                    rc = dpapi_vault_cli.main(
                        ["seal", "--env-file", str(env_path), "--json"],
                        runner=self._fake_dpapi_runner,
                    )
            self.assertEqual(rc, 0)
            self.assertEqual(stderr_seal.getvalue(), "")

            seal_out = stdout_seal.getvalue()
            result = json.loads(seal_out)
            self.assertTrue(result["ok"])
            self.assertTrue(result["changed"])
            self.assertEqual(result["sealed_keys"], ["EVIDENCERAG_API_KEY", "XAI_API_KEY", "GROK_API_KEY"])
            self.assertEqual(result["already_sealed_keys"], [])
            self.assertEqual(result["unset_keys"], [])
            self.assertEqual(result["permissions"], "0600")

            # Verify 0o600 was applied and no leftover temporary files remain
            self.assertTrue(any(mode == 0o600 for _, mode in chmod_calls))
            if os.name == "posix":
                self.assertEqual(stat.S_IMODE(env_path.stat().st_mode), 0o600)
            self.assertEqual([p.name for p in Path(tmp_dir).iterdir()], [".env"])

            # Verify disk content has zero plaintext, preserves structure, and round-trips via unseal_secret
            disk_text = env_path.read_text(encoding="utf-8")
            for secret in (plain_api, plain_xai, plain_grok):
                self.assertNotIn(secret, disk_text)
                self.assertNotIn(secret, seal_out)
            self.assertNotIn(DPAPI_PREFIX, seal_out)
            self.assertIn("# Header comment preserved", disk_text)
            self.assertIn("ACCESS_PROFILE=hosted", disk_text)
            self.assertIn("LLM_PROVIDER=xai", disk_text)
            self.assertIn("  # prod key", disk_text)
            self.assertIn("export GROK_API_KEY=dpapi:v1:", disk_text)

            parsed = dpapi_vault_cli.parse_env_file_values(env_path)
            self.assertTrue(parsed["EVIDENCERAG_API_KEY"].startswith(DPAPI_PREFIX))
            self.assertTrue(parsed["XAI_API_KEY"].startswith(DPAPI_PREFIX))
            self.assertTrue(parsed["GROK_API_KEY"].startswith(DPAPI_PREFIX))
            self.assertEqual(unseal_secret(parsed["EVIDENCERAG_API_KEY"], runner=self._fake_dpapi_runner), plain_api)
            self.assertEqual(unseal_secret(parsed["XAI_API_KEY"], runner=self._fake_dpapi_runner), plain_xai)
            self.assertEqual(unseal_secret(parsed["GROK_API_KEY"], runner=self._fake_dpapi_runner), plain_grok)

            # Second seal pass is idempotent (changed=False, already_sealed preserved)
            stdout_idem = io.StringIO()
            with contextlib.redirect_stdout(stdout_idem):
                rc_idem = dpapi_vault_cli.main(
                    ["seal", "--env-file", str(env_path), "--json"],
                    runner=self._fake_dpapi_runner,
                )
            self.assertEqual(rc_idem, 0)
            idem = json.loads(stdout_idem.getvalue())
            self.assertFalse(idem["changed"])
            self.assertEqual(idem["sealed_keys"], [])
            self.assertEqual(idem["already_sealed_keys"], ["EVIDENCERAG_API_KEY", "XAI_API_KEY", "GROK_API_KEY"])
            self.assertEqual(env_path.read_text(encoding="utf-8"), disk_text)

            # Status after sealing reports zero plaintext
            status_after = dpapi_vault_cli.collect_vault_status(env_path, environ={})
            self.assertFalse(status_after["has_plaintext"])
            for k in dpapi_vault_cli.SECRET_ENV_KEYS:
                self.assertEqual(status_after["keys"][k]["dotenv"], "sealed")

            # Verify mixed file with already-sealed + empty key leaves both intact
            mixed_path = Path(tmp_dir) / "mixed.env"
            mixed_path.write_text(
                f"EVIDENCERAG_API_KEY={already_sealed}\nXAI_API_KEY=\n",
                encoding="utf-8",
            )
            mixed_res = dpapi_vault_cli.seal_env_file(mixed_path, runner=self._fake_dpapi_runner)
            self.assertFalse(mixed_res["changed"])
            self.assertEqual(mixed_res["already_sealed_keys"], ["EVIDENCERAG_API_KEY"])
            self.assertEqual(mixed_res["unset_keys"], ["XAI_API_KEY", "GROK_API_KEY"])

    def test_seal_fails_closed_without_leaking_secrets_on_error_or_unavailable_dpapi(self) -> None:
        plain_secret = "xai-never-leak-this-secret-4242"

        with tempfile.TemporaryDirectory(prefix="erag-vault-fail-") as tmp_dir:
            env_path = Path(tmp_dir) / ".env"
            original = f"XAI_API_KEY={plain_secret}\n"
            env_path.write_text(original, encoding="utf-8")

            # 1. Runner raises error containing the plaintext secret -> stripped from stderr, file unchanged
            def crashing_runner(_action: str, _b64_in: str) -> str:
                raise RuntimeError(f"powershell failure with {plain_secret}")

            out1, err1 = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(out1), contextlib.redirect_stderr(err1):
                rc1 = dpapi_vault_cli.main(["seal", "--env-file", str(env_path)], runner=crashing_runner)
            self.assertEqual(rc1, 1)
            self.assertEqual(out1.getvalue(), "")
            self.assertNotIn(plain_secret, err1.getvalue())
            self.assertIn("XAI_API_KEY", err1.getvalue())
            self.assertEqual(env_path.read_text(encoding="utf-8"), original)
            self.assertEqual([p.name for p in Path(tmp_dir).iterdir()], [".env"])

            # 2. DPAPI unavailable and no runner injected -> fails closed, file unchanged
            out2, err2 = io.StringIO(), io.StringIO()
            with mock.patch("scripts.dpapi_vault_cli.is_dpapi_available", return_value=False):
                with contextlib.redirect_stdout(out2), contextlib.redirect_stderr(err2):
                    rc2 = dpapi_vault_cli.main(["seal", "--env-file", str(env_path)])
            self.assertEqual(rc2, 1)
            self.assertNotIn(plain_secret, err2.getvalue())
            self.assertEqual(env_path.read_text(encoding="utf-8"), original)

            # 3. Runner returns empty string (causing seal_secret fallback) -> fails closed
            out3, err3 = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(out3), contextlib.redirect_stderr(err3):
                rc3 = dpapi_vault_cli.main(
                    ["seal", "--env-file", str(env_path)],
                    runner=lambda _a, _b: "",
                )
            self.assertEqual(rc3, 1)
            self.assertNotIn(plain_secret, err3.getvalue())
            self.assertEqual(env_path.read_text(encoding="utf-8"), original)

            # 4. Missing .env file returns exit code 1 cleanly
            out4, err4 = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(out4), contextlib.redirect_stderr(err4):
                rc4 = dpapi_vault_cli.main(
                    ["seal", "--env-file", str(Path(tmp_dir) / "missing.env")],
                    runner=self._fake_dpapi_runner,
                )
            self.assertEqual(rc4, 1)
            self.assertIn("not found", err4.getvalue().lower())

    def test_cli_standalone_subprocess_status_uses_stdlib_only(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        cli_script = repo_root / "scripts" / "dpapi_vault_cli.py"
        plain_secret = "erag-subprocess-plain-secret-888"

        with tempfile.TemporaryDirectory(prefix="erag-vault-subproc-") as tmp_dir:
            env_path = Path(tmp_dir) / ".env"
            env_path.write_text(f"EVIDENCERAG_API_KEY={plain_secret}\n", encoding="utf-8")

            child_env = {
                k: v
                for k, v in os.environ.items()
                if k.upper() not in dpapi_vault_cli.SECRET_ENV_KEYS
            }
            child_env["PYTHONDONTWRITEBYTECODE"] = "1"
            proc = subprocess.run(
                [sys.executable, "-B", str(cli_script), "status", "--env-file", str(env_path), "--json"],
                capture_output=True,
                text=True,
                env=child_env,
                check=False,
                timeout=15,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            payload = json.loads(proc.stdout)
            self.assertEqual(payload["keys"]["EVIDENCERAG_API_KEY"]["effective"], "plaintext")
            self.assertNotIn(plain_secret, proc.stdout)
            self.assertNotIn(plain_secret, proc.stderr)


class TestOcrUpload2ShellHardening(unittest.TestCase):
    _EPS_ASCII = b"%!PS-Adobe-3.0 EPSF-3.0\n%%BoundingBox: 0 0 1 1\nshowpage\n"
    _EPS_BINARY = b"\xc5\xd0\xd3\xc6" + b"\x00" * 28
    _PDF_BYTES = b"%PDF-1.7\n1 0 obj\n<<>>\nendobj\n"

    @staticmethod
    def _encode_image(fmt: str, size: tuple[int, int] = (20, 20), mode: str = "RGB") -> bytes:
        assert _PIL_IMAGE is not None
        color = 0 if mode == "1" else (255 if mode == "L" else (12, 34, 56))
        buf = io.BytesIO()
        _PIL_IMAGE.new(mode, size, color).save(buf, fmt)
        return buf.getvalue()

    def test_constants_and_exception_hierarchy(self) -> None:
        self.assertEqual(ocr.PILLOW_MAX_IMAGE_PIXELS, 16_000_000)
        self.assertEqual(ocr.PILLOW_ALLOWED_FORMATS, ("PNG", "JPEG", "WEBP"))
        self.assertEqual(ocr.PILLOW_MAX_BYTES, 10 * 1024 * 1024)
        self.assertEqual(ocr.FORBIDDEN_IMAGE_SIGNATURES, (b"%!PS", b"\xc5\xd0\xd3\xc6", b"%PDF"))
        self.assertTrue(issubclass(ocr.UnsafeImageError, ValueError))

    def test_validate_image_bytes_rejects_forbidden_signatures_and_invalid_inputs(self) -> None:
        forbidden_payloads = [
            self._EPS_ASCII,
            b" \t\r\n" + self._EPS_ASCII,
            b"\x00\x00" + self._EPS_ASCII,
            b"\x04" + self._EPS_ASCII,
            b"\xef\xbb\xbf" + self._EPS_ASCII,
            self._EPS_BINARY,
            b"\x00" + self._EPS_BINARY,
            self._PDF_BYTES,
            b" \n" + self._PDF_BYTES,
            b"\x00\x00" + self._PDF_BYTES,
            b"",
            bytearray(),
            b"\x89PNG\r\n\x1a\n" + b"x" * ocr.PILLOW_MAX_BYTES,
        ]
        for payload in forbidden_payloads:
            with self.assertRaises(ocr.UnsafeImageError):
                ocr.validate_image_bytes(payload)

        for non_bytes in (None, "not-bytes", 12345, memoryview(b"\x89PNG\r\n\x1a\n")):
            with self.assertRaises(ocr.UnsafeImageError):
                ocr.validate_image_bytes(non_bytes)

        for bad_max_bytes in (True, False, 0, -1, "1024", 1.5, None):
            with self.assertRaises(ValueError):
                ocr.validate_image_bytes(b"\x89PNG\r\n\x1a\n", max_bytes=bad_max_bytes)

        sample = bytearray(b"\x89PNG\r\n\x1a\nsample")
        validated = ocr.validate_image_bytes(sample)
        self.assertIsInstance(validated, bytes)
        self.assertEqual(validated, bytes(sample))

    def test_harden_pillow_with_stub_sets_limit_calls_init_and_strips_eps(self) -> None:
        init_calls: list[bool] = []

        class _StubImageModule:
            MAX_IMAGE_PIXELS = 89_478_485
            OPEN = {"PNG": object(), "EPS": object(), "JPEG": object(), "WEBP": object()}
            ID = ["EPS", "PNG", "EPS", "JPEG", "WEBP"]

            @classmethod
            def init(cls) -> None:
                init_calls.append(True)
                cls.OPEN["EPS"] = object()
                cls.ID.append("EPS")

        returned = ocr.harden_pillow(_StubImageModule)
        self.assertIs(returned, _StubImageModule)
        self.assertEqual(init_calls, [True])
        self.assertEqual(_StubImageModule.MAX_IMAGE_PIXELS, 16_000_000)
        self.assertNotIn("EPS", _StubImageModule.OPEN)
        self.assertNotIn("EPS", _StubImageModule.ID)
        self.assertIn("PNG", _StubImageModule.OPEN)
        self.assertEqual(_StubImageModule.ID, ["PNG", "JPEG", "WEBP"])

    def test_no_pil_simulation_blocks_forbidden_signatures_and_reports_unavailable(self) -> None:
        benign_png_stub = b"\x89PNG\r\n\x1a\nfake-png-body"
        with mock.patch.dict(
            sys.modules,
            {"PIL": None, "PIL.Image": None, "numpy": None, "easyocr": None},
        ):
            for malicious in (self._EPS_ASCII, self._EPS_BINARY, self._PDF_BYTES, b"\x00" + self._EPS_ASCII):
                with self.assertRaises(ocr.UnsafeImageError):
                    ocr.open_image_safely(malicious)

            for bad_max_pixels in (True, False, 0, -10, "100", 12.5):
                with self.assertRaises(ValueError):
                    ocr.open_image_safely(benign_png_stub, max_pixels=bad_max_pixels)

            with self.assertRaises(ocr.OCRUnavailableError):
                ocr.open_image_safely(benign_png_stub)

            with self.assertRaises(ocr.OCRUnavailableError):
                ocr.harden_pillow()

            self.assertEqual(ocr.read_page_images(SimpleNamespace(images=[])), "")

            for malicious in (self._EPS_ASCII, self._EPS_BINARY, self._PDF_BYTES, b""):
                malicious_page = SimpleNamespace(images=[SimpleNamespace(data=malicious)])
                with self.assertRaises(ocr.UnsafeImageError):
                    ocr.read_page_images(malicious_page)

            benign_page = SimpleNamespace(images=[SimpleNamespace(data=benign_png_stub)])
            with self.assertRaises(ocr.OCRUnavailableError):
                ocr.read_page_images(benign_page)

    @unittest.skipUnless(_HAS_PIL, "Pillow not installed")
    def test_open_image_safely_allows_safe_formats_strips_eps_rejects_gif_and_keeps_pdf_save(self) -> None:
        assert _PIL_IMAGE is not None and _PIL_UNIDENTIFIED_ERROR is not None
        for fmt in ("PNG", "JPEG", "WEBP"):
            raw = self._encode_image(fmt, size=(20, 20))
            with ocr.open_image_safely(raw) as opened:
                self.assertEqual(opened.format, fmt)
                self.assertEqual(opened.size, (20, 20))

        self.assertEqual(_PIL_IMAGE.MAX_IMAGE_PIXELS, 16_000_000)
        self.assertNotIn("EPS", _PIL_IMAGE.OPEN)
        self.assertNotIn("EPS", _PIL_IMAGE.ID)

        gif_raw = self._encode_image("GIF", size=(16, 16))
        with self.assertRaises(_PIL_UNIDENTIFIED_ERROR):
            ocr.open_image_safely(gif_raw)

        # Bare Image.open must also refuse EPS after harden_pillow strips the EPS plugin.
        with self.assertRaises(_PIL_UNIDENTIFIED_ERROR):
            _PIL_IMAGE.open(io.BytesIO(self._EPS_ASCII))

        # Saving synthetic PDF fixtures via Pillow continues to work after EPS stripping.
        pdf_buf = io.BytesIO()
        _PIL_IMAGE.new("L", (20, 20), 255).save(pdf_buf, "PDF")
        self.assertTrue(pdf_buf.getvalue().startswith(b"%PDF-"))

    @unittest.skipUnless(_HAS_PIL, "Pillow not installed")
    def test_open_image_safely_enforces_decompression_bomb_limits_including_warning_band(self) -> None:
        assert _PIL_IMAGE is not None
        png_20x20 = self._encode_image("PNG", size=(20, 20))

        # >2x limit (400 px vs 100 px)
        with self.assertRaises(_PIL_IMAGE.DecompressionBombError):
            ocr.open_image_safely(png_20x20, max_pixels=100)

        # 1x..2x Pillow warning-only band (400 px vs 250 px) must still raise hard error before load()
        with self.assertRaises(_PIL_IMAGE.DecompressionBombError):
            ocr.open_image_safely(png_20x20, max_pixels=250)

        # Exact boundary succeeds
        with ocr.open_image_safely(png_20x20, max_pixels=400) as exact:
            self.assertEqual(exact.size, (20, 20))

        # Real 20.25 MP PNG sits inside Pillow's 1x..2x warning band above 16 MP;
        # open_image_safely must reject it even if caller requests a higher max_pixels.
        bomb_20mp = self._encode_image("PNG", size=(4500, 4500), mode="1")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", _PIL_IMAGE.DecompressionBombWarning)
            with self.assertRaises(_PIL_IMAGE.DecompressionBombError):
                ocr.open_image_safely(bomb_20mp)
            with self.assertRaises(_PIL_IMAGE.DecompressionBombError):
                ocr.open_image_safely(bomb_20mp, max_pixels=50_000_000)

    @unittest.skipUnless(_HAS_PIL, "Pillow not installed")
    def test_read_page_images_extracts_safe_skips_benign_and_propagates_unsafe(self) -> None:
        assert _PIL_IMAGE is not None
        fake_reader = SimpleNamespace(
            readtext=lambda arr, detail=0, paragraph=True: ["Metin okundu"]
        )

        # Use real numpy when available, or a tiny pure-Python stub when running without numpy.
        numpy_patch = contextlib.nullcontext()
        try:
            import numpy as _  # noqa: F401
        except ImportError:
            numpy_stub = SimpleNamespace(array=lambda img: img.tobytes())
            numpy_patch = mock.patch.dict(sys.modules, {"numpy": numpy_stub})

        png_bytes = self._encode_image("PNG", size=(16, 16))
        jpeg_bytes = self._encode_image("JPEG", size=(16, 16))
        webp_bytes = self._encode_image("WEBP", size=(16, 16))
        gif_bytes = self._encode_image("GIF", size=(16, 16))
        jbig2_opaque = b"\x97\x4a\x42\x32\x0d\x0a\x1a\x0a" + b"\x00" * 32
        bomb_20mp = self._encode_image("PNG", size=(4500, 4500), mode="1")

        with numpy_patch, mock.patch.object(ocr, "get_reader", return_value=fake_reader):
            for safe_bytes in (png_bytes, jpeg_bytes, webp_bytes):
                page = SimpleNamespace(images=[SimpleNamespace(data=safe_bytes)])
                self.assertEqual(ocr.read_page_images(page, languages=("tr",)), "Metin okundu")

            # Benign unsupported embedded formats (GIF, JBIG2) are skipped without failing the page.
            benign_only = SimpleNamespace(
                images=[SimpleNamespace(data=gif_bytes), SimpleNamespace(data=jbig2_opaque)]
            )
            self.assertEqual(ocr.read_page_images(benign_only, languages=("tr",)), "")

            mixed_benign = SimpleNamespace(
                images=[SimpleNamespace(data=jbig2_opaque), SimpleNamespace(data=png_bytes)]
            )
            self.assertEqual(ocr.read_page_images(mixed_benign, languages=("tr",)), "Metin okundu")

            # Malicious signatures, oversized streams, and decompression bombs must NEVER be swallowed.
            oversized = b"\x89PNG\r\n\x1a\n" + b"0" * ocr.PILLOW_MAX_BYTES
            for malicious in (self._EPS_ASCII, self._EPS_BINARY, self._PDF_BYTES, oversized, b""):
                bad_page = SimpleNamespace(
                    images=[SimpleNamespace(data=png_bytes), SimpleNamespace(data=malicious)]
                )
                with self.assertRaises(ocr.UnsafeImageError):
                    ocr.read_page_images(bad_page, languages=("tr",))

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", _PIL_IMAGE.DecompressionBombWarning)
                bomb_page = SimpleNamespace(
                    images=[SimpleNamespace(data=png_bytes), SimpleNamespace(data=bomb_20mp)]
                )
                with self.assertRaises(ocr.UnsafeImageError):
                    ocr.read_page_images(bomb_page, languages=("tr",))


if __name__ == "__main__":
    unittest.main()
