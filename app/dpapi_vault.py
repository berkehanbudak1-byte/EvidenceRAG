"""Windows DPAPI (CurrentUser) secret vault helper for EvidenceRAG.

Allows sealing EVIDENCERAG_API_KEY and XAI_API_KEY with `dpapi:v1:<base64>` on Windows
so local deployments avoid storing plaintext secrets in `.env`.
"""

from __future__ import annotations

import base64
import os
import subprocess
import sys
from typing import Callable

DPAPI_PREFIX = "dpapi:v1:"


def is_dpapi_available() -> bool:
    return sys.platform == "win32" and os.getenv("EVIDENCERAG_TEST_MODE") != "1"


def _run_powershell_dpapi(action: str, b64_input: str) -> str:
    op = (
        "[System.Security.Cryptography.ProtectedData]::Protect($raw,$null,"
        "[System.Security.Cryptography.DataProtectionScope]::CurrentUser)"
        if action == "protect"
        else "[System.Security.Cryptography.ProtectedData]::Unprotect($raw,$null,"
        "[System.Security.Cryptography.DataProtectionScope]::CurrentUser)"
    )
    script = (
        "$ErrorActionPreference='Stop';"
        "Add-Type -AssemblyName System.Security;"
        "$raw=[Convert]::FromBase64String($env:ERAG_DPAPI_B64);"
        f"$out={op};"
        "[Console]::Out.Write([Convert]::ToBase64String($out))"
    )
    encoded = base64.b64encode(script.encode("utf-16le")).decode("ascii")
    env = dict(os.environ)
    env["ERAG_DPAPI_B64"] = str(b64_input or "")
    out = subprocess.check_output(
        ["powershell.exe", "-NoProfile", "-NonInteractive", "-EncodedCommand", encoded],
        env=env,
        timeout=5,
        text=True,
    )
    return out.strip()


def seal_secret(plaintext: str, runner: Callable[[str, str], str] | None = None) -> str:
    raw = str(plaintext or "")
    if not raw or raw.startswith(DPAPI_PREFIX):
        return raw
    if runner is None and not is_dpapi_available():
        return raw
    b64_in = base64.b64encode(raw.encode("utf-8")).decode("ascii")
    active_runner = runner or _run_powershell_dpapi
    sealed_b64 = active_runner("protect", b64_in)
    return f"{DPAPI_PREFIX}{sealed_b64}" if sealed_b64 else raw


def unseal_secret(stored: str | None, runner: Callable[[str, str], str] | None = None) -> str | None:
    if stored is None:
        return None
    raw = str(stored).strip()
    if not raw or not raw.startswith(DPAPI_PREFIX):
        return raw
    if runner is None and sys.platform != "win32":
        return ""
    payload = raw[len(DPAPI_PREFIX):].strip()
    active_runner = runner or _run_powershell_dpapi
    try:
        b64_out = active_runner("unprotect", payload)
        return base64.b64decode(b64_out).decode("utf-8") if b64_out else ""
    except Exception:
        return ""
