#!/usr/bin/env python3
"""Zero-dependency Windows DPAPI vault CLI for EvidenceRAG secrets.

Inspects whether EVIDENCERAG_API_KEY, XAI_API_KEY, and GROK_API_KEY in the
process environment or a `.env` file are sealed (`dpapi:v1:`), plaintext, or
unset, and atomically seals plaintext keys in-place inside `.env` with `0o600`
file permissions. Never prints or logs secret values.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Any, Callable

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from app.dpapi_vault import DPAPI_PREFIX, is_dpapi_available, seal_secret

SECRET_ENV_KEYS: tuple[str, ...] = (
    "EVIDENCERAG_API_KEY",
    "XAI_API_KEY",
    "GROK_API_KEY",
)

_ASSIGN_RE = re.compile(r"^(\s*(?:export\s+)?)([A-Za-z_][A-Za-z0-9_]*)(\s*=\s*)(.*)$")
_INLINE_COMMENT_RE = re.compile(r"(\s+#.*)$")


def _split_line_ending(line: str) -> tuple[str, str]:
    if line.endswith("\r\n"):
        return line[:-2], "\r\n"
    if line.endswith("\n"):
        return line[:-1], "\n"
    if line.endswith("\r"):
        return line[:-1], "\r"
    return line, ""


def _parse_rhs_value(rhs: str) -> tuple[str, str]:
    """Parse a `.env` value expression into `(value, trailing_suffix)`."""
    stripped = rhs.strip()
    if not stripped:
        return "", ""
    if stripped[0] in ('"', "'"):
        quote = stripped[0]
        chars: list[str] = []
        idx = 1
        escaped = False
        closed = False
        while idx < len(stripped):
            ch = stripped[idx]
            if quote == '"' and escaped:
                chars.append(ch)
                escaped = False
                idx += 1
                continue
            if quote == '"' and ch == "\\":
                escaped = True
                idx += 1
                continue
            if ch == quote:
                closed = True
                idx += 1
                break
            chars.append(ch)
            idx += 1
        if closed:
            return "".join(chars), stripped[idx:]
        return stripped, ""
    if stripped.startswith("#"):
        return "", rhs
    match = _INLINE_COMMENT_RE.search(rhs)
    if match:
        return rhs[: match.start()].strip(), match.group(1)
    return stripped, ""


def classify_secret_value(value: str | None) -> str:
    """Classify a secret value as `'sealed'`, `'plaintext'`, or `'unset'`."""
    if value is None:
        return "unset"
    raw = str(value).strip()
    if not raw:
        return "unset"
    if raw.startswith(DPAPI_PREFIX):
        return "sealed" if raw[len(DPAPI_PREFIX) :].strip() else "unset"
    return "plaintext"


def parse_env_file_values(env_file: str | Path) -> dict[str, str]:
    """Read key-value assignments from a `.env` file using stdlib only."""
    target = Path(env_file)
    if not target.is_file():
        return {}
    values: dict[str, str] = {}
    for line in target.read_bytes().decode("utf-8-sig").splitlines():
        body, _ = _split_line_ending(line)
        match = _ASSIGN_RE.match(body)
        if not match:
            continue
        _, key, _, rhs = match.groups()
        val, _ = _parse_rhs_value(rhs)
        values[key] = val
    return values


def collect_vault_status(
    env_file: str | Path = ".env",
    *,
    keys: tuple[str, ...] = SECRET_ENV_KEYS,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Inspect secret key protection state across process env and `.env`."""
    target = Path(env_file)
    env_source = os.environ if environ is None else environ
    env_file_exists = target.is_file()
    dotenv_values = parse_env_file_values(target) if env_file_exists else {}

    key_statuses: dict[str, dict[str, str]] = {}
    has_plaintext = False
    for key in keys:
        env_state = classify_secret_value(env_source.get(key))
        dotenv_state = classify_secret_value(dotenv_values.get(key))
        if env_state != "unset":
            effective = env_state
            source = "env"
        elif dotenv_state != "unset":
            effective = dotenv_state
            source = "dotenv"
        else:
            effective = "unset"
            source = "none"

        if env_state == "plaintext" or dotenv_state == "plaintext":
            has_plaintext = True

        key_statuses[key] = {
            "env": env_state,
            "dotenv": dotenv_state,
            "effective": effective,
            "source": source,
        }

    return {
        "ok": True,
        "dpapi_available": bool(is_dpapi_available()),
        "env_file": str(target),
        "env_file_exists": env_file_exists,
        "has_plaintext": has_plaintext,
        "keys": key_statuses,
    }


def seal_env_file(
    env_file: str | Path,
    *,
    keys: tuple[str, ...] = SECRET_ENV_KEYS,
    runner: Callable[[str, str], str] | None = None,
) -> dict[str, Any]:
    """Seal plaintext managed secrets in `.env` atomically with `0o600` permissions."""
    target = Path(env_file)
    if not target.exists():
        raise FileNotFoundError("Target .env file does not exist")
    if not target.is_file():
        raise ValueError("Target .env path must be a regular file")

    raw_bytes = target.read_bytes()
    has_bom = raw_bytes.startswith(b"\xef\xbb\xbf")
    raw_text = raw_bytes.decode("utf-8-sig")
    managed = set(keys)
    sealed_set: set[str] = set()
    already_sealed_set: set[str] = set()
    new_lines: list[str] = []
    changed = False

    for line in raw_text.splitlines(keepends=True):
        body, eol = _split_line_ending(line)
        match = _ASSIGN_RE.match(body)
        if not match:
            new_lines.append(line)
            continue

        prefix, key, eq_part, rhs = match.groups()
        if key not in managed:
            new_lines.append(line)
            continue

        val, suffix = _parse_rhs_value(rhs)
        state = classify_secret_value(val)
        if state == "unset":
            new_lines.append(line)
            continue
        if state == "sealed":
            already_sealed_set.add(key)
            new_lines.append(line)
            continue

        if runner is None and not is_dpapi_available():
            raise RuntimeError(f"Windows DPAPI is unavailable; cannot seal {key}")

        try:
            sealed_val = seal_secret(val, runner=runner)
        except Exception as exc:
            raise RuntimeError(f"DPAPI sealing failed for {key} ({type(exc).__name__})") from None

        if (
            not isinstance(sealed_val, str)
            or not sealed_val.startswith(DPAPI_PREFIX)
            or not sealed_val[len(DPAPI_PREFIX) :].strip()
            or sealed_val == val
            or (len(val) >= 4 and val in sealed_val[len(DPAPI_PREFIX) :])
        ):
            raise RuntimeError(f"DPAPI sealing did not produce a valid sealed value for {key}")

        new_lines.append(f"{prefix}{key}{eq_part}{sealed_val}{suffix}{eol}")
        sealed_set.add(key)
        changed = True

    if changed:
        output_bytes = "".join(new_lines).encode("utf-8-sig" if has_bom else "utf-8")
        fd, tmp_name = tempfile.mkstemp(
            prefix=f".{target.name}.",
            suffix=".tmp",
            dir=str(target.parent),
        )
        tmp_path = Path(tmp_name)
        fd_closed = False
        try:
            os.chmod(tmp_path, 0o600)
            with os.fdopen(fd, "wb") as handle:
                fd_closed = True
                handle.write(output_bytes)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_path, target)
            os.chmod(target, 0o600)
        except Exception:
            if not fd_closed:
                try:
                    os.close(fd)
                except OSError:
                    pass
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass
            raise
    else:
        os.chmod(target, 0o600)

    sealed_keys = [k for k in keys if k in sealed_set]
    already_sealed_keys = [k for k in keys if k in already_sealed_set and k not in sealed_set]
    unset_keys = [k for k in keys if k not in sealed_set and k not in already_sealed_set]

    return {
        "ok": True,
        "env_file": str(target),
        "changed": changed,
        "sealed_keys": sealed_keys,
        "already_sealed_keys": already_sealed_keys,
        "unset_keys": unset_keys,
        "permissions": "0600",
    }


def _format_status_text(report: dict[str, Any]) -> str:
    lines = [
        f"Windows DPAPI available: {report['dpapi_available']}",
        f"Env file: {report['env_file']} ({'present' if report['env_file_exists'] else 'absent'})",
    ]
    for key, info in report["keys"].items():
        lines.append(
            f"  {key}: effective={info['effective']} (source={info['source']}, "
            f"env={info['env']}, dotenv={info['dotenv']})"
        )
    return "\n".join(lines)


def _format_seal_text(result: dict[str, Any]) -> str:
    sealed = ", ".join(result["sealed_keys"]) if result["sealed_keys"] else "none"
    already = ", ".join(result["already_sealed_keys"]) if result["already_sealed_keys"] else "none"
    unset = ", ".join(result["unset_keys"]) if result["unset_keys"] else "none"
    return (
        f"DPAPI seal complete ({result['env_file']}, permissions={result['permissions']}, "
        f"changed={result['changed']})\n"
        f"  sealed: {sealed}\n"
        f"  already_sealed: {already}\n"
        f"  unset: {unset}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect or seal EvidenceRAG secrets with Windows DPAPI (dpapi:v1:)."
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_status = sub.add_parser(
        "status",
        help="Check whether managed secret keys are sealed, plaintext, or unset",
    )
    p_status.add_argument("--env-file", default=".env", help="Path to .env file (default: .env)")
    p_status.add_argument("--json", action="store_true", dest="as_json", help="Emit JSON summary")

    p_seal = sub.add_parser(
        "seal",
        help="Seal plaintext managed secrets in-place inside a .env file",
    )
    p_seal.add_argument("--env-file", default=".env", help="Path to .env file (default: .env)")
    p_seal.add_argument("--json", action="store_true", dest="as_json", help="Emit JSON summary")
    return parser


def main(
    argv: list[str] | None = None,
    *,
    runner: Callable[[str, str], str] | None = None,
    environ: Mapping[str, str] | None = None,
) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.cmd == "status":
        try:
            report = collect_vault_status(args.env_file, environ=environ)
        except OSError as exc:
            print(f"Error: could not read env file ({type(exc).__name__})", file=sys.stderr)
            return 1
        if args.as_json:
            print(json.dumps(report, indent=2, ensure_ascii=False))
        else:
            print(_format_status_text(report))
        return 0

    if args.cmd == "seal":
        try:
            result = seal_env_file(args.env_file, runner=runner)
        except FileNotFoundError:
            print("Error: target .env file not found.", file=sys.stderr)
            return 1
        except (ValueError, RuntimeError) as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        except OSError as exc:
            print(f"Error: failed to update .env file ({type(exc).__name__})", file=sys.stderr)
            return 1
        if args.as_json:
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            print(_format_seal_text(result))
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
