"""Deployment and repository hygiene checks for LICENSE, Dockerfile, and render.yaml."""

from __future__ import annotations

from pathlib import Path
import re

import pytest
import yaml

pytestmark = pytest.mark.model_free

REPO_ROOT = Path(__file__).resolve().parents[1]

SECRET_PATTERNS = (
    re.compile(r"\bxai-[A-Za-z0-9_-]{16,}\b"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"),
    re.compile(r"\bghp_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"(?i)\b(?:XAI_API_KEY|GROK_API_KEY|EVIDENCERAG_API_KEY)\s*=\s*['\"]?[^\s'\"]+"),
)


def _assert_no_hardcoded_secrets(text: str, source_name: str) -> None:
    for pattern in SECRET_PATTERNS:
        match = pattern.search(text)
        assert match is None, f"Possible hardcoded secret matched in {source_name}: {pattern.pattern}"


def test_license_file_exists_and_is_mit() -> None:
    license_path = REPO_ROOT / "LICENSE"
    assert license_path.is_file(), "Root LICENSE file must exist"
    content = license_path.read_text(encoding="utf-8")
    assert "MIT License" in content
    assert "Copyright (c) 2026 Berke Han Budak" in content
    assert "Permission is hereby granted, free of charge" in content


def test_dockerfile_hardened_non_root_and_healthcheck() -> None:
    dockerfile_path = REPO_ROOT / "Dockerfile"
    assert dockerfile_path.is_file(), "Root Dockerfile must exist"
    content = dockerfile_path.read_text(encoding="utf-8")
    _assert_no_hardcoded_secrets(content, "Dockerfile")

    lines = [line.strip() for line in content.splitlines() if line.strip() and not line.strip().startswith("#")]

    assert any(line.startswith("FROM python:3.12-slim") for line in lines), (
        "Dockerfile must use python:3.12-slim base image"
    )
    assert "--no-cache-dir" in content, "Dockerfile must install Python dependencies with --no-cache-dir"

    user_directives = [line for line in lines if re.match(r"^USER\s+", line, re.IGNORECASE)]
    assert user_directives, "Dockerfile must specify a non-root USER directive"
    final_user = user_directives[-1].split()[1].split(":")[0].lower()
    assert final_user not in {"root", "0"}, f"Final Dockerfile USER must be non-root, got {final_user!r}"
    assert final_user == "appuser"

    assert any(re.match(r"^EXPOSE\s+8000\b", line, re.IGNORECASE) for line in lines), (
        "Dockerfile must expose port 8000"
    )

    healthcheck_lines = [line for line in lines if line.upper().startswith("HEALTHCHECK")]
    assert healthcheck_lines, "Dockerfile must define a HEALTHCHECK"
    assert "http://127.0.0.1:${PORT:-8000}/health" in content

    assert "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}" in content, (
        "Dockerfile CMD must launch uvicorn binding 0.0.0.0:${PORT:-8000}"
    )


def test_render_yaml_docker_deploy_and_no_hardcoded_secrets() -> None:
    render_path = REPO_ROOT / "render.yaml"
    assert render_path.is_file(), "Root render.yaml must exist"
    raw = render_path.read_text(encoding="utf-8")
    _assert_no_hardcoded_secrets(raw, "render.yaml")

    config = yaml.safe_load(raw)
    assert isinstance(config, dict)
    services = config.get("services")
    assert isinstance(services, list) and len(services) >= 1

    service = services[0]
    assert service.get("type") == "web"
    assert service.get("runtime") == "docker"
    assert service.get("plan") == "free"
    assert service.get("healthCheckPath") == "/health"

    env_vars = {entry["key"]: entry for entry in service.get("envVars", [])}
    assert env_vars.get("ACCESS_PROFILE", {}).get("value") == "hosted"
    assert env_vars.get("LLM_PROVIDER", {}).get("value") in {"xai", "local"}

    for secret_key in ("EVIDENCERAG_API_KEY", "XAI_API_KEY"):
        assert secret_key in env_vars, f"render.yaml must declare {secret_key}"
        assert env_vars[secret_key].get("sync") is False, f"{secret_key} must use sync: false"
        assert "value" not in env_vars[secret_key], f"{secret_key} must not hardcode a value"
