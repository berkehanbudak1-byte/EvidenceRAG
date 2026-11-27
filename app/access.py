"""Access profiles: direct loopback (local) or API-key hosted. Not multi-tenant auth."""
import base64
from collections import deque
import hashlib
import hmac
import ipaddress
import json
from pathlib import Path
import re
import time
from urllib.parse import urlsplit

try:
    from starlette.responses import JSONResponse
except ImportError:  # pragma: no cover - stdlib-only unit test fallback
    JSONResponse = None


LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}
PROXY_HEADERS = {b"forwarded", b"x-forwarded-for", b"x-forwarded-host", b"x-forwarded-proto",
                 b"x-forwarded-port", b"x-real-ip"}
SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}
API_KEY_HEADER = b"x-api-key"
PROTECTED_MUTATION_PREFIXES = ("/ask", "/documents", "/swarm")
DOCS_PATHS = {"/docs", "/redoc"}


class AuthFailureRateLimiter:
    """Sliding-window per-IP rate limiter for failed API-key attempts."""

    def __init__(self, max_failures: int = 15, window_seconds: float = 60.0, max_ips: int = 1000):
        self.max_failures = max(1, int(max_failures))
        self.window_seconds = max(1.0, float(window_seconds))
        self.max_ips = max(10, int(max_ips))
        self._failures: dict[str, deque[float]] = {}

    def _prune(self, ip: str, now: float) -> deque[float]:
        cutoff = now - self.window_seconds
        dq = self._failures.get(ip)
        if dq is None:
            return deque()
        while dq and dq[0] <= cutoff:
            dq.popleft()
        if not dq:
            self._failures.pop(ip, None)
            return deque()
        return dq

    def is_limited(self, ip: str, now: float | None = None) -> bool:
        ts = time.monotonic() if now is None else float(now)
        dq = self._prune(str(ip or "unknown"), ts)
        return len(dq) >= self.max_failures

    def record_failure(self, ip: str, now: float | None = None) -> int:
        ts = time.monotonic() if now is None else float(now)
        key = str(ip or "unknown")
        cutoff = ts - self.window_seconds
        if key not in self._failures and len(self._failures) >= self.max_ips:
            expired = [k for k, v in self._failures.items() if not v or v[-1] <= cutoff]
            for k in expired:
                del self._failures[k]
            while len(self._failures) >= self.max_ips:
                self._failures.pop(next(iter(self._failures)))
        dq = self._failures.setdefault(key, deque())
        while dq and dq[0] <= cutoff:
            dq.popleft()
        dq.append(ts)
        return len(dq)

    def clear(self, ip: str) -> None:
        self._failures.pop(str(ip or "unknown"), None)


def build_strict_csp(html_path: str | Path | None = None) -> str:
    """Build strict CSP with exact SHA-256 digests for any inline scripts (zero unsafe-inline)."""
    hashes: list[str] = []
    if html_path is not None:
        p = Path(html_path)
        if p.is_file():
            text = p.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n")
            for attrs, body in re.findall(r"<script\b([^>]*)>([\s\S]*?)</script\s*>", text, re.I):
                if re.search(r"\bsrc\s*=", attrs or "", re.I):
                    continue
                digest = base64.b64encode(hashlib.sha256(body.encode("utf-8")).digest()).decode("ascii")
                token = f"'sha256-{digest}'"
                if token not in hashes:
                    hashes.append(token)
    script_src = " ".join(["'self'", *hashes])
    return "; ".join([
        "default-src 'self'",
        f"script-src {script_src}",
        "style-src 'self'",
        "img-src 'self' data:",
        "font-src 'self'",
        "connect-src 'self'",
        "object-src 'none'",
        "base-uri 'self'",
        "form-action 'self'",
        "frame-ancestors 'none'",
    ])


_STATIC_INDEX = Path(__file__).resolve().parents[1] / "static" / "index.html"
DEFAULT_STRICT_CSP = build_strict_csp(_STATIC_INDEX)
BASE_SECURITY_HEADERS: tuple[tuple[bytes, bytes], ...] = (
    (b"x-content-type-options", b"nosniff"),
    (b"x-frame-options", b"DENY"),
    (b"referrer-policy", b"no-referrer"),
    (b"permissions-policy", b"camera=(), microphone=(), geolocation=(), payment=(), usb=()"),
)


def normalize_profile(value):
    """Accept exact local/hosted/cloud; cloud aliases to hosted. Case variants are rejected."""
    profile = (value or "local").strip()
    if profile == "cloud":
        return "hosted"
    return profile


def parse_public_hosts(value):
    """Split EVIDENCERAG_PUBLIC_HOST into lowercase host[/port] allowlist entries."""
    if value is None:
        return ()
    if isinstance(value, (list, tuple)):
        raw = value
    else:
        raw = str(value).split(",")
    hosts = []
    for item in raw:
        entry = item.strip().lower()
        if not entry:
            continue
        if "://" in entry or "/" in entry or "@" in entry or any(c.isspace() for c in entry):
            raise ValueError(f"Invalid EVIDENCERAG_PUBLIC_HOST entry: {item!r}")
        hosts.append(entry)
    return tuple(hosts)


def _host_allowed(host, port, allowed_hosts):
    """Return True if host[/port] matches the allowlist (bare host allows any port)."""
    host = host.lower()
    candidates = {host}
    if host == "::1":
        candidates.add("[::1]")
    if port is not None:
        candidates.update(f"{name}:{port}" for name in list(candidates))
    allowed = {entry.lower() for entry in allowed_hosts}
    if candidates & allowed:
        return True
    # Bare allowlist hostname accepts Host with any valid port.
    return bool(candidates & allowed) or host in allowed or (host == "::1" and "[::1]" in allowed)


def authority(value, scheme, allowed_hosts=None):
    """Parse a strict origin authority; reject userinfo, paths and ambiguous ports."""
    local_mode = allowed_hosts is None
    allowed = LOCAL_HOSTS if local_mode else set(allowed_hosts)
    if scheme not in {"http", "https"} or not value or any(c.isspace() for c in value):
        raise ValueError("Invalid authority")
    parsed = urlsplit(f"{scheme}://{value}")
    if parsed.username is not None or parsed.password is not None or parsed.path or parsed.query or parsed.fragment:
        raise ValueError("Invalid authority")
    host = parsed.hostname
    port = parsed.port
    if host is None or (port is not None and not 1 <= port <= 65535):
        raise ValueError("Invalid host/port")
    if local_mode:
        if host not in LOCAL_HOSTS:
            raise ValueError("Invalid local host/port")
        canonical = f"[{host}]" if host == "::1" else host
        if port is not None:
            canonical += f":{port}"
        if value.lower() != canonical:
            raise ValueError("Noncanonical authority")
        return scheme, host, port if port is not None else (443 if scheme == "https" else 80)
    if not _host_allowed(host, port, allowed):
        raise ValueError("Host not in allowlist")
    return scheme, host.lower(), port if port is not None else (443 if scheme == "https" else 80)


def _header_values(headers, name):
    return [value for header_name, value in headers if header_name.lower() == name]


def _path_requires_api_key(method, path):
    if method in SAFE_METHODS:
        return False
    path = path or ""
    return any(path == prefix or path.startswith(prefix + "/") for prefix in PROTECTED_MUTATION_PREFIXES)


def _api_key_from_scope(headers):
    values = _header_values(headers, API_KEY_HEADER)
    if len(values) > 1:
        raise ValueError("Multiple API keys")
    if len(values) == 1:
        return values[0].decode("utf-8")
    auth_values = _header_values(headers, b"authorization")
    if len(auth_values) > 1:
        raise ValueError("Multiple Authorization headers")
    if len(auth_values) == 1:
        raw = auth_values[0].decode("utf-8")
        if raw.lower().startswith("bearer "):
            return raw[7:].strip()
    return None


def access_denial(scope, settings=None, rate_limiter=None):
    headers = scope.get("headers", [])
    profile = normalize_profile(getattr(settings, "access_profile", None) if settings is not None else "local")
    peer = scope.get("client")
    peer_ip = str(peer[0]) if peer else "unknown"
    try:
        local_peer = bool(peer) and ipaddress.ip_address(peer[0]).is_loopback
    except ValueError:
        local_peer = False

    if profile == "local":
        if not local_peer:
            return 403, "Local profile requires a direct loopback peer"
        allowed = None  # authority() local mode
        host_error = "Invalid local Host header"
        origin_error = "Origin must match the local request origin"
        proxy_detail = "Forwarded/proxy headers are unsupported in the local profile"
    elif profile == "hosted":
        allowed = set(getattr(settings, "public_hosts", ()) or ())
        if not allowed:
            return 500, "Hosted profile misconfigured: no public hosts"
        host_error = "Invalid or disallowed Host header"
        origin_error = "Origin must match the hosted request origin"
        proxy_detail = "Forwarded/proxy headers are unsupported"
    else:
        return 500, "Unsupported access profile"

    if any(name.lower() in PROXY_HEADERS for name, _value in headers):
        return 400, proxy_detail

    hosts = _header_values(headers, b"host")
    try:
        if len(hosts) != 1:
            raise ValueError("Exactly one Host required")
        target = authority(hosts[0].decode("ascii"), scope.get("scheme", "http"), allowed_hosts=allowed)
    except (ValueError, UnicodeError):
        return 400, host_error

    method = scope.get("method", "GET")
    path = scope.get("path", "")
    if profile == "hosted" and _path_requires_api_key(method, path):
        if rate_limiter is not None and rate_limiter.is_limited(peer_ip):
            return 429, "Too many failed authentication attempts"
        expected = getattr(settings, "api_key", None) or ""
        try:
            provided = _api_key_from_scope(headers)
        except (ValueError, UnicodeError):
            if rate_limiter is not None:
                rate_limiter.record_failure(peer_ip)
            return 401, "Invalid API key header"
        if not expected or provided is None or not hmac.compare_digest(provided, expected):
            if rate_limiter is not None:
                rate_limiter.record_failure(peer_ip)
            return 401, "API key required"
        if rate_limiter is not None:
            rate_limiter.clear(peer_ip)

    if method not in SAFE_METHODS:
        origins = _header_values(headers, b"origin")
        if origins:
            try:
                if len(origins) != 1:
                    raise ValueError("Multiple origins")
                origin = urlsplit(origins[0].decode("ascii"))
                if origin.path or origin.query or origin.fragment:
                    raise ValueError("Origin is not a bare origin")
                if authority(origin.netloc, origin.scheme, allowed_hosts=allowed) != target:
                    raise ValueError("Origin differs from request origin")
            except (ValueError, UnicodeError):
                return 403, origin_error
        # No Origin is permitted for direct CLI clients. This is not user auth.
    return None


async def _send_json_denial(scope, receive, send, status: int, detail: str, extra_headers=()):
    denial_headers = list(BASE_SECURITY_HEADERS) + list(extra_headers)
    if JSONResponse is not None:
        header_map = {k.decode("ascii"): v.decode("ascii") for k, v in denial_headers}
        return await JSONResponse({"detail": detail}, status_code=status, headers=header_map)(
            scope, receive, send
        )
    body = json.dumps({"detail": detail}).encode("utf-8")
    headers = [
        (b"content-type", b"application/json"),
        (b"content-length", str(len(body)).encode("ascii")),
        *denial_headers,
    ]
    await send({"type": "http.response.start", "status": status, "headers": headers})
    await send({"type": "http.response.body", "body": body})


class AccessMiddleware:
    """ASGI middleware: reject before diagnostics/body parsing and apply strict security headers."""

    def __init__(self, app, settings=None, rate_limiter=None, csp_header: str = DEFAULT_STRICT_CSP):
        self.app = app
        self._settings = settings
        self._rate_limiter = rate_limiter if rate_limiter is not None else AuthFailureRateLimiter()
        self._csp_bytes = csp_header.encode("ascii")

    def _resolve_settings(self):
        if self._settings is None:
            return None
        return self._settings() if callable(self._settings) else self._settings

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            denied = access_denial(scope, self._resolve_settings(), rate_limiter=self._rate_limiter)
            if denied:
                status, detail = denied
                extra_headers = ()
                if status == 429:
                    retry_after = str(max(1, int(self._rate_limiter.window_seconds))).encode("ascii")
                    extra_headers = ((b"retry-after", retry_after),)
                return await _send_json_denial(
                    scope, receive, send, status, detail, extra_headers=extra_headers
                )

            path = scope.get("path", "")
            include_csp = path not in DOCS_PATHS

            async def send_with_security_headers(message):
                if message.get("type") == "http.response.start":
                    headers = list(message.get("headers") or [])
                    existing = {k.lower() for k, _ in headers}
                    for name, value in BASE_SECURITY_HEADERS:
                        if name not in existing:
                            headers.append((name, value))
                    if include_csp and b"content-security-policy" not in existing:
                        headers.append((b"content-security-policy", self._csp_bytes))
                    message = {**message, "headers": headers}
                await send(message)

            return await self.app(scope, receive, send_with_security_headers)
        return await self.app(scope, receive, send)


# Backward-compatible name used by tests and older call sites.
LocalAccessMiddleware = AccessMiddleware

