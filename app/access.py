"""Access profiles: direct loopback (local) or API-key hosted. Not multi-tenant auth."""
import hmac
import ipaddress
from urllib.parse import urlsplit

from starlette.responses import JSONResponse


LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}
PROXY_HEADERS = {b"forwarded", b"x-forwarded-for", b"x-forwarded-host", b"x-forwarded-proto",
                 b"x-forwarded-port", b"x-real-ip"}
SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}
API_KEY_HEADER = b"x-api-key"
PROTECTED_MUTATION_PREFIXES = ("/ask", "/documents")


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


def access_denial(scope, settings=None):
    headers = scope.get("headers", [])
    profile = normalize_profile(getattr(settings, "access_profile", None) if settings is not None else "local")
    peer = scope.get("client")
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
        expected = getattr(settings, "api_key", None) or ""
        try:
            provided = _api_key_from_scope(headers)
        except (ValueError, UnicodeError):
            return 401, "Invalid API key header"
        if not expected or provided is None or not hmac.compare_digest(provided, expected):
            return 401, "API key required"

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


class AccessMiddleware:
    """ASGI middleware: reject before diagnostics/body parsing."""

    def __init__(self, app, settings=None):
        self.app = app
        self._settings = settings

    def _resolve_settings(self):
        if self._settings is None:
            return None
        return self._settings() if callable(self._settings) else self._settings

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            denied = access_denial(scope, self._resolve_settings())
            if denied:
                status, detail = denied
                return await JSONResponse({"detail": detail}, status_code=status)(scope, receive, send)
        return await self.app(scope, receive, send)


# Backward-compatible name used by tests and older call sites.
LocalAccessMiddleware = AccessMiddleware
