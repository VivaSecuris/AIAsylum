"""Authentication for every API route, fail-closed when enabled.

Before this existed the API had a login endpoint and nothing that checked it:
no route declared an auth dependency, so a session cookie was issued and then
ignored. That was tolerable on a laptop behind localhost. It is not tolerable
on a rented GPU box, where an open port means anyone who finds it can load
models, write multi-gigabyte edits and run up the bill.

So when ``REQUIRE_AUTH`` is set, one middleware guards every path except the
health probe and the login itself. It is middleware rather than a per-route
dependency on purpose: a new router cannot forget to opt in.

Credentials accepted, in order:

* ``X-API-Key: <key>`` -- for scripts and ``curl``.
* ``session_token`` cookie -- for the browser, including ``EventSource`` and
  the dashboard ``<iframe>``, neither of which can set headers.

The cookie never contains the key. It carries an expiry, a short digest that
identifies *which* configured key it was issued for, and an HMAC over both.
Rotating a key or the HMAC secret therefore invalidates every session issued
against it, and a stolen cookie does not reveal a reusable credential.

Startup refuses to proceed with auth enabled but no keys, or with the default
HMAC secret: an auth layer that can be satisfied by the placeholder value in
``.env.example`` is not an auth layer.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import threading
import time
from collections import defaultdict, deque
from typing import Deque, Dict, Iterable, Optional

from fastapi import Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

SESSION_COOKIE = "session_token"
SESSION_TTL_SECONDS = 12 * 3600
PLACEHOLDER_SECRET = "change-me-in-production"

# Paths reachable without credentials. Everything else -- including /docs and
# /openapi.json, which would otherwise publish the whole attack surface -- is
# guarded.
PUBLIC_PATHS = frozenset({"/health", "/api/v1/auth/session", "/api/v1/auth/logout"})

# Failed attempts per client address before it is throttled. Keys here are
# 32+ bytes of randomness, so this is about noise and log volume, not about
# making brute force feasible-but-slow.
MAX_FAILURES_PER_MINUTE = 10


class AuthConfigError(RuntimeError):
    """Auth is enabled but cannot actually protect anything."""


def _key_digest(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def _sign(secret: str, payload: str) -> str:
    return hmac.new(secret.encode(), payload.encode(), hashlib.sha256).hexdigest()


def issue_session(key: str, secret: str, now: Optional[float] = None) -> str:
    """A cookie value bound to ``key`` that does not contain it."""
    expiry = int((now or time.time()) + SESSION_TTL_SECONDS)
    payload = f"{expiry}.{_key_digest(key)}"
    return f"{payload}.{_sign(secret, payload)}"


def verify_session(token: str, keys: Iterable[str], secret: str, now: Optional[float] = None) -> bool:
    try:
        expiry_s, digest, sig = token.split(".")
        expiry = int(expiry_s)
    except (ValueError, AttributeError):
        return False
    if expiry < (now or time.time()):
        return False
    if not hmac.compare_digest(sig, _sign(secret, f"{expiry_s}.{digest}")):
        return False
    # The key it was issued for must still be configured.
    return any(hmac.compare_digest(digest, _key_digest(k)) for k in keys)


def verify_key(candidate: Optional[str], keys: Iterable[str]) -> bool:
    if not candidate:
        return False
    # compare_digest on every key, no early exit, so timing does not reveal
    # which key or how many characters matched.
    ok = False
    for k in keys:
        ok |= hmac.compare_digest(candidate.encode(), k.encode())
    return ok


def check_config(require_auth: bool, keys: list, secret: str) -> None:
    if not require_auth:
        return
    if not keys:
        raise AuthConfigError("REQUIRE_AUTH is set but API_KEYS is empty; every request would be refused.")
    if any(len(k) < 24 for k in keys):
        raise AuthConfigError("Every entry in API_KEYS must be at least 24 characters.")
    if not secret or secret == PLACEHOLDER_SECRET:
        raise AuthConfigError(
            "REQUIRE_AUTH is set but API_KEY_HMAC_SECRET is unset or the placeholder "
            "from .env.example; session cookies would be forgeable."
        )


class FailureThrottle:
    """Per-address sliding window of failed auth attempts."""

    def __init__(self, limit: int = MAX_FAILURES_PER_MINUTE, window: float = 60.0):
        self.limit = limit
        self.window = window
        self._hits: Dict[str, Deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def _trim(self, q: Deque[float], now: float) -> None:
        while q and now - q[0] > self.window:
            q.popleft()

    def blocked(self, addr: str) -> bool:
        now = time.monotonic()
        with self._lock:
            q = self._hits[addr]
            self._trim(q, now)
            return len(q) >= self.limit

    def record(self, addr: str) -> None:
        now = time.monotonic()
        with self._lock:
            q = self._hits[addr]
            self._trim(q, now)
            q.append(now)


throttle = FailureThrottle()


def _client(request: Request) -> str:
    return request.client.host if request.client else "unknown"


def make_auth_middleware(settings):
    """Build the middleware, reading settings at request time so tests can flip them."""

    async def auth_middleware(request: Request, call_next):
        if not settings.require_auth:
            return await call_next(request)

        # CORS preflight carries no credentials by design; the CORS layer
        # answers it and the real request that follows is checked.
        if request.method == "OPTIONS" or request.url.path in PUBLIC_PATHS:
            return await call_next(request)

        addr = _client(request)
        if throttle.blocked(addr):
            return JSONResponse({"detail": "Too many failed authentication attempts."}, status_code=429)

        keys = settings.api_keys_list
        secret = settings.api_key_hmac_secret
        header_key = request.headers.get("x-api-key")
        cookie = request.cookies.get(SESSION_COOKIE)

        if verify_key(header_key, keys) or (cookie and verify_session(cookie, keys, secret)):
            return await call_next(request)

        throttle.record(addr)
        logger.warning("Rejected unauthenticated %s %s from %s", request.method, request.url.path, addr)
        return JSONResponse(
            {"detail": "Authentication required."},
            status_code=401,
            headers={"WWW-Authenticate": "API-Key"},
        )

    return auth_middleware
