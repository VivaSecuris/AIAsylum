"""Authentication routes.

Login exchanges an API key for a signed session cookie. The cookie is checked
on every request by the middleware in ``api/security.py`` whenever
``REQUIRE_AUTH`` is on; this module only issues and clears it.
"""

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response
from pydantic import BaseModel

from config import settings
from vivasecuris.aiasylum.api.security import SESSION_COOKIE, SESSION_TTL_SECONDS, issue_session, verify_key, verify_session

router = APIRouter()


class SessionRequest(BaseModel):
    """Session creation request."""
    api_key: str


@router.get("/session")
async def current_session(request: Request):
    """Report cookie validity after a reload, without exposing credentials.

    ``auth`` is "enabled" exactly when the middleware enforces it (REQUIRE_AUTH).
    Configured keys alone do not lock anything, so they must not make the UI ask
    for a login that changes nothing.
    """
    keys = settings.api_keys_list
    cookie = request.cookies.get(SESSION_COOKIE, "")
    return {
        "authenticated": bool(cookie and verify_session(cookie, keys, settings.api_key_hmac_secret)),
        "auth": "enabled" if settings.require_auth else "disabled",
    }


@router.post("/session")
async def create_session(request: SessionRequest, response: Response):
    """Validate an API key and issue a signed, HttpOnly session cookie.

    The cookie carries an HMAC-signed expiry bound to the key, never the key
    itself. When auth is off (local development), login is a no-op that still
    succeeds: nothing is locked, so there is nothing to unlock.
    """
    keys = settings.api_keys_list
    if not settings.require_auth:
        return {"status": "authenticated", "auth": "disabled"}
    if not verify_key(request.api_key, keys):
        raise HTTPException(status_code=401, detail="Invalid API key")

    response.set_cookie(
        key=SESSION_COOKIE,
        value=issue_session(request.api_key, settings.api_key_hmac_secret),
        httponly=True,
        secure=settings.session_cookie_secure,
        samesite="strict",
        max_age=SESSION_TTL_SECONDS,
    )
    return {"status": "authenticated"}


@router.post("/logout")
async def logout(response: Response):
    """Logout and clear session."""
    response.delete_cookie(key=SESSION_COOKIE)
    return {"status": "logged_out"}
