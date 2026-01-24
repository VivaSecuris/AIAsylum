"""Authentication routes."""

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import Response
from pydantic import BaseModel

from config import settings

router = APIRouter()


class SessionRequest(BaseModel):
    """Session creation request."""
    api_key: str


@router.post("/session")
async def create_session(request: SessionRequest, response: Response):
    """
    Create a session with transparent login.
    Issues an HttpOnly cookie for authentication.
    """
    # Validate API key
    valid_keys = settings.api_keys_list
    if valid_keys and request.api_key not in valid_keys:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    # For high-security deployments, verify HMAC signature
    # if settings.api_key_hmac_secret:
    #     # Verify token format: ak_live_<kid>_<secret>
    #     # Verify HMAC signature
    #     pass
    
    # Set HttpOnly cookie
    response.set_cookie(
        key="session_token",
        value=request.api_key,  # In production, use a signed token
        httponly=True,
        secure=True,  # HTTPS only in production
        samesite="lax",
        max_age=86400,  # 24 hours
    )
    
    return {"status": "authenticated"}


@router.post("/logout")
async def logout(response: Response):
    """Logout and clear session."""
    response.delete_cookie(key="session_token")
    return {"status": "logged_out"}
