"""The API must be unusable by anyone without a key once REQUIRE_AUTH is on.

These exist because the API used to issue a login cookie that no route ever
checked. On a rented GPU box that meant an open port was an open GPU.
"""

import time

import pytest

from vivasecuris.aiasylum.api import security

KEY = "k" * 40
OTHER = "o" * 40
SECRET = "s" * 40


@pytest.fixture
def locked(monkeypatch):
    from config import settings

    monkeypatch.setattr(settings, "require_auth", True)
    monkeypatch.setattr(settings, "api_keys", KEY)
    monkeypatch.setattr(settings, "api_key_hmac_secret", SECRET)
    security.throttle._hits.clear()
    yield settings
    security.throttle._hits.clear()


@pytest.fixture
def client():
    from fastapi.testclient import TestClient

    from vivasecuris.aiasylum.api.main import app

    return TestClient(app)


GUARDED = [
    ("get", "/api/v1/weights/stages"),
    ("get", "/api/v1/weights/runs"),
    ("post", "/api/v1/weights/runs"),
    ("post", "/api/v1/weights/models/x/chat"),
    ("get", "/api/v1/interp/modes"),
    ("get", "/api/v1/test-runs/"),
    ("get", "/api/v1/models/providers"),
    ("get", "/docs"),
    ("get", "/openapi.json"),
]


@pytest.mark.parametrize("method,path", GUARDED)
def test_every_route_refuses_without_credentials(locked, client, method, path):
    r = getattr(client, method)(path)
    assert r.status_code == 401, f"{method.upper()} {path} was reachable without a key"


def test_health_stays_public(locked, client):
    assert client.get("/health").status_code == 200


def test_header_key_is_accepted(locked, client):
    r = client.get("/api/v1/weights/stages", headers={"X-API-Key": KEY})
    assert r.status_code == 200


def test_wrong_key_is_refused(locked, client):
    r = client.get("/api/v1/weights/stages", headers={"X-API-Key": OTHER})
    assert r.status_code == 401


def test_login_rejects_a_bad_key(locked, client):
    """It used to accept anything when API_KEYS was empty, and the cookie was
    never checked afterwards anyway."""
    assert client.post("/api/v1/auth/session", json={"api_key": OTHER}).status_code == 401


def test_login_cookie_grants_access_and_does_not_contain_the_key(locked, client):
    r = client.post("/api/v1/auth/session", json={"api_key": KEY})
    assert r.status_code == 200
    cookie = client.cookies.get(security.SESSION_COOKIE)
    assert cookie and KEY not in cookie, "the session cookie must not carry the key"
    assert client.get("/api/v1/weights/stages").status_code == 200


def test_forged_and_expired_sessions_are_refused():
    good = security.issue_session(KEY, SECRET)
    assert security.verify_session(good, [KEY], SECRET)

    expiry, digest, sig = good.split(".")
    assert not security.verify_session(f"{int(expiry) + 999}.{digest}.{sig}", [KEY], SECRET)
    assert not security.verify_session(good, [KEY], "a-different-secret" * 3)
    assert not security.verify_session(good, [OTHER], SECRET), "rotating the key must end the session"

    old = security.issue_session(KEY, SECRET, now=time.time() - security.SESSION_TTL_SECONDS - 5)
    assert not security.verify_session(old, [KEY], SECRET)

    for junk in ("", "a.b", "x.y.z", "1.2.3.4"):
        assert not security.verify_session(junk, [KEY], SECRET)


def test_repeated_failures_are_throttled(locked, client):
    for _ in range(security.MAX_FAILURES_PER_MINUTE):
        client.get("/api/v1/weights/stages", headers={"X-API-Key": OTHER})
    r = client.get("/api/v1/weights/stages", headers={"X-API-Key": KEY})
    assert r.status_code == 429, "a throttled address is refused even with the right key"


@pytest.mark.parametrize("keys,secret", [
    ([], SECRET),                                   # nothing to check against
    (["short"], SECRET),                            # guessable key
    ([KEY], security.PLACEHOLDER_SECRET),           # forgeable cookies
    ([KEY], ""),
])
def test_auth_refuses_to_start_misconfigured(keys, secret):
    with pytest.raises(security.AuthConfigError):
        security.check_config(True, keys, secret)


def test_auth_off_leaves_local_dev_unchanged(client):
    from config import settings

    assert settings.require_auth is False
    assert client.get("/api/v1/weights/stages").status_code == 200


def test_session_status_reports_cookie_validity_without_disclosing_it(monkeypatch):
    from fastapi.testclient import TestClient
    from config import settings
    from vivasecuris.aiasylum.api.main import app
    from vivasecuris.aiasylum.api.security import SESSION_COOKIE, issue_session
    monkeypatch.setattr(settings, 'require_auth', True)
    monkeypatch.setattr(settings, 'api_keys', 'status-test-key')
    monkeypatch.setattr(settings, 'api_key_hmac_secret', 'test-hmac-secret-for-session-status')
    client = TestClient(app)
    assert client.get('/api/v1/auth/session').json() == {'authenticated': False, 'auth': 'enabled'}
    cookie = issue_session('status-test-key', settings.api_key_hmac_secret)
    client.cookies.set(SESSION_COOKIE, cookie)
    response = client.get('/api/v1/auth/session')
    assert response.json() == {'authenticated': True, 'auth': 'enabled'}
    assert cookie not in response.text and 'status-test-key' not in response.text
    client.cookies.set(SESSION_COOKIE, 'invalid')
    assert client.get('/api/v1/auth/session').json()['authenticated'] is False


def test_configured_keys_alone_do_not_report_auth_enabled(monkeypatch, client):
    """Keys without REQUIRE_AUTH lock nothing, so the UI must not ask for a login."""
    from config import settings
    monkeypatch.setattr(settings, "require_auth", False)
    monkeypatch.setattr(settings, "api_keys", KEY)
    assert client.get("/api/v1/auth/session").json() == {"authenticated": False, "auth": "disabled"}
    assert client.get("/api/v1/test-runs/").status_code == 200
