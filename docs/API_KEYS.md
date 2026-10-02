# API keys

AI Asylum uses two kinds of key. They are not interchangeable.

| Key | What it is | Where it goes |
|-----|------------|---------------|
| Provider key | A key you create at OpenAI, Anthropic, or Google so AI Asylum can call that model | `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, or `GOOGLE_API_KEY` in `.env` |
| Your own API key | A key you invent so other people cannot call the AI Asylum API | `API_KEYS` in `.env` |

Ollama does not need a provider key. A local `./start.sh` install does not need your own API key either: `REQUIRE_AUTH` is off and the API listens only on `127.0.0.1`.

`./install.sh` creates `.env` from `.env.example` when one is missing. It generates `API_SECRET_KEY`, `JWT_SECRET_KEY`, and `API_KEY_HMAC_SECRET`. It leaves the provider keys and `API_KEYS` empty, and it does not overwrite an existing `.env`.

## Get a provider key

Create a key on the provider's site, then copy it. You only need one provider if you are not using Ollama.

- **OpenAI.** [platform.openai.com/api-keys](https://platform.openai.com/api-keys). Create a secret key. It starts with `sk-`.
- **Anthropic.** [console.anthropic.com/settings/keys](https://console.anthropic.com/settings/keys). Create a key. It starts with `sk-ant-`.
- **Google.** [aistudio.google.com/apikey](https://aistudio.google.com/apikey). Create an API key in Google AI Studio.

Treat the value like a password. Do not commit `.env`, paste it into a ticket, or put it in a screenshot.

## Set a provider key

Edit `.env` in the repository root:

```bash
OPENAI_API_KEY=sk-your-key
ANTHROPIC_API_KEY=sk-ant-your-key
GOOGLE_API_KEY=your-key
```

Leave unused lines blank. Restart `./start.sh` after editing `.env` by hand.

Or, with the API already running, open [http://127.0.0.1:3000/settings](http://127.0.0.1:3000/settings). Under **API Keys**, paste a key and save. The server writes it to `.env` and uses it immediately. The page never shows a key that is already saved; it only shows whether one is configured. To remove a key, delete its value in `.env` and restart.

## Set your own API key

This key is not issued by OpenAI, Anthropic, or Google. You choose it. When `REQUIRE_AUTH=true`, every API route except `/health` and login requires it.

Generate a key of at least 24 characters:

```bash
python3 -c 'import secrets; print(secrets.token_urlsafe(32))'
```

Put that value in `.env`:

```bash
API_KEYS=the-value-you-just-generated
REQUIRE_AUTH=true
```

Several keys are allowed, separated by commas, with no spaces. Each one must be at least 24 characters. `API_KEY_HMAC_SECRET` must be set to something other than `change-me-in-production` or the API will refuse to start. `./install.sh` already replaces that placeholder when it creates `.env`. If you copied `.env.example` yourself, generate a secret the same way and set `API_KEY_HMAC_SECRET` to it.

Restart the API after changing these lines.

Use the key from the web UI by pasting it into the **API key** field in the header and signing in. The browser receives an HttpOnly session cookie. The cookie is not the key.

Use it from a script with a header:

```bash
curl -H "X-API-Key: the-value-you-just-generated" http://127.0.0.1:8000/api/v1/models
```

Turn this on before binding the API to anything other than `127.0.0.1`. `HOST=0.0.0.0 ./start.sh` publishes the process on every interface. `scripts/remote_session.sh` turns `REQUIRE_AUTH` on for a remote machine and generates the key and HMAC secret there.

## Docker

`docker compose up` refuses to start until `.env` contains `POSTGRES_PASSWORD`, `API_KEY_HMAC_SECRET`, and `API_KEYS`. `REQUIRE_AUTH` defaults to true in Compose. Generate the two secrets as above, and set a provider key if you are not using the optional Ollama profile.

```bash
POSTGRES_PASSWORD=choose-a-long-password
API_KEY_HMAC_SECRET=the-hmac-secret-you-generated
API_KEYS=the-api-key-you-generated
REQUIRE_AUTH=true
OPENAI_API_KEY=sk-your-key
```

Behind HTTPS, also set `SESSION_COOKIE_SECURE=true`. Leave it false when the browser talks to the container over plain HTTP. See [DOCKER.md](DOCKER.md).
