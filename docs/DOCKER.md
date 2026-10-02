# Docker Deployment Guide

## Quick Start

### Development

```bash
# Build and start all services
docker compose up -d

# View logs
docker compose logs -f

# Stop services
docker compose down
```

### Production

```bash
# Set POSTGRES_PASSWORD in .env first. See .env.example.
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

## Services

### API Service
- Port: 8000
- Health check: `http://localhost:8000/health`
- API docs: `http://localhost:8000/docs`
- Listens on all container interfaces. Compose sets `REQUIRE_AUTH=true` and will not start until `API_KEYS` and `API_KEY_HMAC_SECRET` are set. See [API_KEYS.md](API_KEYS.md).

### Database (PostgreSQL)
- Not published to the host. Other containers reach it as `db:5432`.
- User: `aiasylum`
- Password: the `POSTGRES_PASSWORD` value from `.env` (required; there is no default)
- Database: `aiasylum`

### Ollama (Optional)
- Port: 11434
- Start with: `docker compose --profile ollama up -d`

### Frontend (Optional)
- Port: 3000
- Start with: `docker compose --profile frontend up -d`

## Environment Variables

Create a `.env` file in the project root (or let `./install.sh` create one):

```bash
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
GOOGLE_API_KEY=
OLLAMA_BASE_URL=http://ollama:11434
POSTGRES_PASSWORD=choose-a-long-password
API_KEY_HMAC_SECRET=generate-with-secrets.token_urlsafe
API_KEYS=generate-with-secrets.token_urlsafe
REQUIRE_AUTH=true
CORS_ORIGINS=http://localhost:3000,http://localhost:8000
```

## Volumes

- `postgres_data`: PostgreSQL data
- `api_data`: API data directory inside the container
- `ollama_data`: Ollama model storage (only with `--profile ollama`)

## Commands

```bash
# Build images
docker compose build

# Start services
docker compose up -d

# Stop services
docker compose down

# View logs
docker compose logs -f

# Remove containers and the database volume
docker compose down -v
```

## Troubleshooting

### Database Connection Issues
- Ensure database service is healthy: `docker compose ps`
- Check database logs: `docker compose logs db`
- Verify DATABASE_URL in .env

### API Not Starting
- Check API logs: `docker compose logs api`
- Verify migrations ran: `docker compose exec api alembic current`
- Check environment variables

### Port Conflicts
- Change ports in `docker-compose.yml` if 8000 or 3000 are in use. PostgreSQL is not published to the host.
- Update CORS_ORIGINS if changing frontend port

## Production Considerations

1. **Use secrets management**: Don't hardcode API keys
2. **Enable HTTPS**: Use reverse proxy (Nginx/Traefik) with SSL
3. **Set resource limits**: Add memory/CPU limits in docker-compose
4. **Use production database**: Consider managed PostgreSQL service
5. **Enable logging**: Configure log aggregation
6. **Use production images**: Build optimized images for production

## Building Custom Images

```bash
# Build API image
docker build -t ai-asylum-api .

# Build frontend image
docker build -t ai-asylum-frontend ./frontend
```
