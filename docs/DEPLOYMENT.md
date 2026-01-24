# Deployment Guide

## Production Deployment

### Prerequisites

- Python 3.10+
- PostgreSQL (recommended) or SQLite (development)
- Node.js 18+ (for frontend)
- API keys for at least one LLM provider

### Environment Setup

1. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```

2. Configure environment variables:
   ```bash
   # Required: At least one API key
   OPENAI_API_KEY=sk-...
   # OR
   ANTHROPIC_API_KEY=sk-ant-...
   # OR
   GOOGLE_API_KEY=...
   
   # Database (PostgreSQL recommended)
   DATABASE_URL=postgresql://user:password@localhost:5432/aiasylum
   
   # Security (CHANGE IN PRODUCTION)
   API_SECRET_KEY=your-secret-key-here
   JWT_SECRET_KEY=your-jwt-secret-here
   API_KEY_HMAC_SECRET=your-hmac-secret-here
   
   # CORS (adjust for your domain)
   CORS_ORIGINS=https://yourdomain.com,https://api.yourdomain.com
   ```

### Database Setup

1. Initialize database:
   ```bash
   alembic upgrade head
   ```

2. For PostgreSQL with TimescaleDB + pgvector (optional, for RAG):
   ```sql
   CREATE EXTENSION IF NOT EXISTS timescaledb;
   CREATE EXTENSION IF NOT EXISTS vector;
   ```

### Running the Application

#### Option 1: Direct Execution

```bash
# Install dependencies
make install

# Start API
make run-api

# In another terminal, start frontend
cd frontend && npm install && npm run dev
```

#### Option 2: Docker Compose

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://user:password@db:5432/aiasylum
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    depends_on:
      - db
  
  db:
    image: postgres:15
    environment:
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=password
      - POSTGRES_DB=aiasylum
    volumes:
      - postgres_data:/var/lib/postgresql/data
  
  frontend:
    build: ./frontend
    ports:
      - "3000:3000"
    depends_on:
      - api

volumes:
  postgres_data:
```

Run:
```bash
docker-compose up -d
```

### Security Considerations

1. **API Keys**: Never commit API keys to version control
2. **Secrets**: Use environment variables or secret management services
3. **HTTPS**: Always use HTTPS in production
4. **CORS**: Restrict CORS origins to your domain
5. **Rate Limiting**: Configure appropriate rate limits
6. **Authentication**: Enable authentication for production deployments

### High-Security Authentication

For production, implement DB-backed API keys with HMAC verification:

1. Store API keys in database with scoped permissions
2. Use token format: `ak_live_<kid>_<secret>`
3. Verify HMAC signature using `API_KEY_HMAC_SECRET`
4. Implement token rotation and expiration

See `docs/API.md` for API authentication details.

### Monitoring

- Set up logging aggregation (e.g., ELK stack, Datadog)
- Monitor API response times and error rates
- Track database performance
- Set up alerts for failed test runs

### Scaling

- Use a reverse proxy (nginx, Traefik) for load balancing
- Consider using a task queue (Celery, RQ) for long-running tests
- Use connection pooling for database connections
- Cache frequently accessed data (Redis, Memcached)
