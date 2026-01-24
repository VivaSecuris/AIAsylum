# RAG (Retrieval-Augmented Generation)

AI Asylum supports optional RAG functionality using TimescaleDB + pgvector and Ollama embeddings.

## Setup

1. Install TimescaleDB and pgvector extensions:
   ```sql
   CREATE EXTENSION IF NOT EXISTS timescaledb;
   CREATE EXTENSION IF NOT EXISTS vector;
   ```

2. Configure in `.env`:
   ```bash
   ENABLE_RAG=true
   OLLAMA_EMBEDDING_MODEL=nomic-embed-text
   ```

3. Pull embedding model:
   ```bash
   ollama pull nomic-embed-text
   ```

## Features

- Vector search for similar test cases
- Time-series analysis of test results
- Semantic search across conversations
- Pattern detection in model responses

## Usage

RAG features are automatically enabled when:
- `ENABLE_RAG=true` in `.env`
- Database is PostgreSQL with TimescaleDB + pgvector
- Ollama embedding model is available

## Implementation Status

RAG support is currently a placeholder. Full implementation will include:

- Vector embeddings for test cases
- Similarity search
- Time-series aggregation
- Pattern detection
