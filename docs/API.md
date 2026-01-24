# API Documentation

## Authentication

### Transparent Login Flow

1. **POST /api/v1/auth/session**
   - Request body: `{"api_key": "your-api-key"}`
   - Response: Sets HttpOnly cookie `session_token`
   - The API key is not stored in browser localStorage

2. **Subsequent Requests**
   - Include the session cookie automatically
   - No need to send API key in headers

3. **POST /api/v1/auth/logout**
   - Clears the session cookie

### High-Security API Keys

For production deployments, use DB-backed API keys:

- Format: `ak_live_<kid>_<secret>`
- `kid`: Key ID (database lookup)
- `secret`: HMAC-verified secret
- Verification uses `API_KEY_HMAC_SECRET` from environment

## Endpoints

### Test Runs

#### Create Test Run
```
POST /api/v1/test-runs/
```

Request:
```json
{
  "doctor_provider": "ollama",
  "doctor_model": "llama3.2",
  "patient_provider": "ollama",
  "patient_model": "llama2",
  "test_type": "conversation",
  "test_config": {}
}
```

Response:
```json
{
  "id": 1,
  "doctor_provider": "ollama",
  "doctor_model": "llama3.2",
  "patient_provider": "ollama",
  "patient_model": "llama2",
  "test_type": "conversation",
  "status": "running"
}
```

#### List Test Runs
```
GET /api/v1/test-runs/?limit=100&offset=0&test_type=conversation
```

#### Get Test Run
```
GET /api/v1/test-runs/{test_run_id}
```

#### Get Test Results
```
GET /api/v1/test-runs/{test_run_id}/results
```

#### Get Conversation
```
GET /api/v1/test-runs/{test_run_id}/conversation
```

### Analysis

#### Run Analysis
```
POST /api/v1/analysis/test-run/{test_run_id}
```

Request:
```json
{
  "enable_activation_patching": false,
  "enable_cot_detection": true,
  "cot_analysis_mode": "full"
}
```

#### Get Assessments
```
GET /api/v1/analysis/test-run/{test_run_id}/assessments
```

### Benchmarks

#### List Benchmarks
```
GET /api/v1/benchmarks/list
```

#### Run Benchmark
```
POST /api/v1/benchmarks/run
```

Request:
```json
{
  "provider": "ollama",
  "model": "llama3.2",
  "benchmark": "mmlu",
  "num_samples": 100
}
```

## Interactive API Docs

Visit `http://localhost:8000/docs` for interactive Swagger documentation.
