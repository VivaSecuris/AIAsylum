# Benchmark Web UI Flow - Automatic Dataset Download

## Complete Flow: Web UI → Dataset Download → Benchmark Execution

### 1. User Action in Web UI
- User goes to `/benchmarks` page
- Selects benchmark (e.g., "mmlu")
- Selects provider and model
- Sets number of samples
- Clicks "Run Benchmark"

### 2. Frontend → API Call
```
Web UI (benchmarks/index.tsx)
  ↓
useRunBenchmark() hook
  ↓
apiClient.runBenchmark({ provider, model, benchmark, num_samples })
  ↓
POST /api/v1/benchmarks/run
```

### 3. API Route Processing
**File**: `vivasecuris/aiasylum/api/routes/benchmarks.py`

- Creates `TestRun` with:
  - `test_type = "benchmark"`
  - `meta_data.benchmark = "mmlu"`
  - `meta_data.num_samples = 100`
  - `test_config.benchmark_name = "mmlu"`
- Returns immediately with test run ID
- **Starts background task** to run benchmark

### 4. Background Task Execution
**File**: `vivasecuris/aiasylum/runner/runner.py`

```
Background Task (_run_benchmark_background)
  ↓
runner.execute_test_run(test_run_id)
  ↓
Detects test_type == "benchmark"
  ↓
Creates BenchmarkTest instance
  ↓
Calls test.run(patient_model)
```

### 5. Dataset Download (Automatic)
**File**: `vivasecuris/aiasylum/tests/benchmark.py`

```python
# In BenchmarkTest.run()
dataset = await load_benchmark_dataset(self.benchmark_name, self.num_samples)
```

**File**: `vivasecuris/aiasylum/benchmarks/datasets.py`

```python
# In load_benchmark_dataset()
from datasets import load_dataset

# This automatically downloads from HuggingFace if not cached
dataset = load_dataset(dataset_name, split=split)
```

**Key Points:**
- ✅ **Automatic Download**: `load_dataset()` downloads from HuggingFace Hub automatically
- ✅ **Automatic Caching**: HuggingFace caches datasets locally (usually `~/.cache/huggingface/datasets/`)
- ✅ **No Manual Steps**: User doesn't need to download anything manually
- ✅ **First Run**: Downloads on first use (may take a few minutes for large datasets)
- ✅ **Subsequent Runs**: Uses cached version (instant)

### 6. Benchmark Execution
After dataset is loaded:
1. Formats questions (adds choices for multiple choice)
2. Runs each question through the model
3. Evaluates responses against ground truth
4. Calculates accuracy
5. Stores results in database

### 7. Results Available
- Test run status updates to "completed"
- Results visible in web UI at `/test-runs/{id}`
- Accuracy scores and per-question results stored

## Dataset Download Details

### How HuggingFace Datasets Works

1. **First Download**:
   - Checks local cache (`~/.cache/huggingface/datasets/`)
   - If not found, downloads from HuggingFace Hub
   - Shows progress (if running in terminal)
   - Caches locally for future use

2. **Cached Datasets**:
   - Subsequent runs use cached version
   - No re-download needed
   - Much faster execution

3. **Dataset Sizes** (approximate):
   - **MMLU**: ~2GB (test split)
   - **GSM8K**: ~1MB (small)
   - **HellaSwag**: ~50MB
   - **ARC**: ~10MB
   - **MATH**: ~100MB
   - **TruthfulQA**: ~5MB

### Download Behavior

- **Automatic**: No user intervention needed
- **Background**: Happens during test execution
- **Progress**: Visible in server logs
- **Error Handling**: Falls back to simple questions if download fails
- **Caching**: Persistent across restarts

## Verification

To verify datasets are downloading, check server logs:

```
[load_benchmark_dataset] Starting download/load of cais/mmlu (split: test, config: None)
[load_benchmark_dataset] Attempting to load cais/mmlu with split=test
[load_benchmark_dataset] Successfully loaded cais/mmlu, got 1319 samples
[load_benchmark_dataset] Limited to 100 samples
[load_benchmark_dataset] Successfully standardized 100 samples from mmlu
[BenchmarkTest] Loaded 100 samples for benchmark mmlu
```

## Troubleshooting

### If Download Fails

1. **Check Internet Connection**: HuggingFace requires internet access
2. **Check Disk Space**: Large datasets need space (MMLU ~2GB)
3. **Check Permissions**: Cache directory must be writable
4. **Check Logs**: Look for error messages in server logs

### Fallback Behavior

If dataset download fails, the system:
1. Logs the error
2. Falls back to simple predefined questions
3. Still runs the benchmark (with limited questions)
4. Logs warning about fallback

## Summary

✅ **Yes, it automatically downloads datasets** when triggered from the web UI:
- No manual download needed
- Happens in background during test execution
- Cached for future use
- Works for all supported benchmarks (MMLU, GSM8K, HellaSwag, ARC, etc.)

The entire flow is automatic and transparent to the user. They just click "Run Benchmark" and the system handles everything.
