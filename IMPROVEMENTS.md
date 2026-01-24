# Code Improvements Implementation Plan

This document outlines specific improvements to make the codebase follow all OOP and Python best practices.

## Quick Wins (Easy Fixes)

### 1. Add Missing Type Hints

**Files to fix:**
- `vivasecuris/aiasylum/tests/base.py` - TestCase.run() parameters
- `vivasecuris/aiasylum/runner/runner.py` - Various methods
- `vivasecuris/aiasylum/analysis/analyzer.py` - Method parameters

### 2. Make ModelProvider Abstract

**File:** `vivasecuris/aiasylum/models/providers.py`

**Change:**
```python
from abc import ABC, abstractmethod

class ModelProvider(ABC):  # Add ABC
    @abstractmethod
    def create_model(...) -> BaseModel:
        """Create a model instance."""
        pass
```

### 3. Add Custom Exceptions

**New file:** `vivasecuris/aiasylum/exceptions.py`

```python
"""Custom exceptions for AI Asylum."""

class AIAsylumError(Exception):
    """Base exception for AI Asylum."""
    pass

class ModelProviderError(AIAsylumError):
    """Error with model provider."""
    pass

class TestExecutionError(AIAsylumError):
    """Error during test execution."""
    pass

class ConfigurationError(AIAsylumError):
    """Configuration error."""
    pass
```

### 4. Add __str__ Methods

**Files:** All model classes, database models

```python
def __str__(self) -> str:
    return f"{self.model_name} ({self.provider})"
```

### 5. Improve Type Hints

**Replace:**
- `Dict` → `Dict[str, Any]`
- `List` → `List[str]` (or appropriate type)
- Add return types to all async generators

## Medium Effort Improvements

### 6. Add Input Validation

**Use Pydantic validators or manual validation:**

```python
from pydantic import validator

class Settings(BaseSettings):
    @validator('temperature')
    def validate_temperature(cls, v):
        if not 0 <= v <= 2:
            raise ValueError('temperature must be between 0 and 2')
        return v
```

### 7. Extract Constants

**New file:** `vivasecuris/aiasylum/constants.py`

```python
"""Constants used throughout the application."""

# Test types
TEST_TYPE_CONVERSATION = "conversation"
TEST_TYPE_SCENARIO = "scenario"
TEST_TYPE_ADVERSARIAL = "adversarial"

# Status values
STATUS_PENDING = "pending"
STATUS_RUNNING = "running"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"

# Harmful keywords
HARMFUL_KEYWORDS = ["bomb", "hack", "hurt", "kill", "violence"]
```

### 8. Improve Resource Management

**Ensure all HTTP clients use context managers:**

```python
async def generate(...):
    async with httpx.AsyncClient() as client:
        response = await client.post(...)
        return process_response(response)
```

### 9. Add Protocol Interfaces

**New file:** `vivasecuris/aiasylum/models/protocols.py`

```python
from typing import Protocol, AsyncIterator
from vivasecuris.aiasylum.models.base import ModelResponse

class ModelProtocol(Protocol):
    """Protocol for model implementations."""
    
    async def generate(...) -> ModelResponse: ...
    async def stream_generate(...) -> AsyncIterator[str]: ...
```

## Larger Refactorings

### 10. Split TestRunner (SRP)

**Current:** TestRunner does everything

**Proposed:**
```python
class ModelFactory:
    """Creates model instances."""
    pass

class TestExecutor:
    """Executes tests."""
    pass

class TestRepository:
    """Handles database operations."""
    pass

class TestRunner:
    """Orchestrates test execution (facade)."""
    def __init__(self, factory, executor, repository):
        self.factory = factory
        self.executor = executor
        self.repository = repository
```

### 11. Dependency Injection Container

**Use a simple DI container or factory pattern:**

```python
class Container:
    """Simple dependency injection container."""
    def __init__(self):
        self._services = {}
    
    def register(self, name: str, factory):
        self._services[name] = factory
    
    def get(self, name: str):
        return self._services[name]()
```

### 12. Add Comprehensive Logging

**Replace print statements with structured logging:**

```python
import structlog

logger = structlog.get_logger()

logger.info("test_started", test_id=test_run.id, test_type=test_type)
logger.error("test_failed", error=str(e), test_id=test_run.id)
```

## Implementation Priority

1. **Now**: Type hints, abstract classes, custom exceptions
2. **Soon**: Input validation, constants extraction, __str__ methods
3. **Later**: SRP refactoring, DI container, comprehensive logging

## Testing After Improvements

After making improvements:
1. Run `mypy` for type checking
2. Run `pylint` or `ruff` for code quality
3. Run all tests to ensure nothing breaks
4. Update documentation
