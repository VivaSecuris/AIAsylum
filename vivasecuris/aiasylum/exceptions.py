"""Custom exceptions for AI Asylum."""


class AIAsylumError(Exception):
    """Base exception for AI Asylum."""
    pass


class ModelProviderError(AIAsylumError):
    """Error with model provider (API key missing, connection failed, etc.)."""
    pass


class TestExecutionError(AIAsylumError):
    """Error during test execution."""
    pass


class ConfigurationError(AIAsylumError):
    """Configuration error (missing settings, invalid values, etc.)."""
    pass


class DatabaseError(AIAsylumError):
    """Database operation error."""
    pass


class AnalysisError(AIAsylumError):
    """Error during analysis."""
    pass
