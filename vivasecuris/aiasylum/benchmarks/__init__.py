"""Benchmark integration module."""

from vivasecuris.aiasylum.benchmarks.base import Benchmark, BenchmarkResult
from vivasecuris.aiasylum.benchmarks.simple import SimpleBenchmark, create_simple_benchmark
from vivasecuris.aiasylum.benchmarks.datasets import load_benchmark_dataset, BENCHMARK_DATASETS

__all__ = [
    "Benchmark",
    "BenchmarkResult",
    "SimpleBenchmark",
    "create_simple_benchmark",
    "load_benchmark_dataset",
    "BENCHMARK_DATASETS",
]
