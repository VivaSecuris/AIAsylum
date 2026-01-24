#!/usr/bin/env python3
"""Test script to verify benchmark dataset downloading works."""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from vivasecuris.aiasylum.benchmarks.datasets import load_benchmark_dataset, BENCHMARK_DATASETS


async def test_benchmark_download(benchmark_name: str, num_samples: int = 5):
    """Test downloading a benchmark dataset."""
    print(f"\n{'='*60}")
    print(f"Testing benchmark: {benchmark_name}")
    print(f"{'='*60}\n")
    
    if benchmark_name not in BENCHMARK_DATASETS:
        print(f"ERROR: Unknown benchmark '{benchmark_name}'")
        print(f"Available benchmarks: {list(BENCHMARK_DATASETS.keys())}")
        return False
    
    config = BENCHMARK_DATASETS[benchmark_name]
    print(f"Dataset: {config['dataset']}")
    print(f"Split: {config.get('split', 'test')}")
    print(f"Config: {config.get('config', 'None')}")
    print(f"Question field: {config['question_field']}")
    print(f"Answer field: {config.get('answer_field', 'None')}")
    print()
    
    try:
        print(f"Downloading {num_samples} samples...")
        dataset = await load_benchmark_dataset(benchmark_name, num_samples=num_samples)
        
        if not dataset:
            print(f"ERROR: No dataset returned")
            return False
        
        print(f"\n✓ Successfully loaded {len(dataset)} samples\n")
        
        # Show first sample
        if len(dataset) > 0:
            sample = dataset[0]
            print("First sample:")
            print(f"  Question: {sample.get('question', '')[:200]}...")
            print(f"  Answer: {sample.get('answer', 'None')}")
            if sample.get('choices'):
                print(f"  Choices: {sample.get('choices')}")
            print()
        
        return True
        
    except Exception as e:
        import traceback
        print(f"\n✗ ERROR loading dataset:")
        print(f"  {e}")
        print(f"\nTraceback:")
        print(traceback.format_exc())
        return False


async def main():
    """Test all benchmarks or a specific one."""
    if len(sys.argv) > 1:
        benchmark_name = sys.argv[1].lower()
        num_samples = int(sys.argv[2]) if len(sys.argv) > 2 else 5
        success = await test_benchmark_download(benchmark_name, num_samples)
        sys.exit(0 if success else 1)
    else:
        # Test a few key benchmarks
        test_benchmarks = ["gsm8k", "mmlu", "arc", "hellaswag"]
        results = {}
        
        for benchmark in test_benchmarks:
            results[benchmark] = await test_benchmark_download(benchmark, num_samples=3)
        
        print(f"\n{'='*60}")
        print("Summary:")
        print(f"{'='*60}")
        for benchmark, success in results.items():
            status = "✓ PASS" if success else "✗ FAIL"
            print(f"  {benchmark}: {status}")
        
        all_passed = all(results.values())
        sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    asyncio.run(main())
