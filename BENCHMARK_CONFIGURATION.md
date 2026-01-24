# Benchmark Configuration Guide

This document describes the configuration for all supported benchmarks and how they're loaded from HuggingFace.

## Supported Benchmarks

### 1. MMLU (Massive Multitask Language Understanding)
- **Dataset**: `cais/mmlu`
- **Split**: `test`
- **Fields**:
  - `question`: The question text
  - `choices`: List of 4 answer choices
  - `answer`: Class label (0-3, converted to A-D)
  - `subject`: Subject category (e.g., "abstract_algebra")
- **Format**: Multiple choice (4 options)
- **Evaluation**: Checks if model selects correct letter (A-D) or contains correct answer text

### 2. TruthfulQA
- **Dataset**: `truthfulqa/truthful_qa`
- **Config**: `generation` (uses config name, not split)
- **Split**: `validation`
- **Fields**:
  - `question`: The question
  - `best_answer`: The best correct answer
  - `correct_answers`: List of all correct answers
- **Format**: Open-ended questions
- **Evaluation**: Checks if response contains correct answer text

### 3. HellaSwag
- **Dataset**: `Rowan/hellaswag`
- **Split**: `validation`
- **Fields**:
  - `ctx`: Context sentence
  - `endings`: List of 4 possible sentence endings
  - `label`: Class label (0-3, converted to A-D)
- **Format**: Multiple choice (4 options)
- **Evaluation**: Checks if model selects correct ending

### 4. ARC (AI2 Reasoning Challenge)
- **Dataset**: `allenai/ai2_arc`
- **Config**: `ARC-Challenge` (uses config name)
- **Split**: `test`
- **Fields**:
  - `question`: The question
  - `choices`: List of answer choices
  - `answerKey`: Correct answer key (letter)
- **Format**: Multiple choice
- **Evaluation**: Checks if model selects correct answer

### 5. MATH
- **Dataset**: `lighteval/MATH`
- **Split**: `test`
- **Fields**:
  - `problem`: Math problem statement
  - `solution`: Solution to the problem
- **Format**: Open-ended math problems
- **Evaluation**: Checks if response contains solution elements

### 6. GSM8K (Grade School Math 8K)
- **Dataset**: `gsm8k`
- **Split**: `test`
- **Fields**:
  - `question`: Math word problem
  - `answer`: Final numerical answer
- **Format**: Open-ended math problems
- **Evaluation**: Checks if response contains the answer number

### 7. WinoGrande
- **Dataset**: `winogrande`
- **Split**: `validation`
- **Fields**:
  - `sentence`: Sentence with blank
  - `option1`, `option2`: Two answer options
  - `answer`: Correct option (1 or 2)
- **Format**: Binary choice
- **Evaluation**: Checks if model selects correct option

### 8. PIQA (Physical Interaction QA)
- **Dataset**: `piqa`
- **Split**: `validation`
- **Fields**:
  - `goal`: Goal description
  - `sol1`, `sol2`: Two solution options
  - `label`: Correct solution (0 or 1)
- **Format**: Binary choice
- **Evaluation**: Checks if model selects correct solution

### 9. BBQ (Bias Benchmark for QA)
- **Dataset**: `bbq`
- **Split**: `test`
- **Fields**:
  - `context`: Context paragraph
  - `answer`: Correct answer
- **Format**: Open-ended questions
- **Evaluation**: Checks if response contains correct answer

### 10. RealToxicityPrompts
- **Dataset**: `allenai/real-toxicity-prompts`
- **Split**: `train`
- **Fields**:
  - `prompt`: Text prompt
  - No correct answer (toxicity evaluation benchmark)
- **Format**: Prompt completion
- **Evaluation**: Measures toxicity, not correctness

## Answer Format Handling

The benchmark loader handles different answer formats:

1. **Class Labels (0-3)**: Converted to letters (A-D) for multiple choice
   - Used by: MMLU, HellaSwag
   - Example: `answer: 1` → `answer_letter: "B"`, `answer_index: 1`

2. **Letter Answers (A-D)**: Used directly
   - Used by: ARC
   - Example: `answerKey: "B"`

3. **Text Answers**: Used as-is
   - Used by: TruthfulQA, GSM8K, MATH, BBQ
   - Example: `best_answer: "The answer is..."`

4. **Numeric Answers**: Used as-is
   - Used by: WinoGrande (1 or 2), PIQA (0 or 1)

## Evaluation Logic

The evaluation function (`_evaluate_response`) checks:
1. **Letter matching**: If answer is A-D, checks if letter appears in response
2. **Index matching**: If answer is 0-3, checks if index or letter appears
3. **Text matching**: If answer is text, checks if it appears in response (case-insensitive)
4. **Partial matching**: For text answers, checks if key words appear

## Dataset Loading

All datasets are loaded from HuggingFace using the `datasets` library:
- First download is automatic (cached locally)
- Subsequent loads use cached version
- Handles different dataset structures (configs, splits, field names)
- Falls back to simple questions if loading fails

## Testing

Use the test script to verify downloads:
```bash
python scripts/test_benchmark_download.py mmlu 5
python scripts/test_benchmark_download.py gsm8k 10
```

This will show:
- Dataset being downloaded
- Sample questions and answers
- Format verification
