# Validate on RewardBench2

Validate your graders against RewardBench2, a comprehensive benchmark for evaluating response quality across multiple domains. RewardBench2 provides standardized test cases covering factuality, focus, safety, math, instruction following, and specialized domains.

---

## What is RewardBench2?

RewardBench2 is a benchmark dataset designed to evaluate reward models and LLM judges. It tests grader performance across diverse scenarios to measure evaluation quality and identify systematic biases.

**Key Features:**

- **Multi-Domain Coverage** — Factuality, focus, safety, math, precise instruction following
- **Standardized Ground Truth** — Expert-curated correct answers
- **Bias Testing** — Position bias, length bias, and adversarial cases
- **Public Leaderboard** — Compare your grader with state-of-the-art models

### Dataset Overview

| Subset | Samples | Evaluation Mode | Ground Truth |
|--------|---------|-----------------|--------------|
| **Factuality** | 475 | Four-way ranking | Best response among 4 candidates |
| **Focus** | 495 | Four-way ranking | Best response among 4 candidates |
| **Math** | 183 | Four-way ranking | Best response among 4 candidates |
| **Precise IF** | 160 | Four-way ranking | Best response among 4 candidates |
| **Safety** | 450 | Four-way ranking | Best response among 4 candidates |
| **Ties** | 102 | Absolute rating (1-10) | Multiple correct answers (1-26 per sample) |
| **Total** | **1,865** | - | - |

### Evaluation Modes

RewardBench2 uses two complementary evaluation approaches:

**Four-Way Comparison (Default):**
```
Query: "Explain quantum computing"

Candidates:
├─ A: "Quantum computing leverages quantum mechanics..." ← Best
├─ B: "It's a type of advanced computing..."
├─ C: "Computers that use quantum physics..."
└─ D: "I'm not sure about that."

Task: Select the best response (A/B/C/D)
```

- Random shuffling to prevent position bias
- Tests comparative judgment ability
- Binary outcome (correct/incorrect)

**Ties Absolute Rating:**
```
Query: "Write a creative poem about nature"

Candidates with Ground Truth:
├─ A: "The forest whispers..." → 9/10 ✓ Winner
├─ B: "Trees and flowers..." → 9/10 ✓ Winner (tie)
├─ C: "Nature is nice..." → 5/10
└─ D: "Roses are red..." → 6/10

Task: Rate each response (1-10), pick highest-rated
```

- Allows multiple correct answers (ties)
- Tests absolute quality assessment
- More nuanced than binary ranking

---

## Quick Start

```bash
# Install dependencies
pip install rm-gallery datasets pandas pyarrow

# Download dataset
python -c "from datasets import load_dataset; load_dataset('allenai/reward-bench-2', split='test').to_parquet('rewardbench2_test.parquet')"

# Run validation
cd tutorials/cookbooks/grader_validation
python rewardbench2.py --data-path rewardbench2_test.parquet --model qwen3-32b
```

**Reference:**
- Complete implementation: `tutorials/cookbooks/grader_validation/rewardbench2.py`
- Dataset: `allenai/reward-bench-2` on HuggingFace
- Key classes: `RewardBench2Grader`, `RewardBench2Analyzer`, `load_rewardbench2_data()`


---

## Interpreting Results

### Overall Accuracy

The primary metric is overall accuracy across all subsets:

```
Overall Accuracy: 78.5%
Correct: 785/1000
```

**Interpretation:**
- **> 80%** — Excellent: Grader performs well across domains
- **70-80%** — Good: Reliable for most use cases
- **60-70%** — Fair: May need refinement for production use
- **< 60%** — Poor: Requires significant improvement

### Per-Subset Breakdown

```
Per-Subset Performance:
  Factuality      : 82.3% (391/475)
  Focus           : 78.8% (390/495)
  Math            : 65.0% (119/183)
  Precise IF      : 71.9% (115/160)
  Safety          : 88.4% (398/450)
  Ties            : 76.5% ( 78/102)
```

**Analysis:**
- **Strengths**: High Safety (88.4%) and Factuality (82.3%) accuracy
- **Weaknesses**: Lower Math (65.0%) suggests difficulty with mathematical reasoning
- **Action**: Review failed Math cases, consider adding domain-specific examples to prompt

### Common Error Patterns

```python
# Analyze errors by category
errors_by_subset = {}
for sample, result in zip(validation_data, results):
    if result.score < 1.0:  # Incorrect
        subset = sample["subset"]
        if subset not in errors_by_subset:
            errors_by_subset[subset] = []
        errors_by_subset[subset].append({
            "query": sample["query"],
            "predicted": result.metadata.get("predicted_letter"),
            "correct": result.metadata.get("correct_letter"),
            "reason": result.reason
        })

# Review errors
for subset, errors in errors_by_subset.items():
    print(f"\n{subset} Errors ({len(errors)}):")
    for error in errors[:3]:  # Show first 3
        print(f"  Query: {error['query'][:80]}...")
        print(f"  Predicted: {error['predicted']}, Correct: {error['correct']}")
```

---

## Tips for Success

!!! tip "Improve Prompts"
    **Four-Way Comparison:**
    - Add explicit anti-bias instructions (ignore length/position)
    - Focus on helpfulness, accuracy, clarity, completeness
    - Use structured output format: `[[A]]`, `[[B]]`, `[[C]]`, or `[[D]]`
    
    **Ties Rating:**
    - Provide calibrated scale (1-3: Poor, 4-5: Below avg, 6-7: Good, 8-9: Excellent, 10: Outstanding)
    - Request reasoning before rating
    - Output numeric rating on last line

!!! tip "Check Position Bias"
    Test if grader favors certain positions (should be ~25% each for A/B/C/D). Add anti-bias instructions if biased.

!!! tip "Advanced Usage"
    - **Custom Subsets** — Add domain-specific samples with same format and merge with RewardBench2 data
    - **Compare Graders** — Run multiple graders on same dataset to find best performer
    - **Error Analysis** — Review failed cases per subset to identify systematic issues

---

## Troubleshooting

| Issue | Solutions |
|-------|-----------|
| **Low Ties Accuracy** | Use structured output for reliable numeric parsing |
| **Position Bias** | Add anti-bias instructions; verify shuffling works |
| **Parsing Failures** | Multi-strategy parser (check [[X]], "Answer X", standalone letter) |
| **Inconsistent Results** | Lower temperature, structured output, few-shot examples |
| **Low Overall Accuracy** | Review failed cases, refine prompts, try stronger model |

**Debug:** Analyze by subset → Test small sample with logging → Review responses → Check position distribution (~25% each)

See [Overview - Troubleshooting](overview.md#troubleshooting) for more techniques.

---

## Next Steps

**Improve Your Grader:**
- [Validation Overview](overview.md) — Learn validation concepts and best practices
- [Create Custom Graders](../building_graders/create_custom_graders.md) — Refine your grader implementation
- [Train Reward Models](../building_graders/training/overview.md) — Train models on RewardBench2 data

**Deploy Validated Graders:**
- [Running Graders](../running_graders/run_tasks.md) — Set up production evaluation pipelines
- [Monitor Performance](../running_graders/evaluation_reports.md) — Track grader accuracy over time

