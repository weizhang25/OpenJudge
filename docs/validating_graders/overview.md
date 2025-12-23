# Validating Graders

Validate your graders against benchmark datasets or custom test sets to ensure accurate, reliable evaluation results before production deployment.

---

## Why Validate?

Graders are evaluation systems that need validation themselves. Without validation, you risk deploying unreliable evaluators.

**Key Benefits:**

- **Measure Accuracy** — Quantify agreement with ground truth
- **Compare Approaches** — Benchmark different grader implementations
- **Build Confidence** — Ensure quality thresholds before production
- **Debug Issues** — Identify systematic errors or biases
- **Track Improvements** — Monitor performance over time

---

## How to Validate a Grader?

Validation compares your grader's judgments against known ground truth to measure evaluation quality.

**Validation Flow:**
1. Grader evaluates validation dataset (queries + candidate responses)
2. Compare predictions with ground truth labels
3. Compute accuracy metrics (overall and per-category)
4. Generate validation report

### Approach 1: Benchmark Validation

Validate against public benchmarks with standardized ground truth.

**When to Use:**
- ✅ Want to compare with published baselines
- ✅ Need reproducible validation results
- ✅ Evaluating general-purpose graders
- ✅ Quick validation without data collection

**Available Benchmarks:**
- **[RewardBench2](rewardbench2.md)** — Multi-domain response quality evaluation
- **MT-Bench** — Multi-turn conversation quality (coming soon)
- **AlpacaEval** — Instruction-following evaluation (coming soon)

**Example:**

```python
from rm_gallery.core.graders.llm_grader import LLMGrader
from rm_gallery.core.models import OpenAIChatModel
from rm_gallery.core.runner import GradingRunner

# Create grader
model = OpenAIChatModel(model="qwen3-32b")
grader = LLMGrader(name="my_grader", model=model)

# Run on benchmark
runner = GradingRunner(grader_configs={"my_grader": grader})
results = await runner.arun(rewardbench2_dataset)

# Analyze accuracy
from rm_gallery.core.analyzer import ValidationAnalyzer
analyzer = ValidationAnalyzer()
report = analyzer.analyze(dataset=rewardbench2_dataset, grader_results=results["my_grader"])

print(f"Accuracy: {report.metadata['accuracy']:.2%}")
```

### Approach 2: Custom Validation

Build validation pipelines tailored to your domain and evaluation criteria.

**When to Use:**
- ✅ Domain-specific evaluation (legal, medical, finance)
- ✅ Proprietary test sets with internal standards
- ✅ Non-standard evaluation tasks
- ✅ Need control over validation methodology

**Validation Metrics:**

Choose metrics based on your evaluation task:

=== "Ranking Metrics"

    For graders that rank or select best responses:

    | Metric | When to Use | Interpretation |
    |--------|-------------|----------------|
    | **Accuracy** | Binary classification (correct/incorrect) | % of times grader selects correct answer |
    | **Top-K Accuracy** | Multiple acceptable answers | % of times correct answer is in top K predictions |
    | **Mean Reciprocal Rank (MRR)** | Ranking quality | Average reciprocal rank of correct answer |
    | **Kendall's Tau** | Full ranking correlation | Agreement between predicted and true rankings |

=== "Scoring Metrics"

    For graders that output continuous scores:

    | Metric | When to Use | Interpretation |
    |--------|-------------|----------------|
    | **Pearson Correlation** | Linear relationship | How well scores correlate with ground truth |
    | **Spearman Correlation** | Ranking correlation | Agreement in relative ordering |
    | **Mean Absolute Error (MAE)** | Score accuracy | Average distance from ground truth scores |
    | **F1 Score** | Binary threshold (pass/fail) | Balance between precision and recall |

**See cookbook for complete examples:** `tutorials/cookbooks/grader_validation/`

---

## Tips for Success

!!! tip "Data Quality"
    - **Diverse Test Cases** — Include edge cases, ambiguous queries, domain-specific content
    - **Sufficient Size** — Aim for 100+ samples for reliable accuracy estimates
    - **Balanced Categories** — Ensure each category has enough samples (20+ minimum)
    - **Clear Ground Truth** — Use high-agreement human annotations or verified labels
    - **Hold-Out Sets** — Never validate on training data for trained models

!!! tip "Validation Design"
    - **Match Production** — Validation should mirror real-world use cases
    - **Control Bias** — Randomize answer positions to prevent position bias
    - **Multiple Runs** — Run validation multiple times with different random seeds
    - **Error Analysis** — Review failed cases to identify systematic issues
    - **Threshold Tuning** — Adjust score thresholds based on validation results

!!! tip "Result Interpretation"
    - **Context Matters** — 70% accuracy may be excellent or poor depending on task difficulty
    - **Compare Baselines** — Validate against random/majority baselines and existing graders
    - **Per-Category Analysis** — Overall accuracy may hide category-specific weaknesses
    - **Statistical Significance** — Use confidence intervals for small validation sets
    - **Human Agreement** — Compare grader accuracy to inter-annotator agreement

!!! tip "Advanced Techniques"
    - **Cross-Validation** — Test robustness across multiple data splits (k-fold)
    - **Adversarial Testing** — Test with challenging cases (similar responses, negation, position bias)
    - **Confidence Calibration** — Verify if grader confidence correlates with actual accuracy

---

## Troubleshooting

| Issue | Symptoms | Solutions |
|-------|----------|-----------|
| **Position Bias** | Favors first/last responses | Randomize order, add anti-bias instructions |
| **Length Bias** | Prefers longer/shorter | Normalize by length, adjust prompt |
| **Prompt Mismatch** | Good on some categories only | Refine per category or use specialized graders |
| **Inconsistency** | Varies on same input | Lower temperature, use structured output |
| **Overfitting** | High training, low validation | More data, simplify model, regularize |

**Debug Process:** Review failed cases → Identify patterns → Test fixes → Re-validate

---

## Next Steps

**Validate with Benchmarks:**
- [RewardBench2 Validation](rewardbench2.md) — Validate on multi-domain response quality benchmark

**Build Custom Validation:**
- [Create Custom Graders](../building_graders/create_custom_graders.md) — Build graders to validate
- [Running Graders](../running_graders/run_tasks.md) — Set up batch evaluation pipelines
- [Evaluation Reports](../running_graders/evaluation_reports.md) — Generate detailed validation reports

**Improve Your Graders:**
- [Train Reward Models](../building_graders/training/overview.md) — Train models on your validation data

