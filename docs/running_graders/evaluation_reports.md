# Generate Evaluation Reports

After successfully [running grading tasks](run_tasks.md) on your dataset, the next crucial step is to validate your graders to ensure they're performing accurately and reliably. Evaluation reports help you assess how well your graders evaluate the quality of AI model outputs.

## What is Grader Validation?

Grader validation is the systematic process of evaluating the effectiveness and reliability of your graders. While running graders on your data tells you what scores they produce, validation tells you whether those scores are meaningful and accurate.

Validation is essential because it helps you:

- **Ensure accuracy** by confirming that your graders correctly identify high-quality vs. low-quality responses
- **Measure reliability** by verifying that graders produce consistent results when evaluating the same input
- **Identify biases** by detecting any systematic biases in how graders evaluate different types of responses
- **Optimize performance** by understanding where graders excel or fall short, guiding improvements

Without proper validation, you risk making decisions based on misleading evaluation results. Imagine using a grader that consistently rates poor responses as excellent - your model development efforts would be misdirected.

!!! tip "Think of Validation Like Calibration"
    Think of grader validation like calibrating a measuring instrument. Just as you wouldn't trust a ruler that measures incorrectly, you shouldn't trust graders without validation.

---

## How to Do Grader Validation

Grader validation approaches can be broadly categorized into two types based on data availability: **statistical analysis without ground truth** and **validation with ground truth**. Each approach offers unique insights into your grader's performance characteristics.

### Statistical Analysis Without Ground Truth

In many cases, you won't have reference labels to compare against. In these scenarios, statistical analysis helps you understand your grader's behavior patterns. This approach focuses on examining the intrinsic properties of the scores produced by your graders, revealing patterns that might indicate issues or strengths in their design.

!!! info "Validation Without Labels"
    Even without knowing the "correct" answers, you can gain valuable insights by observing how scores are distributed across your dataset and how consistently your graders behave when presented with the same inputs multiple times.

#### Distribution Analysis

Examine how your grader distributes scores across your dataset:

```python
from rm_gallery.core.analyzer.statistical.distribution_analyzer import DistributionAnalyzer
from rm_gallery.core.runner.grading_runner import GradingRunner

# After running your graders on a dataset (as described in run_tasks.md)
runner = GradingRunner(grader_configs=grader_configs)
results = await runner.arun(dataset)

# Analyze score distribution to understand grader behavior
analyzer = DistributionAnalyzer()
report = analyzer.analyze(dataset, results["grader_name"])

print(f"Mean score: {report.mean}")
print(f"Standard deviation: {report.stdev}")
print(f"Score range: {report.min_score} to {report.max_score}")
```

**Interpreting Distribution Results:**

- **Narrow clustering**: If all scores cluster closely together, the grader may not be sensitive enough to distinguish between different quality levels
- **Even spread**: If scores are spread too evenly, it might indicate that the grader isn't effectively differentiating quality levels
- **Meaningful range**: A well-designed grader should produce scores that span a meaningful range

#### Consistency Analysis

For LLM-based graders, consistency is crucial for reliable evaluations:

```python
from rm_gallery.core.analyzer.validation.consistency_analyzer import ConsistencyAnalyzer

# Check how reliably a grader produces the same score for identical inputs
analyzer = ConsistencyAnalyzer()
consistency_report = analyzer.analyze(dataset, results["grader_name"])
print(f"Consistency score: {consistency_report.consistency}")
```

!!! warning "Importance of Consistency"
    High consistency scores indicate stable performance, which is especially important for LLM-based graders that can sometimes produce variable results. Inconsistent scoring makes it difficult to trust evaluation outcomes and can lead to confusion when trying to improve models based on these assessments.

---

### Validation With Ground Truth

When you have reference labels, you can perform more rigorous validation by comparing grader results against known standards. This approach provides direct measurements of your grader's accuracy and enables calculation of standard performance metrics like precision, recall, and F1 scores.

!!! note "Power of Ground Truth"
    Ground truth validation is particularly powerful because it gives you concrete measures of how well your graders align with human judgment or other authoritative sources of quality assessment.

#### Accuracy Analysis

For classification tasks, start with basic accuracy measurement:

```python
from rm_gallery.core.analyzer.validation.accuracy_analyzer import AccuracyAnalyzer

# Dataset with ground truth labels for comparison
analyzer = AccuracyAnalyzer()
accuracy_report = analyzer.analyze(
    dataset=dataset,
    grader_results=results["your_grader_name"],
    label_path="correct_label"  # Path to ground truth in your data
)

print(f"Overall accuracy: {accuracy_report.accuracy}")
```

!!! info "Understanding Accuracy"
    This tells you the percentage of times your grader made correct predictions, providing a baseline performance measure. While accuracy alone doesn't tell the whole story, it serves as a foundational metric that helps you understand the general reliability of your grader.

#### Advanced Metrics with F1 Scoring

For a more nuanced understanding, especially with imbalanced datasets, use F1 scores:

```python
from rm_gallery.core.analyzer.validation.f1_score_analyzer import F1ScoreAnalyzer

# F1 Score balances precision and recall for comprehensive evaluation
analyzer = F1ScoreAnalyzer(prediction_threshold=0.5)
f1_report = analyzer.analyze(
    dataset=dataset,
    grader_results=results["grader_name"],
    label_path="label_set"
)

print(f"F1 Score: {f1_report.f1_score}")
print(f"Precision: {f1_report.precision}")
print(f"Recall: {f1_report.recall}")
```

!!! example "When to Use F1 Score"
    These metrics are particularly valuable when different types of errors have different costs in your application. For example, in safety-critical applications, false negatives (failing to identify harmful content) might be much more costly than false positives (incorrectly flagging safe content).

#### Error Analysis

Investigate specific types of mistakes your graders make to guide improvements:

```python
from rm_gallery.core.analyzer.validation.false_positive_analyzer import FalsePositiveAnalyzer
from rm_gallery.core.analyzer.validation.false_negative_analyzer import FalseNegativeAnalyzer

# Analyze false positive rate
# False positives: Grader says "good" but actually "bad"
fp_analyzer = FalsePositiveAnalyzer()
fp_report = fp_analyzer.analyze(dataset, results["grader_name"], label_path="label")

# Analyze false negative rate
# False negatives: Grader says "bad" but actually "good"
fn_analyzer = FalseNegativeAnalyzer()
fn_report = fn_analyzer.analyze(dataset, results["grader_name"], label_path="label")

print(f"False positive rate: {fp_report.false_positive_rate}")
print(f"False negative rate: {fn_report.false_negative_rate}")
```

**Understanding Error Types:**

| Error Type | Definition | Example |
|------------|------------|---------|
| False Positive | Grader says "good" but actually "bad" | Flagging safe content as harmful |
| False Negative | Grader says "bad" but actually "good" | Missing harmful content |

---

## Making Validation Actionable

Effective validation leads to concrete improvements in your evaluation setup. When reviewing validation results, you should consider:

- Whether scores are meaningfully distributed across the expected range
- If accuracy is sufficient for your specific use case
- Which types of errors are most problematic
- Whether the grader is producing consistent results

These insights enable you to refine your graders, adjust parameters, or develop new evaluation approaches. The key is to translate validation findings into specific actions that will improve your system's performance.

!!! example "Actionable Insights"
    - **Narrow score distribution** → Adjust your scoring rubric to better differentiate quality levels
    - **Poor consistency** → Implement repetition mechanisms or refine your prompting strategy
    - **High false negative rate** → Lower your threshold or adjust grader sensitivity

!!! warning "Beyond the Numbers"
    Don't just focus on numerical metrics - consider what the results mean for your actual application. A grader with high accuracy might still be unsuitable if it misses critical cases in your domain.

Regular validation should be part of your standard workflow, not a one-time activity. As your models evolve and your requirements change, continuous validation ensures that your evaluation systems remain effective and reliable.

---

## Next Steps

After validating your graders, you can confidently:

- [Refine data quality](../applications/data_refinement.md) using your validated evaluation insights
- [Create custom graders](../building_graders/create_custom_graders.md) to address any shortcomings identified during validation
- [Train reward models](../building_graders/training/bradley_terry.md) with confidence in your evaluation metrics

