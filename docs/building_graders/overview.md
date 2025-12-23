# Building Custom Graders

Extend RM-Gallery beyond built-in evaluators by creating custom graders or training reward models. Build domain-specific evaluation logic that seamlessly integrates with RM-Gallery's evaluation pipeline.

---

## Why Build Custom Graders?

While RM-Gallery provides 50+ pre-built graders, custom graders unlock specialized evaluation capabilities:

!!! tip "Benefits of Custom Graders"
    - **Domain Specialization** — Evaluate industry-specific criteria (legal, medical, financial)
    - **Business Requirements** — Implement proprietary scoring logic and evaluation rules
    - **Data-Driven Evaluation** — Train models that learn from your preference data
    - **Cost Optimization** — Replace expensive API judges with self-hosted models
    - **Consistent Standards** — Maintain stable evaluation criteria across applications

---

## Building Approaches

RM-Gallery supports three paths for creating custom graders:

| Approach | Best For | Time to Deploy | Scalability | Cost |
|----------|----------|----------------|-------------|------|
| **[Create Custom Graders](create_custom_graders.md)** | Quick prototyping, rule-based logic, LLM-as-judge | Minutes | High | Low (API-based) |
| **[Generate Graders from Data](generate_graders_from_data.md)** | Auto-generate rubrics from evaluation data, iterative refinement | Hours | High | Medium (API-based) |
| **[Train Reward Models](training/overview.md)** | Learning from data, high-volume evaluation, cost reduction | Hours-Days | Very High | High (training), Low (inference) |

### Decision Framework

Choose your approach based on requirements:

```
                         START
                           │
                           ▼
               ┌─────────────────────┐
               │ Have evaluation     │
               │ data with labels?   │
               └──────┬───────┬──────┘
                      │       │
                  YES │       │ NO
                      │       │
                      ▼       ▼
           ┌──────────────┐  ┌──────────────────┐
           │ Want to      │  │ Need evaluation  │
           │ train model? │  │ now?             │
           └────┬────┬────┘  └────┬─────────┬───┘
                │    │            │         │
            YES │    │ NO     YES │         │ NO
                │    │            │         │
                ▼    ▼            ▼         ▼
          ┌──────┐ ┌──────────┐ ┌────────┐ ┌──────────┐
          │Train │ │Generator │ │Custom  │ │ Define   │
          │Model │ │ (Rubric) │ │Graders │ │ criteria │
          └──────┘ └──────────┘ └────────┘ └──────────┘
                │         │           │            │
                └─────────┴───────────┴────────────┘
                              │
                              ▼
                ┌───────────────────────────────┐
                │  Use in evaluation pipeline   │
                │  (GradingRunner, batch eval)  │
                └───────────────────────────────┘
```

### Comparison Matrix

| Factor | Custom Graders | Generated Graders | Trained Models |
|--------|---------------|-------------------|----------------|
| **Setup Time** | < 1 hour | 1-4 hours | 1-3 days |
| **Data Required** | None | 50-500 examples | 1K-100K examples |
| **Per-Query Cost** | $0.001-$0.01 (API) | $0.001-$0.01 (API) | $0.0001-$0.001 (self-hosted) |
| **Evaluation Speed** | Fast (API latency) | Fast (API latency) | Very Fast (local inference) |
| **Flexibility** | High (change prompts) | High (regenerate rubrics) | Medium (requires retraining) |
| **Consistency** | Medium (LLM variance) | Medium (LLM variance) | High (deterministic) |
| **Domain Adaptation** | Manual prompt engineering | Auto-generated from data | Automatic from data |
| **Interpretability** | Medium | High (explicit rubrics) | Low (learned weights) |

---

## Approach 1: Create Custom Graders

Define evaluation logic using LLM judges or rule-based functions. No training required—start evaluating immediately.

### Implementation Methods

=== "LLM-based Graders"

    ```python
    from rm_gallery.core.graders.llm_grader import LLMGrader
    from rm_gallery.core.models import OpenAIChatModel

    model = OpenAIChatModel(model="qwen3-32b")

    grader = LLMGrader(
        name="domain_expert",
        model=model,
        template="""
        Evaluate the medical accuracy of this response:
        
        Query: {query}
        Response: {response}
        
        Return JSON: {{"score": <0.0-1.0>, "reason": "<explanation>"}}
        """
    )
    ```

=== "Rule-based Graders"

    ```python
    from rm_gallery.core.graders.function_grader import FunctionGrader
    from rm_gallery.core.graders.schema import GraderScore

    async def compliance_checker(response: str) -> GraderScore:
        """Check for required compliance statements."""
        required_terms = ["disclaimer", "terms", "conditions"]
        found = sum(term in response.lower() for term in required_terms)
        score = found / len(required_terms)
        
        return GraderScore(
            name="compliance_check",
            score=score,
            reason=f"Found {found}/{len(required_terms)} required terms"
        )

    grader = FunctionGrader(func=compliance_checker, name="compliance")
    ```

### When to Use

!!! tip "Use Custom Graders When"
    - ✅ Need evaluation logic immediately
    - ✅ Rule-based criteria are well-defined
    - ✅ Moderate evaluation volume (<1M queries/month)
    - ✅ Access to powerful LLM APIs (GPT-4, Claude)

!!! warning "Consider Alternatives When"
    - ❌ High evaluation costs becoming prohibitive
    - ❌ Need to capture nuanced preferences from data

**📖 Detailed Guide:**
- **[Create Custom Graders →](create_custom_graders.md)** — Complete guide with LLM-based and function-based examples
- **[Built-in Graders Reference →](../built_in_graders/overview.md)** — Explore 50+ graders you can customize

---

## Approach 2: Generate Graders from Data

Automatically generate evaluation rubrics and graders from your existing evaluation data. GraderGenerator analyzes response patterns to create structured scoring criteria.

### How It Works

```python
from rm_gallery.core.generator import GraderGenerator
from rm_gallery.core.models import OpenAIChatModel

model = OpenAIChatModel(model="qwen3-32b")
generator = GraderGenerator(model=model)

# Generate grader from evaluation cases
grader = await generator.generate(
    eval_cases=[
        {"query": "Q1", "response": "A1", "score": 0.8},
        {"query": "Q2", "response": "A2", "score": 0.3},
        # ... more examples
    ],
    task_description="Evaluate response helpfulness"
)

# Use generated grader
result = await grader.aevaluate(query="New query", response="New response")
```

### When to Use

!!! tip "Use Generated Graders When"
    - ✅ Have labeled evaluation data (scores/preferences)
    - ✅ Need data-driven rubrics without training models
    - ✅ Want to iterate quickly on evaluation criteria
    - ✅ Prefer interpretable scoring rubrics

!!! warning "Consider Alternatives When"
    - ❌ Don't have any evaluation examples
    - ❌ Need fully automated high-volume evaluation

**📖 Detailed Guide:**
- **[Generate Graders from Data →](generate_graders_from_data.md)** — Step-by-step guide to auto-generate evaluation rubrics

---

## Approach 3: Train Reward Models

Train neural network models on preference data to learn evaluation criteria. Higher upfront cost, but enables cost-effective large-scale evaluation.

### Training Methods

RM-Gallery supports multiple training paradigms via VERL framework:

| Method | Training Data | Best For | Example |
|--------|---------------|----------|---------|
| **[Bradley-Terry](training/bradley_terry.md)** | Preference pairs (chosen/rejected) | Binary preference learning | "Response A > Response B" |
| **[Generative Pointwise](training/generative_pointwise.md)** | Absolute scores (0-5 scale) | Direct quality scoring | "Response quality: 4/5" |
| **[Generative Pairwise](training/generative_pairwise.md)** | Comparison decisions (A/B/tie) | Ranking responses | "Prefer A over B" |
| **[SFT](training/sft.md)** | Multi-turn conversations | Model initialization | "Supervised fine-tuning" |

### Training Architecture

```
┌─────────────────────────────────────────────────┐
│  Preference Data Collection                     │
│  ├─ Human annotations                           │
│  ├─ Existing grader outputs                     │
│  └─ LLM-generated preferences                   │
└──────────────────┬──────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────┐
│  Training with VERL Framework                   │
│  ├─ Multi-GPU/Multi-node (FSDP)                │
│  ├─ Bradley-Terry / Generative objectives      │
│  └─ Ray-based distributed training             │
└──────────────────┬──────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────┐
│  Trained Reward Model                           │
│  ├─ Self-hosted inference                       │
│  ├─ Integrated as RM-Gallery grader            │
│  └─ Cost-effective large-scale evaluation      │
└─────────────────────────────────────────────────┘
```

### Quick Start: Train a Model

=== "Step 1: Prepare Data"

    ```bash
    # Prepare training data
    python -m rm_gallery.core.generator.export \
        --dataset helpsteer2 \
        --output-dir ./data \
        --format parquet
    ```

=== "Step 2: Train Model"

    ```bash
    # Choose training method and run
    cd tutorials/cookbooks/training_reward_model/bradley_terry
    bash run_bt.sh
    ```

=== "Step 3: Integrate"

    ```python
    from rm_gallery.core.models import OpenAIChatModel
    from rm_gallery.core.graders.common import RelevanceGrader

    # Load your trained model
    model = OpenAIChatModel(
        model="./checkpoints/my-reward-model",
        is_local=True
    )

    # Use as a grader
    grader = RelevanceGrader(model=model)
    result = await grader.aevaluate(query="...", response="...")
    ```

### When to Use

!!! tip "Use Trained Models When"
    - ✅ Have preference/score data (>1K examples)
    - ✅ High evaluation volume (>1M queries/month)
    - ✅ Need consistent evaluation criteria
    - ✅ Want to reduce API costs long-term
    - ✅ Can invest in training infrastructure

!!! warning "Consider Alternatives When"
    - ❌ Need results immediately (training takes hours-days)
    - ❌ Don't have sufficient training data

**📖 Detailed Guides:**
- **[Training Overview →](training/overview.md)** — Compare training methods and choose the right approach
- **[Bradley-Terry Training →](training/bradley_terry.md)** — Train with preference pairs (most common)
- **[Generative Pointwise →](training/generative_pointwise.md)** — Train with absolute scores
- **[Generative Pairwise →](training/generative_pairwise.md)** — Train with comparison decisions

---

## Integration with RM-Gallery

Both approaches produce graders that work identically in RM-Gallery:

### Single Evaluation

```python
result = await grader.aevaluate(
    query="What is machine learning?",
    response="ML is a subset of AI..."
)
print(result.score, result.reason)
```

### Batch Evaluation

```python
from rm_gallery.core.runner import GradingRunner

runner = GradingRunner(graders=[custom_grader, trained_grader])
results = await runner.arun_batch([
    {"query": "Q1", "response": "A1"},
    {"query": "Q2", "response": "A2"}
])
```

### Multi-Grader Evaluation

```python
runner = GradingRunner(
    graders=[
        RelevanceGrader(),          # Built-in
        custom_llm_grader,          # Custom LLM-based
        trained_reward_model        # Trained model
    ]
)
results = await runner.arun(query="...", response="...")
```

---

## Tips for Success

!!! tip "Custom Graders"
    - **Start Simple** — Begin with rule-based graders, add LLM judges as needed
    - **Test Thoroughly** — Validate LLM-based graders on diverse inputs
    - **Handle Errors** — Implement robust error handling for production use
    - **Version Control** — Track prompt versions for reproducibility
    - **Monitor Costs** — Set usage limits for API-based graders

!!! tip "Generated Graders"
    - **Quality Over Quantity** — 50-100 high-quality examples beat 500 poor ones
    - **Diverse Examples** — Include edge cases and failure modes
    - **Iterate & Refine** — Regenerate rubrics as you collect more data
    - **Validate Rubrics** — Test generated graders on held-out samples

!!! tip "Trained Models"
    - **Data Quality First** — Prioritize high-quality preference data over quantity
    - **Use Validation Sets** — Hold out 10-20% for evaluation
    - **Start Small** — Begin with smaller models (1B-7B parameters)
    - **Iterate Quickly** — Run short training runs to validate setup
    - **Monitor Drift** — Track evaluation consistency over time

---

## Complete Example: Build an Evaluation Pipeline

Combine multiple custom graders for comprehensive assessment:

```python
from rm_gallery.core.graders.llm_grader import LLMGrader
from rm_gallery.core.graders.function_grader import FunctionGrader
from rm_gallery.core.graders.schema import GraderScore
from rm_gallery.core.runner import GradingRunner
from rm_gallery.core.models import OpenAIChatModel

# 1. Rule-based grader: Length check
async def length_check(response: str) -> GraderScore:
    length = len(response)
    score = 1.0 if 50 <= length <= 500 else 0.5
    return GraderScore(
        name="length_check",
        score=score,
        reason=f"Length: {length} chars"
    )

length_grader = FunctionGrader(func=length_check, name="length")

# 2. LLM-based grader: Domain accuracy
model = OpenAIChatModel(model="qwen3-32b")
accuracy_grader = LLMGrader(
    name="accuracy",
    model=model,
    template="""
    Rate technical accuracy (0.0-1.0):
    Query: {query}
    Response: {response}
    Return JSON: {{"score": <score>, "reason": "<reason>"}}
    """
)

# 3. Trained model: Custom preferences (example)
# trained_model = OpenAIChatModel(model="./checkpoints/my-model", is_local=True)
# preference_grader = RelevanceGrader(model=trained_model)

# 4. Combine in evaluation pipeline
runner = GradingRunner(
    graders=[length_grader, accuracy_grader]  # Add preference_grader when ready
)

# 5. Run evaluation
results = await runner.arun_batch([
    {"query": "Explain quantum computing", "response": "Quantum computing uses..."},
    {"query": "What is AI?", "response": "Artificial Intelligence is..."}
])

for result in results:
    print(f"Scores: {result}")
```

---

## Next Steps

Start building your custom evaluation pipeline:

### Create Custom Graders
- **[Create Custom Graders Guide](create_custom_graders.md)** — LLM-based and rule-based graders
- **[Built-in Graders Reference](../built_in_graders/overview.md)** — Explore existing graders to customize

### Generate Graders from Data
- **[Generate Graders from Data](generate_graders_from_data.md)** — Auto-generate rubrics from evaluation data

### Train Reward Models
- **[Training Overview](training/overview.md)** — Compare training methods
- **[Bradley-Terry Training](training/bradley_terry.md)** — Start with preference pairs
- **[Generative Training](training/generative_pointwise.md)** — Train with score labels

### Deploy at Scale
- **[Run Grading Tasks](../running_graders/run_tasks.md)** — Batch evaluation workflows
- **[Generate Validation Reports](../running_graders/evaluation_reports.md)** — Quality assurance


### Applications
- **[Refine Data Quality](../applications/data_refinement.md)** — Filter training data
- **[Pairwise Model Evaluation](../applications/select_rank.md)** — Compare and rank models using pairwise evaluation

