# Contribute to RM-Gallery

Welcome! RM-Gallery is an open-source reward model platform. Your contributions help make AI alignment and evaluation more accessible to the community.

!!! info "Ways to Contribute"
    We welcome contributions of all kinds:

    - **Bug Fixes**: Identify and resolve issues
    - **New Features**: Add graders, training methods, or integrations
    - **Documentation**: Improve guides and examples
    - **Examples**: Share practical use cases and tutorials

---

## Set Up Development Environment

=== "Step 1: Fork and Clone"

    ```bash
    git clone https://github.com/YOUR_USERNAME/RM-Gallery.git
    cd RM-Gallery
    ```

=== "Step 2: Install Dependencies"

    ```bash
    pip install -e ".[dev]"
    ```

=== "Step 3: Verify Installation"

    ```bash
    python -c "from rm_gallery.core.graders.common import RelevanceGrader; print('✓ Installation successful')"
    ```

---

## Follow Code Standards

!!! tip "Python Coding Standards"
    **Naming:**
    - `snake_case` for functions/variables
    - `PascalCase` for classes
    - `UPPER_CASE` for constants
    
    **Requirements:**
    - Use type hints for all parameters and returns
    - Include docstrings (Args, Returns, Example)
    - Handle exceptions with context
    - Validate inputs early
    - Optimize for readability

---

## Testing Your Changes

Before submitting, verify your changes work:

```bash
# Verify installation
python -c "from rm_gallery.core.graders.common import RelevanceGrader; print('✓ Works')"

# Run tests (optional)
pytest tests/ -v
```

!!! tip "Testing Guidelines"
    - **Manual testing is recommended** — Test your changes with real examples
    - **Automated tests** — We'll help you add tests during PR review if needed
    - **Focus on functionality** — Make sure your code works for your use case

---

## Contributing Graders

When contributing a new grader, follow these guidelines:

### Grader Categories

Place your grader in the appropriate category:

| Category | Path | Purpose |
|----------|------|---------|
| **Common** | `rm_gallery/core/graders/common/` | General-purpose (Relevance, Hallucination) |
| **Agent** | `rm_gallery/core/graders/agent/` | Agent evaluation |
| **Code** | `rm_gallery/core/graders/code/` | Code evaluation |
| **Format** | `rm_gallery/core/graders/format/` | Format validation |
| **Text** | `rm_gallery/core/graders/text/` | Text similarity |
| **Multimodal** | `rm_gallery/core/graders/multimodal/` | Vision/image |

### Grader Specifications

!!! tip "Grader Requirements"
    **Code:**
    - Inherit from `BaseGrader`
    - Use type hints for all parameters
    - Include clear docstring with usage example
    - Return `GraderScore` with `score` (0.0-1.0) and `reason`
    - Use naming: `MyGrader` (class), `my_grader.py` (file), `"my_grader"` (name)
    
    **Documentation:**
    - Add to `docs/built_in_graders/` with When to use, Parameters, Scoring criteria, and Example
    - Use `qwen3-32b` as model name in examples
    
    **Testing:**
    - Manually test your grader works
    - We'll help with automated tests during PR review

### Minimal Grader Template

```python
from rm_gallery.core.graders.base_grader import BaseGrader
from rm_gallery.core.graders.schema import GraderScore

class MyGrader(BaseGrader):
    """Evaluate [specific aspect].
    
    Args:
        model: LLM model for evaluation
    """
    
    def __init__(self, model, **kwargs):
        super().__init__(name="my_grader", **kwargs)
        self.model = model
    
    async def aevaluate(self, query: str, response: str, **kwargs) -> GraderScore:
        """Evaluate response quality."""
        # Your evaluation logic
        score = 0.8
        reason = "Evaluation explanation"
        
        return GraderScore(name=self.name, score=score, reason=reason)
```

---

## Submit Your Contribution

<div class="workflow-single">
<div class="workflow-header">Contribution Workflow</div>

<div class="workflow">
<ol class="workflow-steps">
<li><strong>Create a Branch</strong>

```bash
git checkout -b feat/your-feature-name
```
</li>
<li><strong>Make Changes</strong>

- Write clear, focused commits
- Add tests for new features
- Update relevant documentation</li>
<li><strong>Commit with Convention</strong>

Format: `<type>(<scope>): <subject>`

**Types:** `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

```bash
git commit -m "feat(grader): add code quality grader"
git commit -m "fix(training): resolve memory leak in BT training"
git commit -m "docs(guide): update quickstart tutorial"
```

!!! info "Commit Rules"
    - Use imperative mood ("Add" not "Added")
    - Subject < 50 characters
    - Body wraps at 72 characters
</li>
<li><strong>Push and Open PR</strong>

```bash
git push origin feat/your-feature-name
```

Open a Pull Request on GitHub with:

- Clear description of changes
- Link to related issues
- Test results (if applicable)</li>
</ol>
</div>
</div>

---

## Contribute Documentation

!!! tip "Documentation Guidelines"
    - Start with "What & Why" (1-2 sentences)
    - Use short paragraphs and bullet lists
    - **Bold** key terms for easy scanning
    - Include complete examples with `qwen3-32b` as model
    - See [Documentation Style Guide](style-guide.md) for formatting details

---

## Get Help

Need assistance? Here's how to reach us:

| Type | Where to Go |
|------|-------------|
| **Report Bugs** | [Open an Issue](https://github.com/modelscope/RM-Gallery/issues) |
| **Propose Features** | [Start a Discussion](https://github.com/modelscope/RM-Gallery/discussions) |
| **Contact Team** | Check README for communication channels |

!!! warning "Before Starting Major Work"
    Before starting major work, open an issue to discuss your approach. This prevents duplicate efforts and ensures alignment with project goals.

---

Thank you for contributing to RM-Gallery!

