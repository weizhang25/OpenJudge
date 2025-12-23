<div align="center">

<p align="center">
  <img src="./docs/images/logo.png" alt="RM-Gallery Logo" width="500">
</p>

Precision Evaluation, Quality Rewards: Driving Agent Excellence.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue)](https://pypi.org/project/rm-gallery/)
[![PyPI](https://img.shields.io/badge/pypi-v0.2.0-blue?logo=pypi)](https://pypi.org/project/rm-gallery/)
[![Documentation](https://img.shields.io/badge/docs-online-blue?logo=markdown)](https://modelscope.github.io/RM-Gallery/)

[Documentation](https://modelscope.github.io/RM-Gallery/) | [Contributing](./docs/community/contributing.md) | [中文](./README_zh.md)

</div>



Open-Judge unifies reward signals and evaluation metrics into one **Grader** interface—with pre-built graders, flexible customization, and seamless framework integration.

## News

- **2025-10-20** - [Auto-Rubric: Learning to Extract Generalizable Criteria for Reward Modeling](https://arxiv.org/abs/2510.17314) - We released a new paper on learning generalizable reward criteria for robust modeling.
- **2025-10-17** - [Taming the Judge: Deconflicting AI Feedback for Stable Reinforcement Learning](https://arxiv.org/abs/2510.15514) - We introduced techniques to align judge feedback and improve RL stability.
- **2025-07-09** - Released RM-Gallery v0.1.0 on [PyPI](https://pypi.org/project/rm-gallery/)

## Key Features

<div class="key-features" markdown>

+ **Systematic & Quality-Assured Grader Library**: Access N+ production-ready graders featuring a comprehensive taxonomy, rigorously validated for reliable performance.
    - **Multi-Scenario Coverage:** Extensive support for diverse domains including Agent, text, code, math, and multimodal tasks via specialized graders.
    - **Holistic Agent Evaluation:** Beyond final outcomes, we assess the entire lifecycle—including trajectories and specific components (Memory, Reflection, Tool Use).
    - **Quality Assurance:** Built for reliability. Every grader comes with benchmark datasets and pytest integration for immediate quality validation.

+ **Flexible Grader Building Methods**: Choose the build method that fits your requirements:
    - **Customization:** Easily extend or modify pre-defined graders to fit your specific needs.
    - **Data-Driven Rubrics:** Have a few examples but no clear rules? Use our tools to automatically generate white-box evaluation criteria (Rubrics) based on your data.
    - **Trainable Judge Models:** For high-scale scenarios, train dedicated Judge models as Graders. We support SFT, **Bradley-Terry models, and Reinforcement Learning** workflows.

+ **Easy Integration**: Seamlessly connect with mainstream evaluation platforms (e.g., LangSmith, LangFuse) and training frameworks (e.g., VERL) using our comprehensive tutorials and flexible APIs.
</div>


## Installation
```bash
pip install rm-gallery
```
More installation methods can be found in the [here](https://modelscope.cn/docs/rm-gallery/installation).

## Quickstart
```python
import asyncio
from rm_gallery.core.models import OpenAIChatModel
from rm_gallery.core.graders.common.relevance import RelevanceGrader


# step1 create model client
model = OpenAIChatModel(model="qwen3-32b")

# step2 choose and initialize proper grader
grader = RelevanceGrader(model=model)

# step3 Prepare data

data = {
    "query": "What is machine learning?",
    "response": "Machine learning is a subset of AI that enables computers to learn from data.",
}

# step 4 Evaluate using the data
result = await grader.aevaluate(**data)

print(f"Score: {result.score}")  # Score: 5
print(f"Reason: {result.reason}")
```
Complete Quickstart can be found in [here](https://modelscope.cn/docs/rm-gallery/quickstart).

## Integrations

| Integration | Documentation |
|-------------|---------------|
| LangSmith   | [LangSmith](https://modelscope.cn/docs/rm-gallery/integrations/langsmith) |
| LangFuse    | [LangFuse](https://modelscope.cn/docs/rm-gallery/integrations/langfuse) |

## Contributing
We welcome contributions from the community! 
1. Raise and comment on [Issues](https://github.com/modelscope/RM-Gallery/issues).
2. Open a PR - Whether you're fixing bugs, adding new features, improving documentation, or sharing
ideas, your contributions help make Open-Judge better for everyone. See [Contributing](https://github.com/modelscope/RM-Gallery/blob/main/CONTRIBUTING.md) for more details.

## Citation

If you use Open-Judge in your research, please cite:

```
@software{
title = {RM-Gallery: XXXX},
author = {The Open-Judge Team},
url = {https://github.com/modelscope/Open-Judge},
month = {07},
year = {2025}
}
```