# RM-Gallery

## Why RM-Gallery?
**RM-Gallery** unifies reward signals and evaluation metrics into one **Grader** interface—with pre-built graders, flexible customization, and seamless framework integration.

### Key Features

<div class="key-features" markdown>

+ **Systematic & Quality-Assured Grader Library**: Access N+ production-ready graders featuring a comprehensive taxonomy, rigorously validated for reliable performance.
    - **Multi-Scenario Coverage:** Extensive support for diverse domains including Agent, text, code, math, and multimodal tasks via specialized graders.
    - **Holistic Agent Evaluation:** Beyond final outcomes, we assess the entire lifecycle—including trajectories and specific components (Memory, Reflection, Tool Use).
    - **Quality Assurance:** Built for reliability. Every grader comes with benchmark datasets and pytest integration for immediate quality validation.

+ **Flexible Grader Building**: Choose the build method that fits your requirements:
    - **Customization:** Easily extend or modify pre-defined graders to fit your specific needs.
    - **Data-Driven Rubrics:** Have a few examples but no clear rules? Use our tools to automatically generate white-box evaluation criteria (Rubrics) based on your data.
    - **Trainable Judge Models:** For high-scale scenarios, train dedicated Judge models as Graders. We support SFT, **Bradley-Terry models, and Reinforcement Learning** workflows.

+ **Easy Integration**: Seamlessly connect with mainstream evaluation platforms (e.g., LangSmith, LangFuse) and training frameworks (e.g., VERL) using our comprehensive tutorials and flexible APIs.

</div>



## Quick Tutorials

<div class="card-grid">
  
  <a href="./tutorials/agent-evaluation.md" class="feature-card">
    <div class="card-header card-header-lg">
      <img src="https://unpkg.com/lucide-static@latest/icons/bot.svg" class="card-icon card-icon-agent">
      <h3>Evaluate An AI Agent</h3>
    </div>
    <p class="card-desc card-desc-lg">
      <b>Comprehensive evaluation for AI Agents:</b> Learn to evaluate the full lifecycle—including final response, trajectory, tool usage, plan, memory, reflection, observation—using RM-Gallery Graders.
    </p>
  </a>

  <a href="./tutorials/build-rewards.md" class="feature-card">
    <div class="card-header card-header-lg">
      <img src="https://unpkg.com/lucide-static@latest/icons/brain-circuit.svg" class="card-icon card-icon-tool">
      <h3>Build Rewards for Training</h3>
    </div>
    <p class="card-desc card-desc-lg">
      <b>Construct High-Quality Reward Signals:</b> Create robust reward functions for model and agent alignment by aggregating diverse graders with custom weighting and high-concurrency support.
    </p>
  </a>

</div>


## More Tutorials

### Built-in Graders

<div class="card-grid">

  <a href="#agent-graders" class="feature-card">
    <div class="card-header">
      <img src="https://unpkg.com/lucide-static@latest/icons/bot.svg" class="card-icon card-icon-agent">
      <h3>Agent</h3>
    </div>
    <p class="card-desc">
      Agent graders for evaluating various aspects of AI agent behavior. These graders assess action selection, tool usage, memory management, planning, reflection, and overall trajectory quality.
    </p>
  </a>

  <a href="#general-graders" class="feature-card">
    <div class="card-header">
      <img src="https://unpkg.com/lucide-static@latest/icons/globe.svg" class="card-icon card-icon-general">
      <h3>General Tasks</h3>
    </div>
    <p class="card-desc">
      Assess fundamental capabilities such as instruction following, text quality, safety guardrails, and format.
    </p>
  </a>

  <a href="#multimodal-graders" class="feature-card">
    <div class="card-header">
      <img src="https://unpkg.com/lucide-static@latest/icons/image.svg" class="card-icon card-icon-multimodal">
      <h3>Multimodal</h3>
    </div>
    <p class="card-desc">
      Vision-language graders for evaluating AI responses involving images. These graders assess image-text coherence, image helpfulness, and text-to-image generation quality.
    </p>
  </a>

  <a href="#math-code-graders" class="feature-card">
    <div class="card-header">
      <img src="https://unpkg.com/lucide-static@latest/icons/calculator.svg" class="card-icon card-icon-math">
      <h3>Math & Code</h3>
    </div>
    <p class="card-desc">
      Specialized graders for evaluating code generation and mathematical problem-solving capabilities. These graders assess syntax correctness, execution results, code style, and mathematical expression accuracy.
    </p>
  </a>

</div>


### Build Graders

<div class="card-grid">

  <a href="#Customization" class="feature-card-sm">
    <div class="card-header">
      <img src="https://unpkg.com/lucide-static@latest/icons/wrench.svg" class="card-icon card-icon-tool">
      <h3>Customization</h3>
    </div>
    <p class="card-desc">
      <b>Clear requirements, but no existing grader?</b> If you have explicit rules or logic, use our Python interfaces or Prompt templates to quickly define your own grader.
    </p>
  </a>

  <a href="#Data-Driven Rubrics" class="feature-card-sm">
    <div class="card-header">
      <img src="https://unpkg.com/lucide-static@latest/icons/database.svg" class="card-icon card-icon-data">
      <h3>Data-Driven Rubrics</h3>
    </div>
    <p class="card-desc">
      <b>Ambiguous requirements, but have few examples?</b> Use the GraderGenerator to automatically summarize evaluation Rubrics from your annotated data, and generate a llm-based grader.
    </p>
  </a>

  <a href="#Trainable Grader" class="feature-card-sm">
    <div class="card-header">
      <img src="https://unpkg.com/lucide-static@latest/icons/scale.svg" class="card-icon card-icon-integration">
      <h3>Trainable Judge Model</h3>
    </div>
    <p class="card-desc">
      <b>Massive data and need peak performance?</b> Use our training pipeline to train a dedicated Judge Model. This is ideal for complex scenarios where prompt-based grading falls short.
    </p>
  </a>

</div>

### Integrations

<div class="card-grid">

  <a href="#eval-integrations" class="feature-card">
    <div class="card-header">
      <img src="https://unpkg.com/lucide-static@latest/icons/bar-chart-3.svg" class="card-icon card-icon-integration">
      <h3>Evaluation Frameworks</h3>
    </div>
    <p class="card-desc">
      Seamlessly connect with mainstream platforms like <strong>LangSmith</strong> and <strong>LangFuse</strong>. Streamline your evaluation pipelines and monitor agent performance with flexible APIs.
    </p>
  </a>

  <a href="#training-integrations" class="feature-card">
    <div class="card-header">
      <img src="https://unpkg.com/lucide-static@latest/icons/dumbbell.svg" class="card-icon card-icon-tool">
      <h3>Training Frameworks</h3>
    </div>
    <p class="card-desc">
      Directly integrate into training loops such as <strong>VERL</strong>. Use Graders as high-quality reward functions for RLHF/RLAIF to align models effectively.
    </p>
  </a>

</div>


### Applications

<div class="card-grid">

  <a href="#data-refinement" class="feature-card">
    <div class="card-header">
      <img src="https://unpkg.com/lucide-static@latest/icons/gem.svg" class="card-icon card-icon-data">
      <h3>Data Refinement</h3>
    </div>
    <p class="card-desc">
      Automate the curation of high-quality datasets. Use Graders to filter, rank, and synthesize training data for Supervised Fine-Tuning (SFT).
    </p>
  </a>

  <div class="feature-card-wip">
    <div class="card-header">
      <img src="https://unpkg.com/lucide-static@latest/icons/rocket.svg" class="card-icon card-icon-agent">
      <h3>Agent Training</h3>
      <span class="badge-wip">🚧 Work in Progress</span>
    </div>
    <p class="card-desc">
      Utilize Graders as precise <strong>Reward Models</strong> in Reinforcement Learning (RLHF/RLAIF) loops to align agent behaviors and improve success rates.
    </p>
  </div>

</div>

### Improve Graders <span class="badge-wip">🚧 Work in Progress</span>

<div class="card-grid">

  <div class="feature-card-wip">
    <div class="card-header">
      <img src="https://unpkg.com/lucide-static@latest/icons/vote.svg" class="card-icon card-icon-general">
      <h3>Voting Strategies</h3>
    </div>
    <p class="card-desc">
      Enhance reliability through <strong>Best-of-N</strong> or <strong>Majority Voting</strong>. Aggregate results from multiple grader instances to reduce variance.
    </p>
  </div>

  <div class="feature-card-wip">
    <div class="card-header">
      <img src="https://unpkg.com/lucide-static@latest/icons/sparkles.svg" class="card-icon card-icon-multimodal">
      <h3>Advanced Few-Shot</h3>
    </div>
    <p class="card-desc">
      Boost accuracy with <strong>Dynamic Few-Shot</strong>. Automatically retrieve the most relevant examples (RAG) to guide the grader for specific inputs.
    </p>
  </div>

</div>
### Others

| documentation | Description | Link |
|-------------|-----------|-------------|
| running graders  | xxx | xxx|
| validating graders  | xxx | xxx|