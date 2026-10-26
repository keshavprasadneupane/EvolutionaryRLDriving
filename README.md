# EvolutionaryRLDriving

A reinforcement learning project exploring DQN, genetic optimization, and TD3 for autonomous driving. Experiments begin with Gymnasium's CarRacing environment and may later extend to a custom Pygame environment.

## Project Structure

```text
EvolutionaryRLDriving/
├── agents/
│   ├── dqn/                     # DQN implementation
│   └── td3/                     # TD3 implementation
├── evolution/                   # Reusable genetic algorithm components
├── workflows/
│   ├── dqn_baseline/
│   │   ├── train.py
│   │   ├── test.py
│   │   ├── config.json
│   │   └── models/
│   ├── genetic_dqn/
│   ├── td3_baseline/
│   └── genetic_td3/
├── training/
│   └── run_logger.py
├── evaluation/
│   └── aggregate.py
├── experiments/                 # Experiment plans and conclusions
├── journal/                     # Daily development notes
├── runs/                        # Per-run logs and checkpoints
├── results/                     # Generated summaries and plots
├── tests/
├── pyproject.toml
├── uv.lock
├── README.md
└── task.md
```

`agents/` contains algorithm implementations without experiment-specific training or testing scripts. Each workflow assembles an agent for a particular approach and owns its training, testing, configuration, and model exports. Shared utilities provide functionality used across workflows without coupling their implementations.

The structure can evolve as the project grows; introduce abstractions when they provide a concrete benefit.

## Running Workflows

Run commands from the repository root using `uv`:

```bash
uv run python -m workflows.dqn_baseline.train
uv run python -m workflows.dqn_baseline.test
```

Use `uv run python -m package.module` for other modules as appropriate. Keep configuration paths and model paths explicit rather than relying on the current working directory.

## Experiment Tracking

Use `RunLogger` to create a separate record for each training run:

```text
runs/<experiment>/<timestamp>_seed<seed>/
├── config.json
├── run_info.json
├── metrics.csv
├── eval.csv
└── checkpoint.pt
```

Each run records its configuration, seed, Git commit, software versions, device, status, and training metrics. The training script is responsible for saving the checkpoint.

After completing an experiment, aggregate its runs:

```bash
uv run python -m evaluation.aggregate dqn_baseline --window 20
```

Generated reports are written to:

```text
results/<experiment>/
├── summary.csv
├── summary.md
└── learning_curve.png
```

Use `--target <reward>` to measure environment steps to a target reward, or `--include-unfinished` to include interrupted, failed, and running runs.

## Experiment Guidelines

- Record the research question, setup, training budget, seeds, and success criteria in an experiment file before training. Start from `experiments/TEMPLATE.md`.
- Commit code before each run and record any uncommitted changes.
- Use separate, fixed seeds for evaluation and disable exploration during evaluation.
- Keep environment settings, interaction budgets, and evaluation procedures consistent when comparing methods.
- Generate reports from recorded run data rather than manually copying results.
- Document findings, limitations, and next steps in the experiment report and daily journal.

## Development Roadmap

1. **DQN baseline:** Train a CNN-based DQN on `CarRacing-v3` with `continuous=False`.
2. **Genetic DQN:** Explore evolutionary optimization and compare it with standard DQN.
3. **TD3:** Implement and evaluate continuous-control reinforcement learning.
4. **Genetic TD3:** Investigate combining evolutionary search with TD3.
5. **Custom environment:** Build a Gymnasium-compatible Pygame driving environment and evaluate the agents.

See [`task.md`](task.md) for the full task checklist.
