# Project Task Roadmap

## Project goal

Build a reusable reinforcement learning experimentation project for autonomous driving. Start with DQN in Gymnasium's CarRacing environment, explore genetic optimization, learn TD3 for continuous control, and eventually test the methods in a custom Pygame environment.

## Phase 1 — DQN baseline in CarRacing

- [ ] Initialize the repository and Python environment with `uv`.
- [ ] Create `CarRacing-v3` with `continuous=False` and inspect its observation, action, and reward spaces.
- [ ] Understand the environment's five discrete actions.
- [ ] Implement a CNN-based Q-network for image observations.
- [ ] Integrate experience replay, target-network updates, and epsilon-greedy exploration.
- [ ] Add configurable training, checkpoint saving, and checkpoint resuming.
- [ ] Log per-episode metrics and run metadata.
- [ ] Evaluate with fixed seeds separate from training seeds.
- [ ] Establish a reproducible baseline across multiple training seeds.

**Phase exit criteria:** DQN can train, save, resume, and be evaluated using a repeatable workflow.

## Phase 2 — Genetic DQN

- [ ] Implement a reusable genetic algorithm module.
- [ ] Define individual representation and fitness evaluation.
- [ ] Implement population initialization, selection, mutation, and elitism.
- [ ] Explore crossover after the basic evolutionary loop works.
- [ ] Experiment with evolving DQN hyperparameters.
- [ ] Experiment with evolving initial weights or selected network parameters.
- [ ] Compare Genetic DQN with standard DQN using the same evaluation procedure.
- [ ] Record performance, variability, compute cost, and limitations.

**Phase exit criteria:** A repeatable evolutionary experiment with results compared against standard DQN.

## Phase 3 — TD3

- [ ] Study actor-critic learning and deterministic policies.
- [ ] Implement the actor and twin critic networks.
- [ ] Implement target networks and experience replay.
- [ ] Implement target-policy smoothing and delayed actor updates.
- [ ] Add exploration noise during training.
- [ ] Test TD3 in a manageable continuous-control environment.
- [ ] Train TD3 on `CarRacing-v3` with `continuous=True`.
- [ ] Compare continuous-control results with the DQN baseline where meaningful.

**Phase exit criteria:** A working TD3 training and evaluation pipeline with documented results.

## Phase 4 — Genetic TD3

- [ ] Define how the evolutionary module will interact with TD3.
- [ ] Experiment with evolving actor initialization or actor parameters.
- [ ] Evaluate candidate policies under consistent conditions.
- [ ] Investigate transferring strong candidates into TD3 gradient-based training.
- [ ] Compare Genetic TD3 with standard TD3.
- [ ] Analyze reward, stability, sample efficiency, and computational cost.

**Phase exit criteria:** A reproducible hybrid experiment with results and limitations documented.

## Phase 5 — Custom Pygame driving environment

- [ ] Implement car movement, orientation, steering, acceleration, and braking.
- [ ] Create a track and implement collision handling.
- [ ] Define observation and action spaces.
- [ ] Design and document the reward function.
- [ ] Define termination and truncation conditions.
- [ ] Implement the Gymnasium `reset` and `step` interface.
- [ ] Add optional human rendering with Pygame.
- [ ] Validate the environment with Gymnasium's environment checker.
- [ ] Test DQN, TD3, and evolutionary variants where their action spaces fit.
- [ ] Compare results between the custom environment and CarRacing.

**Phase exit criteria:** A Gymnasium-compatible Pygame driving environment that can be trained and evaluated with the project agents.

## Evaluation checklist

For each comparison, keep evaluation conditions consistent and report:

- [ ] Mean and standard deviation of evaluation reward across runs.
- [ ] Best and worst episode reward.
- [ ] Episode length and track progress or lap completion, where applicable.
- [ ] Environment steps required to reach a defined target.
- [ ] Training time and compute cost.
- [ ] Performance variation across seeds.
- [ ] Limitations and whether the hypothesis was supported, contradicted, or inconclusive.

## Working rules

- Start with Phase 1; do not build all phases before the baseline works.
- Plan each experiment in `experiments/` before running it.
- Commit code before training runs and record the commit in each run's metadata.
- Use distinct training and evaluation seeds. Disable exploration during evaluation.
- Make one major change at a time when comparing against a baseline.
- Generate reports from recorded run data instead of manually copying metrics.
- Treat performance improvements as hypotheses to test, not assumptions.
