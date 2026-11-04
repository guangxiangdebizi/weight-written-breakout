# Weight-written Breakout: Constructing a Neural Network Without Training

English | [简体中文](README.zh-CN.md)

**Weight-written Breakout** is an AI-assisted learning experiment. Starting from game rules, we directly construct a neural network's architecture and weights to play MinAtar Breakout.

This repository preserves three iterations, checkpoints, per-episode results, and reproduction instructions as a reference for others. Questions, corrections, reproduction reports, and improvements are welcome through [Issues](https://github.com/guangxiangdebizi/weight-written-breakout/issues) and pull requests.

## What "without training" means here

- No gradient descent, backpropagation, reinforcement-learning training, or fitting weights to gameplay samples.
- Weights are constructed from explicit rules, geometric calculations, and finite-horizon dynamic programming.
- **Development feedback was used:** we inspected failures, revised the design, compared candidates, and then froze the selected model for held-out evaluation. This was not a zero-feedback, one-shot process.
- At inference time, the controller executes only the fixed neural network and its internal memory updates. It does not call an LLM or run an external dynamic-programming solver.
- "Without training" refers to the target network we construct. The AI acting as its designer was itself pretrained.

Dynamic programming still optimizes action selection. LLM inference, weight construction, and evaluation also consume computation. No training does not mean no computation or optimization.

## My intuition: an ant inside a maze, and a designer looking at the maze

Here is the analogy that motivated this experiment: **learning a policy through trial and error feels like putting an ant inside a maze, letting it act and receive feedback, and using a learning algorithm to update its behavior until it finds a way out.**

Direct weight construction feels more like someone with existing knowledge examining the maze's layout and rules from the outside, designing a route, and writing that behavior into the ant's controller. In this experiment, an already capable AI is the designer, and the small neural network is the controller being constructed.

When I describe these as being in "different dimensions," I mean **a change in perspective and level of reasoning**: one develops a policy through actions and feedback within the task; the other analyzes the task from outside, derives a strategy, and constructs a network that implements it.

This made me want to explore whether a designer's existing knowledge and reasoning could become a working small network directly, reducing the need for that target network to learn everything from scratch. Our workflow was:

```text
Understand the rules -> Design a strategy -> Construct architecture and weights
    -> Test -> Revise the design
```

The outside view depends on access to the task's rules and structure, together with the designer's prior knowledge. This analogy mainly describes policy learning through feedback; classical planning and model-based methods also use structure and prior knowledge.

## Starting point and iterations

The starting point was the MIT CSAIL researchers' blog post [Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/). Its Experiment 5 asks an LLM to write the weights of small MinAtar networks directly. Our learning experiment explores designing our own network, encoding a strategy in its weights, and iterating with a development set.

| Stage | Main design | Mean held-out score in that round | Comparison in the same round |
| --- | --- | ---: | ---: |
| [V1](experiments/v1) | Detect ball/trail configurations and predict landing positions after side-wall reflections | 16.539 | Reactive ball tracking: 8.366 |
| [V2](experiments/v2) | Read the brick layout and anticipate the next landing while the ball is still ascending | 255.997 | V1 rerun on the same seeds: 17.959 |
| [V3](experiments/v3) | Add short-term memory and compile a sticky-action-aware controller into weights | 672.681 | Released reference model rerun on the same seeds: 512.860 |

**Different rounds used different test seeds. Direct comparisons in this table should be made within a row.** The design changes, unselected candidates, and seed splits are documented in the [experiment log (Chinese)](docs/experiment-process.md).

The project initiator proposed the direction and requested iterations. An AI assistant performed most of the network construction, implementation, experiment execution, and documentation. This repository records actual runs; it does not imply that the initiator independently wrote or derived every component.

## Final comparison: matched game protocol, different model sizes

We downloaded and preserved the [original researchers' released checkpoint](https://github.com/RikiyaT/lm_understand-logs/tree/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp5_write_from_scratch). We first reran it to check compatibility with the reported result, then compared it with the frozen V3 on the same fresh test seeds.

Shared settings: MinAtar **1.0.15**; the same 600-dimensional object-channel encoding; actions sampled from softmax; sticky-action probability **0.1**; at most **10,000 steps** per episode; **1,000 episodes** per model; seeds **200000–200999**.

| Metric | V3 | Released reference weights |
| --- | ---: | ---: |
| Mean score | 672.681 | 512.860 |
| 95% bootstrap interval for the mean | 658.322–687.535 | 497.943–527.695 |
| Median score | 764.0 | 647.5 |
| Episodes reaching the step limit | 784 | 517 |

V3's mean score was **31.16%** higher. The paired mean difference was **159.821 points**, with a 95% bootstrap interval of **142.271–178.183 points**. Each broken brick earns one point. Episodes reaching the step limit were included with their accumulated scores, not discarded.

**Model sizes and development conditions differ:**

| Property | V3 | Released reference model |
| --- | --- | --- |
| Network | Recurrent ReLU network | Tanh MLP with two hidden layers |
| Dense parameter slots | 5,393,916 | 20,486 |
| Nonzero weights and biases | 48,202 | 815 |
| Internal memory | 12 dimensions | None |
| Construction process | Multiple rounds with development feedback | One-shot, without environment feedback |

The result shows that this larger, iteratively developed task-specific network scored higher under the stated protocol. It does not establish an advantage at equal model size or development budget, or show that general-purpose LLMs can be constructed without training.

In a stress test with sticky-action probability **0.25**, each model played 200 episodes: V3 scored **163.990**, and the reference model scored **75.005**. Both degraded substantially. Statistics and raw CSV files are available in the [full results](experiments/v3/comparison.json).

### A recorded trajectory

![Recorded trajectory rendered from seed 200000: V3 on the left, the released reference model on the right](assets/replay.gif)

This animation is rendered from recorded evaluation states, not a native game screenshot. It uses the first test episode, `seed=200000`, with one frame per 20 steps; it is not a selected best run. V3 scored 793 and reached the 10,000-step limit. The reference model scored 289 and ended at step 4,278. These single-episode scores are not the averages in the table above.

## Quick start

Verified Python version: **3.13**. Use an isolated virtual environment.

```bash
git clone https://github.com/guangxiangdebizi/weight-written-breakout.git
cd weight-written-breakout
python -m venv .venv
```

Activate it with `.venv\Scripts\Activate.ps1` in Windows PowerShell, or `source .venv/bin/activate` on Linux/macOS.

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python experiments/v3/verify.py --model experiments/v3/recurrent-0.01.npz --reference experiments/v3/reference-hand.npz
```

Run the full 1,000-episode comparison:

```bash
python experiments/v3/benchmark.py --model experiments/v3/recurrent-0.01.npz --kind compiled_fast --plan experiments/v3/protocol.json --split heldout --output work/rerun-ours
python experiments/v3/benchmark.py --model experiments/v3/reference-hand.npz --kind reference --plan experiments/v3/protocol.json --split heldout --output work/rerun-reference
```

Output directories must not already exist, to prevent overwriting results. `compiled_fast` uses bit-parallel evaluation of the neural circuit; replace it with `compiled` to use the generic sparse-matrix/ReLU backend.

Rebuild the weights from rules, without gameplay trajectories:

```bash
python experiments/v3/build_weights.py --output work/rebuilt
```

Instructions for the other versions are in the [reproduction notes (Chinese)](docs/reproduce.md).

## How the network works

V3 follows this computation chain:

```text
600 observation values + 12 internal memory values
    -> Detect trajectory and collision conditions
    -> Estimate catch time, catch positions, and previous movement
    -> Activate the neurons encoding the fixed control table
    -> Produce 6 action logits + the next 12 memory values
```

During weight construction, we compute a control table from the known game transitions and compile it into sparse ReLU connections. The table is not solved again during inference. Equations, memory updates, and implementation limitations are described in the [mechanism notes (Chinese)](docs/how-it-works.md).

## Repository contents

- `experiments/v1/`: first implementation, weights, and complete results.
- `experiments/v2/`: six second-round candidates and held-out evaluation.
- `experiments/v3/`: recurrent network, three candidates, released reference, and final comparison.
- `docs/`: experiment history, mechanism, and reproduction notes, currently in Chinese.
- `tests/`: weight reconstruction, hashes, statistics, and data-integrity checks.
- `scripts/render_replay.py`: rebuild the example animation from recorded trajectories.

## Feedback and discussion

This is a learning experiment and reference implementation. Corrections, alternative implementations, and reproduction reports are welcome. In particular, we would like to explore:

1. Can the number of neurons and nonzero connections be reduced substantially while preserving performance?
2. What can direct weight construction achieve with matched model size, memory, and development budgets?
3. Which tasks remain suitable when the designer does not have access to complete environment source code?

Please include versions, seeds, commands, and results with reproduction feedback. See the [contributing guide (Chinese)](CONTRIBUTING.md) for details.

## License and sources

Original code, constructed weights, and project-generated results are released under the [MIT License](LICENSE).

The third-party `reference-hand.npz` retains **CC BY 4.0**; authorship, source, and hashes are in the [attribution notice](experiments/v3/ATTRIBUTION.txt). MinAtar is an independently installed **GPL-3.0** dependency and is not bundled here. The released reference weights were not modified or used to initialize our models.

See also the [third-party notices](THIRD_PARTY_NOTICES.md).

- [Original research blog](https://rikiyat.github.io/blog/weight-space-programming/)
- [Original experiment logs and checkpoints](https://github.com/RikiyaT/lm_understand-logs)
- [Official MinAtar repository](https://github.com/kenjyoung/MinAtar)
