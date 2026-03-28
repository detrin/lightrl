![](https://i.imgur.com/VXUdVOB.jpeg)
# LightRL

A lightweight multi-armed bandit library for Python. Zero heavy dependencies. Built for agents.

![PyPI - Python Version](https://img.shields.io/pypi/pyversions/lightrl) [![Docs](https://github.com/detrin/lightrl/actions/workflows/docs.yml/badge.svg)](https://detrin.github.io/lightrl/) ![main](https://github.com/detrin/lightrl/actions/workflows/test_main.yml/badge.svg) ![PyPI - Version](https://img.shields.io/pypi/v/lightrl)

## Why LightRL

Most RL packages (Vowpal Wabbit, RLlib, MABWiser) are built for data science pipelines — they require heavy dependencies, custom serialization, and framework buy-in. LightRL is built for a different use case: **operational decisions in software systems and AI agents**.

Think of LightRL as `functools.lru_cache` for decision-making. You don't reach for Redis when you need to memoize a function. You don't reach for Vowpal Wabbit when you need an agent to learn which API endpoint is fastest.

### The case for lightweight bandits

| | LightRL | Vowpal Wabbit | MABWiser | RLlib |
|---|---|---|---|---|
| Dependencies | `tqdm` | C++ runtime | sklearn, numpy, scipy | Ray, torch |
| Install size | ~50KB | ~50MB | ~200MB+ | ~1GB+ |
| Core code | ~300 lines | ~100k lines | ~3k lines | ~500k lines |
| Persistence | `bandit.save("state.json")` | Custom model files | Pickle | Ray checkpoints |
| Time to integrate | Minutes | Hours | Hours | Days |
| Agent-native API | `BanditRouter` | No | No | No |

### When to use LightRL

LightRL is the right choice when:

- **You need a decision, not a research paper.** Which endpoint is fastest? What batch size avoids rate limits? Which prompt template works best? These don't need gradient descent.
- **You're building an agent.** LLMs burn tokens and latency "reasoning" about operational choices. A bandit answers in microseconds with better accuracy after 50 observations.
- **Dependencies matter.** Lambda functions, edge devices, minimal containers, CI pipelines — anywhere scipy is too heavy.
- **You want to understand the code.** The entire library is auditable in 10 minutes. No hidden complexity.

### When NOT to use LightRL

- You need industrial-scale contextual bandits processing billions of events (use Vowpal Wabbit)
- You need full reinforcement learning with environments and policies (use RLlib)
- You need Bayesian optimization with Gaussian processes (use BoTorch)

## Installation

```
pip install lightrl
```

Or with uv:
```
uv pip install lightrl
```

## Quick Start

### Simple bandit
```python
from lightrl import EpsilonGreedyBandit

bandit = EpsilonGreedyBandit(arms=["model_a", "model_b", "model_c"], epsilon=0.1)

for _ in range(1000):
    arm = bandit.select_arm()
    reward = get_reward(bandit.arms[arm])  # your reward function
    bandit.update(arm, reward)

bandit.report()
```

### Agent router with persistence
```python
from lightrl import BanditRouter, ThompsonBandit

router = BanditRouter()
router.register("model", ThompsonBandit(arms=["haiku", "sonnet", "opus"]))

# agent loop
model_idx = router.select("model")
model = router._bandits["model"].arms[model_idx]
# ... agent does work, gets quality score ...
router.update("model", model_idx, reward=quality_score)

# persist across restarts
router.save("agent_state.json")
router = BanditRouter.load("agent_state.json")
```

### Contextual decisions
```python
from lightrl import LinUCBBandit

bandit = LinUCBBandit(n_arms=3, n_features=4, alpha=1.0)

context = [task_complexity, input_length, is_code, urgency]
arm = bandit.select_arm(context)
# ... execute with chosen arm ...
bandit.update(arm, context, reward)
```

## Features

### Bandit Strategies
| Strategy | Best for |
|---|---|
| `EpsilonGreedyBandit` | Simple explore/exploit with fixed exploration rate |
| `EpsilonFirstBandit` | Pure exploration phase followed by exploitation |
| `EpsilonDecreasingBandit` | Exploration that decays over time |
| `UCB1Bandit` | Upper confidence bound — principled exploration |
| `ThompsonBandit` | Bayesian posterior sampling — best general-purpose |
| `GreedyBanditWithHistory` | Sliding window for non-stationary environments |
| `LinUCBBandit` | Context-dependent decisions with linear features |

### Cross-cutting features
- **Warm start** — initialize with prior beliefs: `EpsilonGreedyBandit(arms=[...], priors=[0.3, 0.7, 0.9])`
- **EMA decay** — make any bandit adaptive to change: `EpsilonGreedyBandit(arms=[...], ema_alpha=0.1)`
- **Persistence** — JSON save/load on all bandits and the router
- **BanditRouter** — manage multiple named decision points with one object

### Runners
- `two_state_time_dependent_process` — alive/waiting state machine for rate-limited systems

## Examples

See [`examples/`](examples/) for runnable scripts:

| Example | Demonstrates |
|---|---|
| `ab_testing.py` | A/B test across landing page variants |
| `ad_serving.py` | Ad placement optimization with explore-first |
| `resource_allocation.py` | Dynamic worker pool sizing |
| `hyperparameter_search.py` | Bandit-based learning rate search |
| `network_routing.py` | Endpoint selection with sliding window |
| `retry_backoff.py` | Learning optimal retry wait times |
| `prompt_template_selection.py` | Per-task-type prompt template optimization |
| `minimal_example.py` | Two-state process with failure simulation |

## Development

```bash
uv venv --python 3.13
source .venv/bin/activate
uv pip install -e ".[dev]"
pre-commit install && pre-commit install --hook-type commit-msg
```

Run tests:
```bash
pytest -v tests/
tox
```

Read more about [Multi-armed bandits](https://en.wikipedia.org/wiki/Multi-armed_bandit).
