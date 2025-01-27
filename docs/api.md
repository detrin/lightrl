# LightRL API Reference

Welcome to the detailed API reference for LightRL. Below you'll find documentation for the key classes and functions available in the library, complete with usage guidelines and examples.

## Bandits Module

LightRL includes a variety of bandit algorithms, each tailored for specific use cases in reinforcement learning environments. The following classes are part of the `lightrl.bandits` module:

### Base Bandit Class

### `Bandit`

::: lightrl.bandits.Bandit

The foundational class for all bandit algorithms. Subclasses provide specialized implementations.

### Epsilon-Based Bandits

These bandits use epsilon strategies to balance exploration and exploitation.

- **`EpsilonGreedyBandit`**

  ::: lightrl.bandits.EpsilonGreedyBandit

  Implements an epsilon-greedy algorithm, allowing for a tunable exploration rate.

- **`EpsilonFirstBandit`**

  ::: lightrl.bandits.EpsilonFirstBandit

  Prioritizes exploration for a set number of initial steps before switching to exploitation.

- **`EpsilonDecreasingBandit`**

  ::: lightrl.bandits.EpsilonDecreasingBandit

  Uses a decreasing epsilon value over time to reduce exploration as understanding improves.

### Other Bandit Strategies

- **`UCB1Bandit`**

  ::: lightrl.bandits.UCB1Bandit

  Employs the UCB1 algorithm, focusing on arm pulls with calculated confidence bounds.

- **`GreedyBanditWithHistory`**

  ::: lightrl.bandits.GreedyBanditWithHistory

  A variant that uses historical performance data to adjust its greedy selection strategy.

## Runners Module

### `two_state_time_dependent_process`

::: lightrl.runners.two_state_time_dependent_process

This function models a process with time-dependent state transitions, useful in simulating dynamic environments.

---

If you have any questions or require further assistance, feel free to [open an issue](https://github.com/detrin/lightrl/issues).