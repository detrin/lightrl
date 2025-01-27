# LightRL API Reference

Welcome to the detailed API reference for LightRL. Below you'll find documentation for the key classes and functions available in the library, complete with usage guidelines and examples.

## Bandits Module

LightRL includes a variety of bandit algorithms, each tailored for specific use cases in reinforcement learning environments. The following classes are part of the `lightrl.bandits` module:

### Base Bandit Class

**`Bandit`**: The foundational class for all bandit algorithms. Subclasses provide specialized implementations.
::: lightrl.bandits.Bandit

### Epsilon-Based Bandits

These bandits use epsilon strategies to balance exploration and exploitation.

**`EpsilonGreedyBandit`**: Implements an epsilon-greedy algorithm, allowing for a tunable exploration rate.
::: lightrl.bandits.EpsilonGreedyBandit

**`EpsilonFirstBandit`**: Prioritizes exploration for a set number of initial steps before switching to exploitation.
::: lightrl.bandits.EpsilonFirstBandit

**`EpsilonDecreasingBandit`**: Uses a decreasing epsilon value over time to reduce exploration as understanding improves.
::: lightrl.bandits.EpsilonDecreasingBandit

### Other Bandit Strategies

**`UCB1Bandit`**: Employs the UCB1 algorithm, focusing on arm pulls with calculated confidence bounds.
::: lightrl.bandits.UCB1Bandit

**`GreedyBanditWithHistory`**: A variant that uses historical performance data to adjust its greedy selection strategy.
::: lightrl.bandits.GreedyBanditWithHistory

## Runners Module

**`two_state_time_dependent_process`**: This function models a process with time-dependent state transitions, useful in simulating dynamic environments.
::: lightrl.runners.two_state_time_dependent_process

---

If you have any questions or require further assistance, feel free to [open an issue](https://github.com/detrin/lightrl/issues).