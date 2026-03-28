import math
import random
from abc import ABC, abstractmethod


class Bandit(ABC):
    def __init__(self, arms: list) -> None:
        self.arms = arms
        self.q_values = [0.0] * len(arms)
        self.counts = [0] * len(arms)

    @abstractmethod
    def select_arm(self) -> int: ...

    def update(self, arm_index: int, reward: float) -> None:
        self.counts[arm_index] += 1
        n = self.counts[arm_index]
        self.q_values[arm_index] = ((n - 1) * self.q_values[arm_index] + reward) / n

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(arms={self.arms})"

    def report(self) -> None:
        print("Q-values per arm:")
        for arm, q, cnt in zip(self.arms, self.q_values, self.counts):
            print(f"  num_tasks={arm}: avg_reward={q:.5f}, count={cnt}")

    def _exploit(self) -> int:
        max_q = max(self.q_values)
        return random.choice([i for i, q in enumerate(self.q_values) if q == max_q])


class EpsilonGreedyBandit(Bandit):
    def __init__(self, arms: list, epsilon: float = 0.1) -> None:
        super().__init__(arms)
        self.epsilon = epsilon

    def select_arm(self) -> int:
        if random.random() < self.epsilon:
            return random.randint(0, len(self.arms) - 1)
        return self._exploit()


class EpsilonFirstBandit(Bandit):
    def __init__(self, arms: list, exploration_steps: int = 100, epsilon: float = 0.1) -> None:
        super().__init__(arms)
        self.exploration_steps = exploration_steps
        self.epsilon = epsilon
        self.step = 0

    def select_arm(self) -> int:
        if self.step < self.exploration_steps or random.random() < self.epsilon:
            return random.randint(0, len(self.arms) - 1)
        self.step += 1
        return self._exploit()


class EpsilonDecreasingBandit(Bandit):
    def __init__(
        self,
        arms: list,
        initial_epsilon: float = 1.0,
        limit_epsilon: float = 0.1,
        half_decay_steps: int = 100,
    ) -> None:
        super().__init__(arms)
        self.epsilon = initial_epsilon
        self.initial_epsilon = initial_epsilon
        self.limit_epsilon = limit_epsilon
        self.half_decay_steps = half_decay_steps
        self.step = 0

    def _update_epsilon(self) -> None:
        decay = 0.5 ** (self.step / self.half_decay_steps)
        self.epsilon = self.limit_epsilon + (self.initial_epsilon - self.limit_epsilon) * decay

    def select_arm(self) -> int:
        self.step += 1
        self._update_epsilon()
        if random.random() < self.epsilon:
            return random.randint(0, len(self.arms) - 1)
        return self._exploit()


class UCB1Bandit(Bandit):
    def __init__(self, arms: list) -> None:
        super().__init__(arms)
        self.total_count = 0

    def select_arm(self) -> int:
        for i, count in enumerate(self.counts):
            if count == 0:
                return i
        ucb_values = [
            self.q_values[i] + math.sqrt(2 * math.log(self.total_count) / self.counts[i])
            for i in range(len(self.arms))
        ]
        return ucb_values.index(max(ucb_values))

    def update(self, arm_index: int, reward: float) -> None:
        if not (0 <= reward <= 1):
            raise ValueError("Reward must be in the range [0, 1].")
        self.total_count += 1
        super().update(arm_index, reward)


class GreedyBanditWithHistory(Bandit):
    def __init__(self, arms: list, history_length: int = 100) -> None:
        super().__init__(arms)
        self.history_length = history_length
        self.history: list[list[float]] = [[] for _ in range(len(arms))]

    def select_arm(self) -> int:
        incomplete = [i for i, h in enumerate(self.history) if len(h) < self.history_length]
        if incomplete:
            return random.choice(incomplete)
        return self._exploit()

    def update(self, arm_index: int, reward: float) -> None:
        h = self.history[arm_index]
        if len(h) >= self.history_length:
            h.pop(0)
        h.append(reward)
        self.counts[arm_index] = len(h)
        self.q_values[arm_index] = sum(h) / len(h)
