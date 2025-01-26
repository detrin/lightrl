import random
import math
from abc import ABC, abstractmethod
from typing import List, Any


class Bandit(ABC):
    def __init__(self, arms: List[Any]) -> None:
        """
        Initialize a Bandit with a specified number of arms.

        Args:
            arms (List[Any]): A list representing different arms or tasks
                              that the Bandit can choose from.
        """
        self.arms: List[Any] = arms
        self.q_values: List[float] = [0.0] * len(arms)  # Estimated rewards for each arm
        self.counts: List[int] = [0] * len(
            arms
        )  # Number of times each arm has been selected

    @abstractmethod
    def select_arm(self) -> int:
        """
        Abstract method to select the next arm to be used.

        Returns:
            int: The index of the selected arm.
        """
        pass

    def update(self, arm_index: int, reward: float) -> None:
        """
        Update the value estimates for a given arm based on the reward received.

        Args:
            arm_index (int): Index of the arm that was selected.
            reward (float): Reward received after selecting the arm.
        """
        self.counts[arm_index] += 1
        n = self.counts[arm_index]
        old_q = self.q_values[arm_index]
        self.q_values[arm_index] = ((n - 1) * old_q + reward) / n

    def __repr__(self) -> str:
        """
        String representation of the Bandit object.

        Returns:
            str: String representation of the Bandit, showing its arms.
        """
        return f"{self.__class__.__name__}(arms={self.arms})"

    def report(self) -> None:
        """
        Print a report of the average rewards (Q-values) and selection counts for each arm.
        """
        print("Q-values per arm:")
        for arm, q, cnt in zip(self.arms, self.q_values, self.counts):
            print(f"  num_tasks={arm}: avg_reward={q:.5f}, count={cnt}")


class EpsilonGreedyBandit(Bandit):
    def __init__(self, arms: List[Any], epsilon: float = 0.1) -> None:
        """
        Initialize an EpsilonGreedyBandit with a specified number of arms and an exploration probability.

        Args:
            arms (List[Any]): A list representing different arms or tasks that the Bandit can choose from.
            epsilon (float, optional): The probability of choosing a random arm for exploration.
                                       Defaults to 0.1.
        """
        super().__init__(arms)
        self.epsilon: float = epsilon

    def select_arm(self) -> int:
        """
        Select an arm to use based on the epsilon-greedy strategy.

        This method uses exploration with probability 'epsilon' and exploitation otherwise,
        selecting the arm with the highest estimated value.

        Returns:
            int: The index of the selected arm.
        """
        if random.random() < self.epsilon:
            # Explore: select a random arm
            return random.randint(0, len(self.arms) - 1)

        # Exploit: select the arm with maximum estimated value
        max_q = max(self.q_values)
        candidates = [i for i, q in enumerate(self.q_values) if q == max_q]
        return random.choice(candidates)


class EpsilonFirstBandit(Bandit):
    def __init__(
        self, arms: List[Any], exploration_steps: int = 100, epsilon: float = 0.1
    ) -> None:
        """
        Initialize an EpsilonFirstBandit with a specified number of arms, exploration steps, and exploration probability.

        Args:
            arms (List[Any]): A list representing different arms or tasks that the Bandit can choose from.
            exploration_steps (int, optional): The number of initial steps to purely explore. Defaults to 100.
            epsilon (float, optional): The probability of choosing a random arm during the exploration phase.
                                       Defaults to 0.1.
        """
        super().__init__(arms)
        self.exploration_steps: int = exploration_steps
        self.epsilon: float = epsilon
        self.step: int = 0

    def select_arm(self) -> int:
        """
        Select an arm to use based on the epsilon-first strategy.

        This method uses pure exploration for a defined number of initial steps and then follows
        an epsilon-greedy strategy thereafter.

        Returns:
            int: The index of the selected arm.
        """
        if self.step < self.exploration_steps or random.random() < self.epsilon:
            # Explore: select a random arm either during the exploration phase or if chosen randomly
            return random.randint(0, len(self.arms) - 1)

        # Exploit: select the arm with maximum estimated value
        max_q = max(self.q_values)
        candidates = [i for i, q in enumerate(self.q_values) if q == max_q]

        self.step += 1
        return random.choice(candidates)


class EpsilonDecreasingBandit(Bandit):
    def __init__(
        self,
        arms: List[Any],
        initial_epsilon: float = 1.0,
        limit_epsilon: float = 0.1,
        half_decay_steps: int = 100,
    ) -> None:
        """
        Initialize an EpsilonDecreasingBandit with a specified number of arms and epsilon parameters.

        Args:
            arms (List[Any]): A list representing different arms or tasks that the Bandit can choose from.
            initial_epsilon (float, optional): The initial exploration probability. Defaults to 1.0.
            limit_epsilon (float, optional): The minimum limit for the exploration probability. Defaults to 0.1.
            half_decay_steps (int, optional): The number of steps at which the exploration probability is reduced
                                              to half of the difference between `initial_epsilon` and `limit_epsilon`.
                                              Defaults to 100.
        """
        super().__init__(arms)
        self.epsilon: float = initial_epsilon
        self.initial_epsilon: float = initial_epsilon
        self.limit_epsilon: float = limit_epsilon
        self.half_decay_steps: int = half_decay_steps
        self.step: int = 0

    def select_arm(self) -> int:
        """
        Select an arm to use based on the epsilon-decreasing strategy.

        This method adjusts the exploration probability over time and selects an arm accordingly.

        Returns:
            int: The index of the selected arm.
        """
        self.step += 1
        self.update_epsilon()

        if random.random() < self.epsilon:
            # Explore: select a random arm
            return random.randint(0, len(self.arms) - 1)
        print(self.q_values)
        # Exploit: select the arm with maximum estimated value
        max_q = max(self.q_values)
        candidates = [i for i, q in enumerate(self.q_values) if q == max_q]
        return random.choice(candidates)

    def update_epsilon(self) -> None:
        """
        Update the exploration probability `epsilon` based on the current step.

        The exploration probability decays towards the limit probability over time, according to a half-life decay model.
        """
        self.epsilon = self.limit_epsilon + (
            self.initial_epsilon - self.limit_epsilon
        ) * (0.5 ** (self.step / self.half_decay_steps))


class UCB1Bandit(Bandit):
    def __init__(self, arms: List[Any]) -> None:
        """
        Initialize a UCB1Bandit with a specified number of arms.

        Args:
            arms (List[Any]): A list representing different arms or tasks that the Bandit can choose from.
        """
        super().__init__(arms)
        self.total_count: int = 0  # Total number of times any arm has been selected

    def select_arm(self) -> int:
        """
        Select an arm to use based on the Upper Confidence Bound (UCB1) strategy.

        This method selects an arm that maximizes the UCB estimate, accounting for exploration and exploitation.

        Returns:
            int: The index of the selected arm.
        """
        for arm_index, count in enumerate(self.counts):
            if count == 0:
                # If an arm has not been selected yet, select it
                return arm_index

        # Calculate UCB values for each arm and choose the arm with the highest UCB value
        ucb_values = [
            self.q_values[i]
            + math.sqrt((2 * math.log(self.total_count)) / self.counts[i])
            for i in range(len(self.arms))
        ]
        return ucb_values.index(max(ucb_values))

    def update(self, arm_index: int, reward: float) -> None:
        """
        Update the value estimates for a given arm based on the reward received and increment the total count.

        Args:
            arm_index (int): Index of the arm that was selected.
            reward (float): Reward received after selecting the arm. Must be in the range [0, 1].

        Raises:
            ValueError: If the reward is not within the range [0, 1].
        """
        if not (0 <= reward <= 1):
            raise ValueError("Reward must be in the range [0, 1].")
        self.total_count += 1
        super().update(arm_index, reward)


class GreedyBanditWithHistory(Bandit):
    def __init__(self, arms: List[Any], history_length: int = 100) -> None:
        """
        Initialize a GreedyBanditWithHistory with a specified number of arms and history length.

        Args:
            arms (List[Any]): A list representing different arms or tasks that the Bandit can choose from.
            history_length (int, optional): The maximum length of history to maintain for each arm's rewards.
                                            Defaults to 100.
        """
        super().__init__(arms)
        self.history_length: int = history_length
        self.history: List[List[float]] = [
            [] for _ in range(len(arms))
        ]  # History of rewards for each arm

    def select_arm(self) -> int:
        """
        Select an arm to use based on the greedy strategy with bounded history.

        This method ensures that each arm's history reaches the defined length before purely exploiting.

        Returns:
            int: The index of the selected arm.
        """
        if any(len(history) < self.history_length for history in self.history):
            # If any arm has not reached the history length, select one of these arms for exploration
            candidates = [
                i
                for i, history in enumerate(self.history)
                if len(history) < self.history_length
            ]
            return random.choice(candidates)

        # Once history length is reached for all arms, exploit the arm with maximum estimated value
        max_q = max(self.q_values)
        candidates = [i for i, q in enumerate(self.q_values) if q == max_q]
        return random.choice(candidates)

    def update(self, arm_index: int, reward: float) -> None:
        """
        Update the value estimates for a given arm based on the reward received and update its history.

        Args:
            arm_index (int): Index of the arm that was selected.
            reward (float): Reward received after selecting the arm.
        """
        if len(self.history[arm_index]) >= self.history_length:
            # Maintain bounded history by removing the oldest reward if limit is exceeded
            self.history[arm_index].pop(0)
        self.history[arm_index].append(reward)

        # Update the count and Q-value for the arm based on its history
        self.counts[arm_index] = len(self.history[arm_index])
        self.q_values[arm_index] = sum(self.history[arm_index]) / self.counts[arm_index]
