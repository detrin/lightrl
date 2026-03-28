import random

from lightrl import GreedyBanditWithHistory

ENDPOINTS = {
    "us-east-1": {"base_latency": 20, "variance": 5},
    "eu-west-1": {"base_latency": 80, "variance": 15},
    "ap-south-1": {"base_latency": 150, "variance": 30},
    "us-west-2": {"base_latency": 25, "variance": 8},
}


def simulate_latency(endpoint_idx):
    cfg = list(ENDPOINTS.values())[endpoint_idx]
    latency = cfg["base_latency"] + random.gauss(0, cfg["variance"])
    return max(0, 1 - latency / 200)


if __name__ == "__main__":
    bandit = GreedyBanditWithHistory(arms=list(ENDPOINTS.keys()), history_length=30)

    for step in range(500):
        arm = bandit.select_arm()
        reward = simulate_latency(arm)
        bandit.update(arm, reward)

    bandit.report()
    best = bandit.arms[bandit.q_values.index(max(bandit.q_values))]
    print(f"\nBest endpoint: {best}")
