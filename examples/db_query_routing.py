import random

from lightrl import EpsilonGreedyBandit

STRATEGIES = ["seq_scan", "index_scan", "hash_join", "merge_join"]
TABLE_SIZES = ["small", "medium", "large"]

LATENCY = {
    "small": [2, 8, 15, 12],
    "medium": [300, 15, 35, 20],
    "large": [5000, 40, 100, 30],
}


def simulate_query_latency(strategy_idx, table_size):
    base_ms = LATENCY[table_size][strategy_idx]
    actual_ms = max(1, base_ms + random.gauss(0, base_ms * 0.1))
    return max(0.0, 1.0 / (1.0 + actual_ms / 10.0))


if __name__ == "__main__":
    bandits = {
        size: EpsilonGreedyBandit(arms=STRATEGIES, epsilon=0.1, ema_alpha=0.1)
        for size in TABLE_SIZES
    }

    for _ in range(1000):
        size = random.choice(TABLE_SIZES)
        arm = bandits[size].select_arm()
        reward = simulate_query_latency(arm, size)
        bandits[size].update(arm, reward)

    for size in TABLE_SIZES:
        print(f"\n[{size} tables]")
        bandits[size].report()
        best_idx = bandits[size].q_values.index(max(bandits[size].q_values))
        print(f"Best strategy: {STRATEGIES[best_idx]}")

    print("\nExpected: fast scan for small, index-based for medium/large")
    print("EMA decay adapts if workload patterns shift over time")
