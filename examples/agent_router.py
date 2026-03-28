import os
import random
import tempfile

from lightrl import BanditRouter, ThompsonBandit, UCB1Bandit

MODEL_QUALITY = {"haiku": 0.5, "sonnet": 0.75, "opus": 0.95}
MODEL_COST = {"haiku": 0.01, "sonnet": 0.1, "opus": 1.0}
BATCH_THROUGHPUT = {10: 0.3, 50: 0.7, 100: 0.9, 500: 0.5}


def simulate_model_task(model):
    quality = MODEL_QUALITY[model] + random.gauss(0, 0.05)
    cost = MODEL_COST[model]
    return max(0, min(1, quality - cost))


def simulate_batch_throughput(batch_size):
    base = BATCH_THROUGHPUT[batch_size]
    return max(0, min(1, base + random.gauss(0, 0.05)))


if __name__ == "__main__":
    path = os.path.join(tempfile.gettempdir(), "agent_router.json")

    if os.path.exists(path):
        router = BanditRouter.load(path)
        print(f"Resumed router from {path}")
    else:
        router = BanditRouter()
        router.register("model", ThompsonBandit(arms=list(MODEL_QUALITY.keys())))
        router.register("batch", UCB1Bandit(arms=list(BATCH_THROUGHPUT.keys())))
        print("Starting fresh router")

    for _ in range(300):
        model_idx = router.select("model")
        model = router._bandits["model"].arms[model_idx]
        router.update("model", model_idx, simulate_model_task(model))

        batch_idx = router.select("batch")
        batch = router._bandits["batch"].arms[batch_idx]
        router.update("batch", batch_idx, simulate_batch_throughput(batch))

    router.save(path)
    print(f"Saved to {path}")
    router.report()
