import random

from lightrl import BanditRouter, EpsilonGreedyBandit

TEMPLATES = {
    "concise": {"quality_mean": 0.6, "quality_std": 0.1},
    "detailed": {"quality_mean": 0.75, "quality_std": 0.15},
    "chain_of_thought": {"quality_mean": 0.85, "quality_std": 0.1},
    "few_shot": {"quality_mean": 0.8, "quality_std": 0.12},
    "zero_shot": {"quality_mean": 0.55, "quality_std": 0.2},
}

TASK_CONFIGS = {
    "code_gen": {"concise": 0.5, "chain_of_thought": 0.9, "few_shot": 0.85},
    "summarize": {"concise": 0.8, "detailed": 0.6, "chain_of_thought": 0.65},
    "qa": {"few_shot": 0.9, "chain_of_thought": 0.85, "zero_shot": 0.7},
}


def simulate_quality(template_name, task_type=None):
    if task_type and template_name in TASK_CONFIGS.get(task_type, {}):
        mean = TASK_CONFIGS[task_type][template_name]
    else:
        cfg = TEMPLATES[template_name]
        mean = cfg["quality_mean"]
    noise = random.gauss(0, 0.1)
    return max(0.0, min(1.0, mean + noise))


if __name__ == "__main__":
    router = BanditRouter()
    template_names = list(TEMPLATES.keys())

    for task_type in TASK_CONFIGS:
        router.register(task_type, EpsilonGreedyBandit(arms=template_names, epsilon=0.15))

    for _ in range(1000):
        for task_type in TASK_CONFIGS:
            idx = router.select(task_type)
            template = template_names[idx]
            reward = simulate_quality(template, task_type)
            router.update(task_type, idx, reward)

    for task_type in TASK_CONFIGS:
        bandit = router._bandits[task_type]
        best = bandit.arms[bandit.q_values.index(max(bandit.q_values))]
        print(f"{task_type}: best template = {best}")

    router.report()
