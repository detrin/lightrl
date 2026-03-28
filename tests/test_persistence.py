import json

from lightrl import (
    BanditRouter,
    EpsilonDecreasingBandit,
    EpsilonGreedyBandit,
    ThompsonBandit,
    UCB1Bandit,
)


class TestBanditSaveLoad:
    def test_save_load_epsilon_greedy(self, tmp_path):
        path = tmp_path / "bandit.json"
        b = EpsilonGreedyBandit(arms=[1, 2, 3], epsilon=0.2)
        b.update(0, 1.0)
        b.update(1, 0.5)
        b.save(path)

        loaded = EpsilonGreedyBandit.load(path)
        assert loaded.arms == [1, 2, 3]
        assert loaded.q_values == b.q_values
        assert loaded.counts == b.counts
        assert loaded.epsilon == 0.2

    def test_save_load_thompson(self, tmp_path):
        path = tmp_path / "thompson.json"
        b = ThompsonBandit(arms=["x", "y"])
        b.update(0, 1.0)
        b.update(1, 0.0)
        b.save(path)

        loaded = ThompsonBandit.load(path)
        assert loaded.alpha == b.alpha
        assert loaded.beta == b.beta

    def test_save_load_ucb1(self, tmp_path):
        path = tmp_path / "ucb.json"
        b = UCB1Bandit(arms=[10, 20])
        b.update(0, 0.5)
        b.save(path)

        loaded = UCB1Bandit.load(path)
        assert loaded.total_count == 1
        assert loaded.q_values == b.q_values

    def test_load_preserves_class(self, tmp_path):
        path = tmp_path / "bandit.json"
        b = EpsilonDecreasingBandit(arms=[1, 2], half_decay_steps=50)
        b.save(path)

        loaded = EpsilonDecreasingBandit.load(path)
        assert isinstance(loaded, EpsilonDecreasingBandit)
        assert loaded.half_decay_steps == 50

    def test_roundtrip_file_content(self, tmp_path):
        path = tmp_path / "bandit.json"
        b = EpsilonGreedyBandit(arms=["a", "b"])
        b.save(path)
        data = json.loads(path.read_text())
        assert data["class"] == "EpsilonGreedyBandit"
        assert "state" in data


class TestRouterSaveLoad:
    def test_save_load_router(self, tmp_path):
        path = tmp_path / "router.json"
        router = BanditRouter(epsilon=0.15)
        router.select("model", arms=["haiku", "sonnet", "opus"])
        router.update("model", 0, 0.8)
        router.select("batch", arms=[10, 50, 100])
        router.update("batch", 1, 0.5)
        router.save(path)

        loaded = BanditRouter.load(path)
        assert "model" in loaded._bandits
        assert "batch" in loaded._bandits
        assert loaded._bandits["model"].arms == ["haiku", "sonnet", "opus"]

    def test_router_with_mixed_bandits(self, tmp_path):
        path = tmp_path / "router.json"
        router = BanditRouter()
        router.register("clicks", ThompsonBandit(arms=["ad1", "ad2"]))
        router.register("model", UCB1Bandit(arms=["a", "b"]))
        router.save(path)

        loaded = BanditRouter.load(path)
        assert isinstance(loaded._bandits["clicks"], ThompsonBandit)
        assert isinstance(loaded._bandits["model"], UCB1Bandit)
