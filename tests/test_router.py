import pytest

from lightrl import BanditRouter, EpsilonGreedyBandit, ThompsonBandit


class TestBanditRouter:
    def test_auto_create_bandit(self):
        router = BanditRouter(epsilon=0.1)
        idx = router.select("test", arms=["a", "b", "c"])
        assert 0 <= idx < 3

    def test_select_arm_value(self):
        router = BanditRouter(epsilon=0.0)
        router.register("test", EpsilonGreedyBandit(arms=["x", "y"], priors=[0.1, 0.9], epsilon=0))
        assert router.select_arm_value("test") == "y"

    def test_register_custom_bandit(self):
        router = BanditRouter()
        router.register("clicks", ThompsonBandit(arms=["ad1", "ad2"]))
        idx = router.select("clicks")
        assert 0 <= idx < 2

    def test_update_and_exploit(self):
        router = BanditRouter(epsilon=0.0)
        router.select("test", arms=["a", "b", "c"])
        for _ in range(50):
            router.update("test", 2, 1.0)
            router.update("test", 0, 0.1)
            router.update("test", 1, 0.1)
        assert router.select("test") == 2

    def test_missing_bandit_no_arms_raises(self):
        router = BanditRouter()
        with pytest.raises(ValueError, match="not registered"):
            router.select("nonexistent")

    def test_multiple_independent_bandits(self):
        router = BanditRouter(epsilon=0.0)
        router.select("model", arms=["haiku", "opus"])
        router.select("batch", arms=[10, 100])
        router.update("model", 1, 1.0)
        router.update("batch", 0, 1.0)
        assert router.select("model") == 1
        assert router.select("batch") == 0

    def test_report(self, capsys):
        router = BanditRouter()
        router.select("test", arms=["a"])
        router.report("test")
        out = capsys.readouterr().out
        assert "[test]" in out

    def test_report_all(self, capsys):
        router = BanditRouter()
        router.select("x", arms=["a"])
        router.select("y", arms=["b"])
        router.report()
        out = capsys.readouterr().out
        assert "[x]" in out
        assert "[y]" in out
