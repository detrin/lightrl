from lightrl import ThompsonBandit


class TestThompsonBandit:
    def setup_method(self):
        self.bandit = ThompsonBandit(arms=["a", "b", "c"])

    def test_initialization(self):
        assert self.bandit.alpha == [1.0, 1.0, 1.0]
        assert self.bandit.beta == [1.0, 1.0, 1.0]

    def test_select_arm_range(self):
        for _ in range(100):
            assert 0 <= self.bandit.select_arm() < 3

    def test_update_success(self):
        self.bandit.update(0, 1.0)
        assert self.bandit.alpha[0] == 2.0
        assert self.bandit.beta[0] == 1.0

    def test_update_failure(self):
        self.bandit.update(0, 0.0)
        assert self.bandit.alpha[0] == 1.0
        assert self.bandit.beta[0] == 2.0

    def test_converges_to_best(self):
        for _ in range(500):
            self.bandit.update(0, 0.1)
            self.bandit.update(1, 0.9)
            self.bandit.update(2, 0.3)
        selections = [self.bandit.select_arm() for _ in range(100)]
        assert selections.count(1) > 80
