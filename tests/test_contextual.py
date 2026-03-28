from lightrl import LinUCBBandit


class TestLinUCBBandit:
    def setup_method(self):
        self.bandit = LinUCBBandit(n_arms=3, n_features=2, alpha=1.0)

    def test_initialization(self):
        assert self.bandit.n_arms == 3
        assert self.bandit.n_features == 2
        assert len(self.bandit.A) == 3
        assert len(self.bandit.b) == 3
        assert self.bandit.counts == [0, 0, 0]

    def test_select_arm_range(self):
        for _ in range(50):
            arm = self.bandit.select_arm([1.0, 0.0])
            assert 0 <= arm < 3

    def test_update_modifies_state(self):
        self.bandit.update(0, [1.0, 0.5], 1.0)
        assert self.bandit.counts[0] == 1
        assert self.bandit.A[0] != [[1, 0], [0, 1]]

    def test_learns_context_dependent_arm(self):
        bandit = LinUCBBandit(n_arms=2, n_features=2, alpha=0.5)
        for _ in range(200):
            bandit.update(0, [1.0, 0.0], 0.9)
            bandit.update(0, [0.0, 1.0], 0.1)
            bandit.update(1, [1.0, 0.0], 0.1)
            bandit.update(1, [0.0, 1.0], 0.9)

        assert bandit.select_arm([1.0, 0.0]) == 0
        assert bandit.select_arm([0.0, 1.0]) == 1

    def test_save_load(self, tmp_path):
        path = tmp_path / "linucb.json"
        self.bandit.update(0, [1.0, 0.0], 0.5)
        self.bandit.save(path)

        loaded = LinUCBBandit.load(path)
        assert loaded.n_arms == 3
        assert loaded.n_features == 2
        assert loaded.counts == self.bandit.counts
        assert loaded.A == self.bandit.A
