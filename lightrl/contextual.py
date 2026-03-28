import json
import math
from pathlib import Path


class LinUCBBandit:
    def __init__(self, n_arms: int, n_features: int, alpha: float = 1.0):
        self.n_arms = n_arms
        self.n_features = n_features
        self.alpha = alpha
        self.A = [_identity(n_features) for _ in range(n_arms)]
        self.b = [[0.0] * n_features for _ in range(n_arms)]
        self.counts = [0] * n_arms

    def select_arm(self, context: list[float]) -> int:
        best_arm, best_ucb = 0, float("-inf")
        for a in range(self.n_arms):
            A_inv = _invert(self.A[a])
            theta = _matvec(A_inv, self.b[a])
            ucb = _dot(theta, context) + self.alpha * math.sqrt(_quad(context, A_inv))
            if ucb > best_ucb:
                best_ucb = ucb
                best_arm = a
        return best_arm

    def update(self, arm_index: int, context: list[float], reward: float) -> None:
        self.counts[arm_index] += 1
        x = context
        for i in range(self.n_features):
            for j in range(self.n_features):
                self.A[arm_index][i][j] += x[i] * x[j]
            self.b[arm_index][i] += reward * x[i]

    def report(self) -> None:
        print("LinUCB arm counts:")
        for i, cnt in enumerate(self.counts):
            print(f"  arm {i}: count={cnt}")

    def save(self, path: str | Path) -> None:
        data = {
            "n_arms": self.n_arms,
            "n_features": self.n_features,
            "alpha": self.alpha,
            "A": self.A,
            "b": self.b,
            "counts": self.counts,
        }
        Path(path).write_text(json.dumps(data))

    @classmethod
    def load(cls, path: str | Path) -> "LinUCBBandit":
        data = json.loads(Path(path).read_text())
        obj = cls(data["n_arms"], data["n_features"], data["alpha"])
        obj.A = data["A"]
        obj.b = data["b"]
        obj.counts = data["counts"]
        return obj


def _identity(n):
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def _matvec(m, v):
    return [_dot(row, v) for row in m]


def _quad(v, m):
    return _dot(v, _matvec(m, v))


def _invert(matrix):
    n = len(matrix)
    aug = [row[:] + [1.0 if i == j else 0.0 for j in range(n)] for i, row in enumerate(matrix)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(aug[r][col]))
        aug[col], aug[pivot] = aug[pivot], aug[col]
        diag = aug[col][col]
        aug[col] = [x / diag for x in aug[col]]
        for row in range(n):
            if row != col:
                factor = aug[row][col]
                aug[row] = [aug[row][j] - factor * aug[col][j] for j in range(2 * n)]
    return [row[n:] for row in aug]
