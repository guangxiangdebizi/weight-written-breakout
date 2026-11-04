"""Exhaustively verify legal local observations, without running a game."""

from pathlib import Path
import json
import numpy as np
from build_model import target_for
from policy import NeuralPolicy

HERE = Path(__file__).resolve().parent


def main():
    results = {}
    for name, anticipatory in [("anticipation", True), ("reactive", False)]:
        net = NeuralPolicy(HERE / f"{name}.npz")
        count = 0
        for y in range(10):
            for x in range(10):
                for py in range(max(0, y - 1), min(10, y + 2)):
                    for px in range(max(0, x - 1), min(10, x + 2)):
                        target = target_for(y, x, py, px, anticipatory)
                        for paddle in range(10):
                            state = np.zeros((10, 10, 4), dtype=np.float32)
                            state[y, x, 1] = state[py, px, 2] = state[9, paddle, 0] = 1
                            expected = 1 if paddle > target else 3 if paddle < target else 0
                            assert net(state) == expected, (name, y, x, py, px, paddle)
                            count += 1
        with np.load(HERE / f"{name}.npz") as w:
            batch = np.random.default_rng(17).integers(0, 2, size=(32, 400)).astype(np.float32)
            dense = np.maximum(batch @ w["w1"].T + w["b1"], 0) @ w["w2"].T + w["b2"]
            sparse = np.stack([net.logits(x) for x in batch])
            np.testing.assert_array_equal(dense, sparse)
            assert not np.count_nonzero(w["w1"][:, 3::4])
        results[name] = {"local_observations_checked": count, "sparse_dense_equal": True, "brick_weights_zero": True}
    # Independent wall-step simulation, not used in the neural forward path.
    from build_model import reflected_x
    for x in range(10):
        for direction in [-1, 1]:
            for steps in range(21):
                p, d = x, direction
                for _ in range(steps):
                    p += d
                    if p < 0:
                        p, d = 0, 1
                    elif p > 9:
                        p, d = 9, -1
                assert reflected_x(x, direction, steps) == p
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
