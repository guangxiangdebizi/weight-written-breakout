"""Verify detector exclusivity and exact dense/batched inference agreement."""

from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix
from run_batch import Network

HERE = Path(__file__).resolve().parent
path = HERE / "extended_old.npz"
with np.load(path) as archive:
    w = dict(archive)
net = Network(path)
rng = np.random.default_rng(112233)
states = []
for y in range(9):
    for x in range(10):
        for py in range(max(0, y - 1), min(9, y + 2)):
            if py == y and y in [5, 6, 7]:
                continue
            for px in range(max(0, x - 1), min(10, x + 2)):
                if px == x and x not in [0, 9]:
                    continue
                for p in range(10):
                    state = np.zeros((10, 10, 4), dtype=np.float32)
                    state[y, x, 1] = state[py, px, 2] = state[9, p, 0] = 1
                    state[1:4, :, 3] = rng.integers(0, 2, size=(3, 10))
                    states.append(state)
batch = np.stack(states)
h = np.maximum(csr_matrix(w["w1"]) @ batch.reshape(-1, 400).T + w["b1"][:, None], 0)
np.testing.assert_array_equal(h[:784].sum(axis=0), np.ones(len(batch)))
np.testing.assert_array_equal(h[784:].sum(axis=0), np.ones(len(batch)))
sample = batch[::37]
dense = np.maximum(sample.reshape(-1, 400) @ w["w1"].T + w["b1"], 0) @ w["w2"].T + w["b2"]
np.testing.assert_array_equal(net(sample), w["actions"][np.argmax(dense, axis=1)])
single = np.concatenate([net(state[None]) for state in sample])
np.testing.assert_array_equal(net(sample), single)
print(f"PASS: {len(batch)} local observations, one active state detector each, dense/sparse/batch/single agreement")
