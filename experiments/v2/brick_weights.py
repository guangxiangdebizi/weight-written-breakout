"""Compile a brick-aware anticipatory controller into the SAME 400-794-3 MLP."""

import hashlib
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent


def index(y, x, channel):
    return (y * 10 + x) * 4 + channel


def fold(t):
    t %= 20
    return t if t < 10 else 19 - t


def build(mode):
    w1 = np.zeros((794, 400), dtype=np.float32)
    b1 = np.zeros(794, dtype=np.float32)
    w2 = np.zeros((3, 794), dtype=np.float32)
    b2 = np.array([0.75, 0, 0], dtype=np.float32)
    unit = 0

    def detector(required, forbidden, target):
        nonlocal unit
        assert unit < 784
        w1[unit, required] = 1
        w1[unit, forbidden] = -1
        b1[unit] = 1 - len(required)
        w2[1, unit], w2[2, unit] = -target, target
        unit += 1

    # A single linear unit handles all states within the brick field.
    w1[unit, [index(y, x, 1) for y in range(4) for x in range(10)]] = 1
    w2[1, unit], w2[2, unit] = -4.5, 4.5
    unit += 1

    def landing(x, dx, horizon):
        a = fold(x + dx * horizon)
        b = fold(x + dx * (horizon + 1))
        return a if mode == "brick_old" else (a + b) / 2

    def ascending(required, y, x, dx):
        earlier = []
        for row in [3, 2, 1]:
            n = y - row
            brick = index(row, fold(x + dx * n), 3)
            # The brick hit keeps the previous vertical coordinate, then reverses y.
            horizon = n + 8 - (row + 1)
            detector(required + [brick], earlier, landing(x, dx, horizon))
            earlier = earlier + [brick]
        detector(required, earlier, landing(x, dx, y + 9))

    for y in range(4, 9):
        for x in range(10):
            for px in range(max(0, x - 1), min(10, x + 2)):
                if px == x and x not in [0, 9]:
                    continue
                dx = x - px if x != px else 1 if x == 0 else -1
                detector([index(y, x, 1), index(y - 1, px, 2)], [], landing(x, dx, 8 - y))
                if y == 4:
                    detector([index(y, x, 1), index(y, px, 2)], [], landing(x, dx, 8 - y))
                if y < 8:
                    ascending([index(y, x, 1), index(y + 1, px, 2)], y, x, dx)
                else:
                    # At a paddle bounce, its current position identifies which
                    # collision branch ran, hence the outgoing horizontal direction.
                    pair = [index(y, x, 1), index(y, px, 2)]
                    ascending(pair + [index(9, px, 0)], y, x, dx)
                    if x != px:
                        ascending(pair + [index(9, x, 0)], y, x, -dx)
    for x in range(10):
        w1[784 + x, index(9, x, 0)] = 1
        w2[1, 784 + x] = x
        w2[2, 784 + x] = -x
    return {"w1": w1, "b1": b1, "w2": w2, "b2": b2, "actions": np.array([0, 1, 3])}, unit


def main():
    metadata = {"architecture": [400, 794, 3], "models": {}}
    for mode in ["brick_old", "brick_interval"]:
        weights, units = build(mode)
        path = HERE / f"{mode}.npz"
        np.savez_compressed(path, **weights)
        metadata["models"][mode] = {"active_hidden_units": units + 10, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "nonzero_parameters": sum(int(np.count_nonzero(weights[k])) for k in ["w1", "b1", "w2", "b2"])}
    (HERE / "brick-models.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
