"""Full-flight brick anticipation, with optional last-step safe-catch interval."""

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


def build(safe):
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

    for y in [0, 1, 2, 3, 8]:
        # Ambiguous vertical reflections: one neutral frame, then infer anew.
        w1[unit, [index(y, x, c) for x in range(10) for c in [1, 2]]] = 1
        b1[unit] = -1
        w2[1, unit], w2[2, unit] = -4.5, 4.5
        unit += 1

    for y in range(9):
        for x in range(10):
            for px in range(max(0, x - 1), min(10, x + 2)):
                if px == x and x not in [0, 9]:
                    continue
                dx = x - px if x != px else 1 if x == 0 else -1
                if y >= 1:
                    target = fold(x + dx * (8 - y))
                    if safe and y == 8:
                        target = (target + fold(x + dx)) / 2
                    detector([index(y, x, 1), index(y - 1, px, 2)], [], target)
                if y == 4:
                    detector([index(y, x, 1), index(y, px, 2)], [], fold(x + dx * 4))
                if y <= 7:
                    pair = [index(y, x, 1), index(y + 1, px, 2)]
                    previous = []
                    for row in range(min(3, y - 1), 0, -1):
                        n = y - row
                        brick = index(row, fold(x + dx * n), 3)
                        target = fold(x + dx * (n + 8 - (row + 1)))
                        detector(pair + [brick], previous, target)
                        previous = previous + [brick]
                    detector(pair, previous, fold(x + dx * (y + 9)))
    for x in range(10):
        w1[784 + x, index(9, x, 0)] = 1
        w2[1, 784 + x] = x
        w2[2, 784 + x] = -x
    return {"w1": w1, "b1": b1, "w2": w2, "b2": b2, "actions": np.array([0, 1, 3])}, unit


def main():
    result = {}
    for name, safe in [("extended_old", False), ("extended_safe", True)]:
        w, units = build(safe)
        path = HERE / f"{name}.npz"
        np.savez_compressed(path, **w)
        result[name] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "active_hidden_units": units + 10, "nonzero_parameters": sum(int(np.count_nonzero(w[k])) for k in ["w1", "b1", "w2", "b2"])}
    (HERE / "extended-models.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
