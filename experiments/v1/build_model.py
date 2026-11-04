"""Construct every weight from game geometry, without rollouts or training."""

from pathlib import Path
import hashlib
import json
import numpy as np

HERE = Path(__file__).resolve().parent
ACTIONS = np.array([0, 1, 3], dtype=np.int64)


def index(y, x, channel):
    return (y * 10 + x) * 4 + channel


def reflected_x(x, direction, steps):
    t = (x + direction * steps) % 20
    return t if t < 10 else 19 - t


def target_for(y, x, previous_y, previous_x, anticipatory):
    if not anticipatory:
        return x
    dx = x - previous_x
    dy = y - previous_y
    if dx == 0:
        dx = 1 if x < 5 else -1
    # A brick impact leaves y unchanged; a paddle impact does so at row 8.
    descending = dy > 0 or (dy == 0 and y <= 3)
    return reflected_x(x, dx, max(0, 8 - y)) if descending else x


def build(anticipatory=True):
    pairs = [
        (y, x, py, px)
        for y in range(10) for x in range(10)
        for py in range(max(0, y - 1), min(10, y + 2))
        for px in range(max(0, x - 1), min(10, x + 2))
    ]
    width = len(pairs) + 10
    w1 = np.zeros((width, 400), dtype=np.float32)
    b1 = np.zeros(width, dtype=np.float32)
    w2 = np.zeros((3, width), dtype=np.float32)
    b2 = np.array([0.25, 0, 0], dtype=np.float32)
    for unit, (y, x, py, px) in enumerate(pairs):
        w1[unit, index(y, x, 1)] = 1
        w1[unit, index(py, px, 2)] = 1
        b1[unit] = -1
        target = target_for(y, x, py, px, anticipatory)
        w2[1, unit] = -target
        w2[2, unit] = target
    for x in range(10):
        unit = len(pairs) + x
        w1[unit, index(9, x, 0)] = 1
        w2[1, unit] = x
        w2[2, unit] = -x
    return {"w1": w1, "b1": b1, "w2": w2, "b2": b2, "actions": ACTIONS}


def main():
    metadata = {"training_steps": 0, "architecture": [400, 794, 3], "dtype": "float32"}
    for name, anticipatory in [("anticipation", True), ("reactive", False)]:
        weights = build(anticipatory)
        path = HERE / f"{name}.npz"
        np.savez_compressed(path, **weights)
        values = [weights[key] for key in ("w1", "b1", "w2", "b2")]
        metadata[name] = {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "dense_parameter_slots": sum(a.size for a in values),
            "nonzero_parameters": sum(int(np.count_nonzero(a)) for a in values),
            "nonzero_connections": int(np.count_nonzero(weights['w1']) + np.count_nonzero(weights['w2'])),
        }
    metadata["builder_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (HERE / "model.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
