"""Two analytic, same-architecture edits to an existing untrained network."""

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def target_for(y, x, py, px, mode):
    # Below the brick rows, a row-4 stationary-y impact means downward motion.
    downward = y >= 4 and (y > py or (y == py and y < 8))
    if not downward:
        return 4.5
    dx = x - px
    if dx == 0:
        dx = 1 if x < 5 else -1
    def fold(t):
        t %= 20
        return t if t < 10 else 19 - t
    old = fold(x + dx * max(0, 8 - y))
    new = fold(x + dx * max(0, 9 - y))
    return old if mode == "center_old" else (old + new) / 2


def edit(original, mode):
    weights = {k: v.copy() for k, v in original.items()}
    unit = 0
    for y in range(10):
        for x in range(10):
            for py in range(max(0, y - 1), min(10, y + 2)):
                for px in range(max(0, x - 1), min(10, x + 2)):
                    target = target_for(y, x, py, px, mode)
                    weights["w2"][1, unit] = -target
                    weights["w2"][2, unit] = target
                    unit += 1
    weights["b2"][0] = 0.75
    return weights


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    with np.load(args.baseline, allow_pickle=False) as archive:
        original = dict(archive)
    metadata = {"architecture": [400, 794, 3], "baseline_sha256": sha(args.baseline), "edits": {}}
    for mode in ["center_old", "center_interval"]:
        weights = edit(original, mode)
        path = root / f"{mode}.npz"
        np.savez_compressed(path, **weights)
        changed = {k: int(np.count_nonzero(weights[k] != original[k])) for k in original}
        assert changed["w1"] == changed["b1"] == changed["actions"] == 0
        metadata["edits"][mode] = {"sha256": sha(path), "changed": changed, "nonzero_parameters": sum(int(np.count_nonzero(weights[k])) for k in ["w1", "b1", "w2", "b2"])}
    (root / "edits.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
