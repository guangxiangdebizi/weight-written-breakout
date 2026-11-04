"""Evaluate frozen neural weights in the unmodified official MinAtar environment."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
from policy import NeuralPolicy

HERE = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def encode_frame(state, score, action):
    return {
        "p": int(np.flatnonzero(state[9, :, 0])[0]),
        "b": np.flatnonzero(state[:, :, 1]).tolist(),
        "t": np.flatnonzero(state[:, :, 2]).tolist(),
        "k": np.flatnonzero(state[:, :, 3]).tolist(),
        "s": int(score), "a": int(action),
    }


def bootstrap(values):
    rng = np.random.default_rng(20261104)
    samples = values[rng.integers(0, len(values), size=(2000, len(values)))].mean(axis=1)
    return [float(v) for v in np.quantile(samples, [0.025, 0.975])]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--minatar", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--result", type=Path, default=HERE / "results")
    args = parser.parse_args()
    sys.path.insert(0, str(args.minatar.resolve()))
    from minatar import Environment
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    result = args.result
    result.mkdir(parents=True, exist_ok=False)
    paths = [HERE / "anticipation.npz", HERE / "reactive.npz"]
    before = {path.name: digest(path) for path in paths}
    metadata = json.loads((HERE / "model.json").read_text())
    for path in paths:
        assert digest(path) == metadata[path.stem]["sha256"]
    policies = {
        "anticipation": NeuralPolicy(paths[0]),
        "reactive": NeuralPolicy(paths[1]),
        "zero": NeuralPolicy(paths[0], zero=True),
    }
    summary = {"plan": plan, "weights_before": before, "conditions": {}}
    rows = []
    replays = {}
    started = time.perf_counter()
    csv_path = result / "episodes.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["condition", "policy", "seed", "reward", "steps", "terminated", "truncated"])
        writer.writeheader()
        for condition in ["primary", "stress"]:
            config = plan[condition]
            summary["conditions"][condition] = {}
            for name in plan["policies"]:
                group = []
                for seed in range(config["seeds"][0], config["seeds"][1] + 1):
                    env = Environment("breakout", sticky_action_prob=config["sticky_action_prob"])
                    env.seed(seed)
                    env.reset()
                    rng = np.random.default_rng(seed + 1000000)
                    reward = 0
                    frames = []
                    record = condition == "primary" and seed == 0
                    for step in range(1, config["max_steps"] + 1):
                        state = env.state()
                        action = int(rng.choice([0, 1, 3])) if name == "random" else policies[name](state)
                        if record and len(frames) < 600:
                            frames.append(encode_frame(state, reward, action))
                        r, terminal = env.act(action)
                        reward += r
                        if terminal:
                            break
                    row = {"condition": condition, "policy": name, "seed": seed, "reward": int(reward), "steps": step, "terminated": bool(terminal), "truncated": not bool(terminal)}
                    writer.writerow(row)
                    group.append(row)
                    rows.append(row)
                    if record:
                        if len(frames) < 600:
                            frames.append(encode_frame(env.state(), reward, 0))
                        replays[name] = {"frames": frames, "episode": row, "clip_complete": len(frames) > step}
                    if (seed - config["seeds"][0] + 1) % 100 == 0:
                        handle.flush()
                        print(f"{condition} {name}: {seed - config['seeds'][0] + 1}/{config['episodes']}", flush=True)
                values = np.array([r["reward"] for r in group], dtype=float)
                stats = {"episodes": len(group), "mean": float(values.mean()), "median": float(np.median(values)), "ci95": bootstrap(values), "min": float(values.min()), "max": float(values.max()), "mean_steps": float(np.mean([r["steps"] for r in group])), "truncated": sum(r["truncated"] for r in group)}
                summary["conditions"][condition][name] = stats
                print(json.dumps({"condition": condition, "policy": name, **stats}), flush=True)
            for baseline in ["reactive", "random", "zero"]:
                ours = np.array([r["reward"] for r in rows if r["condition"] == condition and r["policy"] == "anticipation"])
                other = np.array([r["reward"] for r in rows if r["condition"] == condition and r["policy"] == baseline])
                difference = ours - other
                summary["conditions"][condition][f"paired_vs_{baseline}"] = {"mean_difference": float(difference.mean()), "ci95": bootstrap(difference)}
    after = {path.name: digest(path) for path in paths}
    assert after == before
    summary["weights_after"] = after
    summary["elapsed_seconds"] = time.perf_counter() - started
    summary["numpy_version"] = np.__version__
    (result / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (result / "replays.json").write_text(json.dumps(replays, separators=(",", ":")), encoding="utf-8")
    print("FINISHED", summary["elapsed_seconds"], flush=True)


if __name__ == "__main__":
    main()
