"""Fixed-weight evaluation; no policy receives environment internals."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
from scipy.sparse import csr_matrix


class Network:
    def __init__(self, path):
        with np.load(path, allow_pickle=False) as w:
            self.w1 = csr_matrix(w["w1"])
            self.b1 = w["b1"][:, None]
            self.w2 = csr_matrix(w["w2"])
            self.b2 = w["b2"][:, None]
            self.actions = w["actions"]

    def __call__(self, observations):
        x = np.asarray(observations, dtype=np.float32).reshape(-1, 400).T
        h = np.maximum(self.w1 @ x + self.b1, 0)
        logits = self.w2 @ h + self.b2
        return self.actions[np.argmax(logits, axis=0)]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ci(values):
    rng = np.random.default_rng(20261104)
    samples = values[rng.integers(len(values), size=(2000, len(values)))].mean(axis=1)
    return np.quantile(samples, [0.025, 0.975]).tolist()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--minatar", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--split", choices=["development", "heldout", "stress"], required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    sys.path.insert(0, str(args.minatar.resolve()))
    from minatar import Environment
    cfg = json.loads(args.plan.read_text())[args.split]
    args.output.mkdir(parents=True, exist_ok=False)
    before = digest(args.model)
    net = Network(args.model)
    envs = []
    for seed in range(cfg["first_seed"], cfg["first_seed"] + cfg["episodes"]):
        env = Environment("breakout", sticky_action_prob=cfg["sticky"])
        env.seed(seed)
        env.reset()
        envs.append(env)
    active = np.arange(len(envs))
    rewards = np.zeros(len(envs), dtype=np.int64)
    steps = np.zeros(len(envs), dtype=np.int64)
    terminated = np.zeros(len(envs), dtype=bool)
    replay = []
    started = time.perf_counter()
    for step in range(1, cfg["cap"] + 1):
        states = np.stack([envs[int(i)].state() for i in active])
        actions = net(states)
        keep = []
        for j, i in enumerate(active):
            if i == 0:
                state = states[j]
                replay.append({"p": int(np.flatnonzero(state[9, :, 0])[0]), "b": np.flatnonzero(state[:, :, 1]).tolist(), "t": np.flatnonzero(state[:, :, 2]).tolist(), "k": np.flatnonzero(state[:, :, 3]).tolist(), "s": int(rewards[i]), "a": int(actions[j])})
            reward, terminal = envs[int(i)].act(int(actions[j]))
            rewards[i] += reward
            steps[i] = step
            terminated[i] = terminal
            if not terminal:
                keep.append(int(i))
        active = np.array(keep, dtype=int)
        if step % 1000 == 0:
            print(f"step={step} active={len(active)} mean_reward={rewards.mean():.3f}", flush=True)
        if not len(active):
            break
    state = envs[0].state()
    replay.append({"p": int(np.flatnonzero(state[9, :, 0])[0]), "b": np.flatnonzero(state[:, :, 1]).tolist(), "t": np.flatnonzero(state[:, :, 2]).tolist(), "k": np.flatnonzero(state[:, :, 3]).tolist(), "s": int(rewards[0]), "a": 0})
    assert before == digest(args.model)
    summary = {"model": args.model.name, "sha256": before, "split": args.split, "config": cfg, "mean": float(rewards.mean()), "median": float(np.median(rewards)), "ci95": ci(rewards), "min": int(rewards.min()), "max": int(rewards.max()), "mean_steps": float(steps.mean()), "truncated": int((~terminated).sum()), "elapsed_seconds": time.perf_counter() - started}
    with (args.output / "episodes.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["seed", "reward", "steps", "terminated"])
        for i in range(len(envs)):
            writer.writerow([cfg["first_seed"] + i, rewards[i], steps[i], bool(terminated[i])])
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.output / "replay.json").write_text(json.dumps({"frames": replay, "episode": {"seed": cfg["first_seed"], "reward": int(rewards[0]), "steps": int(steps[0]), "terminated": bool(terminated[0])}}, separators=(",", ":")), encoding="utf-8")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
