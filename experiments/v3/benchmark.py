"""Matched official observation, softmax action sampling and sticky-action test."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
from scipy.sparse import csr_matrix


def observation6(states):
    obs = np.zeros((*states.shape[:-1], 6), dtype=np.float32)
    obs[..., 0] = states[..., 0]
    obs[..., 2] = states[..., 3]
    obs[..., 3] = states[..., 1]
    obs[..., 4] = states[..., 2]
    return obs.reshape(len(states), 600)


class Reference:
    def __init__(self, path, episodes):
        with np.load(path, allow_pickle=False) as w:
            self.w1 = csr_matrix(w['actor/dense1/kernel'].T)
            self.b1 = w['actor/dense1/bias'][:, None]
            self.w2 = csr_matrix(w['actor/dense2/kernel'].T)
            self.b2 = w['actor/dense2/bias'][:, None]
            self.w3 = csr_matrix(w['actor/logits/kernel'].T)
            self.b3 = w['actor/logits/bias'][:, None]

    def __call__(self, states, active):
        x = observation6(states).T
        h1 = np.tanh(self.w1 @ x + self.b1)
        h2 = np.tanh(self.w2 @ h1 + self.b2)
        return (self.w3 @ h2 + self.b3).T


class Previous:
    def __init__(self, path, episodes):
        with np.load(path, allow_pickle=False) as w:
            self.w1 = csr_matrix(w['w1'])
            self.b1 = w['b1'][:, None]
            self.w2 = csr_matrix(w['w2'])
            self.b2 = w['b2'][:, None]

    def __call__(self, states, active):
        h = np.maximum(self.w1 @ states.reshape(len(states), 400).T + self.b1, 0)
        old = (self.w2 @ h + self.b2).T
        result = np.full((len(states), 6), -80., dtype=np.float32)
        result[:, [0, 1, 3]] = old * 80
        return result


def bootstrap(values):
    rng = np.random.default_rng(20261104)
    samples = values[rng.integers(len(values), size=(2000, len(values)))].mean(axis=1)
    return np.quantile(samples, [0.025, 0.975]).tolist()


def frame(state, score, action):
    return {"p": int(np.flatnonzero(state[9, :, 0])[0]), "b": np.flatnonzero(state[:, :, 1]).tolist(), "t": np.flatnonzero(state[:, :, 2]).tolist(), "k": np.flatnonzero(state[:, :, 3]).tolist(), "s": int(score), "a": int(action)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--minatar', type=Path)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--kind', choices=['reference', 'previous', 'compiled', 'compiled_fast'], required=True)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--split', choices=['development', 'heldout', 'stress', 'reproduction'], required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.minatar:
        sys.path.insert(0, str(args.minatar.resolve()))
    from minatar import Environment
    if args.kind == 'compiled':
        from network import CompiledNetwork
    if args.kind == 'compiled_fast':
        from fast_network import FastCompiledNetwork
    cfg = json.loads(args.plan.read_text())[args.split]
    args.output.mkdir(parents=True, exist_ok=False)
    before = hashlib.sha256(args.model.read_bytes()).hexdigest()
    classes = {'reference': Reference, 'previous': Previous}
    cls = CompiledNetwork if args.kind == 'compiled' else FastCompiledNetwork if args.kind == 'compiled_fast' else classes[args.kind]
    net = cls(args.model, cfg['episodes'])
    envs = []
    uniform = []
    for seed in range(cfg['first_seed'], cfg['first_seed'] + cfg['episodes']):
        env = Environment('breakout', sticky_action_prob=cfg['sticky'])
        env.seed(seed)
        env.reset()
        envs.append(env)
        uniform.append(np.random.default_rng(seed + 987654321).random(cfg['cap']))
    uniform = np.array(uniform).T
    active = np.arange(len(envs))
    reward = np.zeros(len(envs), dtype=np.int64)
    steps = np.zeros(len(envs), dtype=np.int64)
    terminated = np.zeros(len(envs), dtype=bool)
    replay = []
    start = time.perf_counter()
    for step in range(1, cfg['cap'] + 1):
        states = np.stack([envs[int(i)].state() for i in active])
        logits = np.asarray(net(states, active), dtype=np.float64)
        prob = np.exp(logits - logits.max(axis=1, keepdims=True))
        prob /= prob.sum(axis=1, keepdims=True)
        actions = np.minimum((np.cumsum(prob, axis=1) < uniform[step - 1, active, None]).sum(axis=1), 5)
        keep = []
        for j, i in enumerate(active):
            if i == 0:
                replay.append(frame(states[j], reward[i], actions[j]))
            r, done = envs[int(i)].act(int(actions[j]))
            reward[i] += r
            steps[i] = step
            terminated[i] = done
            if not done:
                keep.append(i)
        active = np.array(keep, dtype=int)
        if step % 1000 == 0:
            print(f"step={step} active={len(active)} mean={reward.mean():.3f}", flush=True)
        if len(active) == 0:
            break
    replay.append(frame(envs[0].state(), reward[0], 0))
    assert before == hashlib.sha256(args.model.read_bytes()).hexdigest()
    summary = {'kind': args.kind, 'model': args.model.name, 'sha256': before, 'config': cfg, 'mean': float(reward.mean()), 'se': float(reward.std(ddof=1) / np.sqrt(len(reward))), 'ci95': bootstrap(reward), 'median': float(np.median(reward)), 'min': int(reward.min()), 'max': int(reward.max()), 'mean_steps': float(steps.mean()), 'truncated': int((~terminated).sum()), 'seconds': time.perf_counter() - start}
    with (args.output / 'episodes.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['seed', 'reward', 'steps', 'terminated'])
        for i in range(len(envs)):
            writer.writerow([cfg['first_seed'] + i, reward[i], steps[i], bool(terminated[i])])
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2))
    (args.output / 'replay.json').write_text(json.dumps({'frames': replay, 'episode': {'seed': cfg['first_seed'], 'reward': int(reward[0]), 'steps': int(steps[0]), 'terminated': bool(terminated[0])}}, separators=(',', ':')))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
