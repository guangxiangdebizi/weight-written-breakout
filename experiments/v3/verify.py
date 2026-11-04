import argparse
from pathlib import Path
import sys
import numpy as np
from scipy.sparse import csr_matrix
from network import CompiledNetwork
from fast_network import FastCompiledNetwork
from benchmark import Reference, observation6


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--minatar', type=Path)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--reference', type=Path, required=True)
    args = p.parse_args()
    if args.minatar:
        sys.path.insert(0, str(args.minatar.resolve()))
    from minatar import Environment
    envs = []
    for seed in range(2000, 2016):
        env = Environment('breakout', sticky_action_prob=0.1)
        env.seed(seed)
        env.reset()
        envs.append(env)
    net = CompiledNetwork(args.model, 16)
    fast = FastCompiledNetwork(args.model, 16)
    initial_states = np.stack([env.state() for env in envs])
    x = np.concatenate([observation6(initial_states), net.memory], axis=1)
    sparse = net.forward(x)
    np.testing.assert_array_equal(fast.forward(x), sparse)
    dense = x
    for matrix, bias in net.layers:
        dense = np.maximum(dense @ matrix.toarray().T + bias, 0)
    np.testing.assert_array_equal(sparse, dense)
    ref = Reference(args.reference, 16)
    with np.load(args.reference) as w:
        dense = np.tanh(observation6(initial_states) @ w['actor/dense1/kernel'] + w['actor/dense1/bias'])
        dense = np.tanh(dense @ w['actor/dense2/kernel'] + w['actor/dense2/bias'])
        dense = dense @ w['actor/logits/kernel'] + w['actor/logits/bias']
    np.testing.assert_allclose(ref(initial_states, np.arange(16)), dense, rtol=1e-5, atol=1e-5)
    active = np.arange(16)
    checked = 0
    for _ in range(300):
        states = np.stack([envs[int(i)].state() for i in active])
        logits = net(states, active)
        np.testing.assert_array_equal(fast(states, active), logits)
        np.testing.assert_array_equal(fast.memory, net.memory)
        np.testing.assert_array_equal((logits == 80).sum(axis=1), np.ones(len(active)))
        np.testing.assert_array_equal(net.memory[active, :10].sum(axis=1), np.ones(len(active)))
        np.testing.assert_array_equal(net.memory[active, 10:].sum(axis=1), np.ones(len(active)))
        # Hidden simulator direction is inspected only here as an audit oracle.
        for j, i in enumerate(active):
            expected_up = envs[int(i)].env.ball_dir in [0, 1]
            assert net.memory[i, 10] == int(expected_up), (i, checked, net.memory[i])
        checked += len(active)
        keep = []
        for j, i in enumerate(active):
            _, done = envs[int(i)].act(int(np.argmax(logits[j])))
            if not done:
                keep.append(i)
        active = np.array(keep, dtype=int)
        if len(active) == 0:
            break
    print(f'PASS: dense/sparse forward equivalence; reference forward tolerance; {checked} states with one-hot actions/memory and correctly inferred hidden direction')


if __name__ == '__main__':
    main()
