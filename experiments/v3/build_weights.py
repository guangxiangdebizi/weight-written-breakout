"""Compile geometric prediction and exact stochastic-control recurrences to weights."""

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix


def index(y, x, channel):
    return (y * 10 + x) * 6 + channel


def fold(t):
    t %= 20
    return t if t < 10 else 19 - t


def control_tables(penalty, sticky=0.1, horizon=18):
    """Bellman recursion from the written transition rules, not sampled experience."""
    tables = {}
    for old in range(10):
        for new in range(max(0, old - 1), min(10, old + 2)):
            values = np.zeros((horizon + 1, 10, 3), dtype=np.float64)
            chosen = np.zeros_like(values, dtype=np.int64)
            values[0, old, :] = 1
            if new != old:
                values[0, new, :] = 1 - penalty
            for t in range(1, horizon + 1):
                for p in range(10):
                    for last in [-1, 0, 1]:
                        best = -1.
                        for command in [0, -1, 1]:
                            normal = values[t - 1, np.clip(p + command, 0, 9), command + 1]
                            repeat = values[t - 1, np.clip(p + last, 0, 9), last + 1]
                            value = (1 - sticky) * normal + sticky * repeat
                            if value > best + 1e-14:
                                best = value
                                chosen[t, p, last + 1] = {0: 0, -1: 1, 1: 3}[command]
                        values[t, p, last + 1] = best
            tables[old, new] = (values, chosen)
    return tables


def matrix(rows, columns, triples):
    r, c, v = zip(*triples)
    return csr_matrix((np.asarray(v, dtype=np.float32), (r, c)), shape=(rows, columns))


def build(penalty):
    features = []
    classes = {}

    def detector(required, forbidden, y, x, dx, up, t):
        key = (t, fold(x + dx * (t - 1)), fold(x + dx * t))
        if key not in classes:
            classes[key] = len(classes)
        features.append((required, forbidden, classes[key], 0 if up else 1))

    def flight(required, y, x, dx, up):
        if not up:
            detector(required, [], y, x, dx, up, 9 - y)
            return
        prior = []
        for row in range(min(3, y - 1), 0, -1):
            brick = index(row, fold(x + dx * (y - row)), 2)
            detector(required + [brick], prior, y, x, dx, up, y + 8 - 2 * row)
            prior = prior + [brick]
        detector(required, prior, y, x, dx, up, y + 10)

    for y in range(9):
        for x in range(10):
            for px in range(max(0, x - 1), min(10, x + 2)):
                if x == px and x not in [0, 9]:
                    continue
                dx = x - px if x != px else 1 if x == 0 else -1
                if y > 0:
                    flight([index(y, x, 3), index(y - 1, px, 4)], y, x, dx, False)
                if y < 8:
                    flight([index(y, x, 3), index(y + 1, px, 4)], y, x, dx, True)
                if y in [0, 1, 2, 3, 4, 8]:
                    pair = [index(y, x, 3), index(y, px, 4)]
                    flight(pair + [610], y, x, dx, False)
                    if y == 8:
                        flight(pair + [611, index(9, px, 0)], y, x, dx, True)
                        if x != px:
                            flight(pair + [611, index(9, x, 0)], y, x, -dx, True)
                    else:
                        flight(pair + [611], y, x, dx, True)

    contexts = [(p, prev) for p in range(10) for prev in range(max(0, p - 1), min(10, p + 2))]
    nf, nc, nq = len(features), len(classes), len(contexts)
    sizes = [612, nf + nq + 10, nc + nq + 12, nc * nq + 12, 18]
    triples = []
    b1 = np.zeros(sizes[1], dtype=np.float32)
    for i, (required, forbidden, _, _) in enumerate(features):
        triples.extend((i, j, 1) for j in required)
        triples.extend((i, j, -1) for j in forbidden)
        b1[i] = 1 - len(required)
    for q, (p, previous) in enumerate(contexts):
        triples += [(nf + q, index(9, p, 0), 1), (nf + q, 600 + previous, 1)]
        b1[nf + q] = -1
    triples.extend((nf + nq + p, index(9, p, 0), 1) for p in range(10))
    w1 = matrix(sizes[1], sizes[0], triples)

    triples = []
    for i, (_, _, c, direction) in enumerate(features):
        triples += [(c, i, 1), (nc + nq + 10 + direction, i, 1)]
    triples.extend((nc + q, nf + q, 1) for q in range(nq))
    triples.extend((nc + nq + p, nf + nq + p, 1) for p in range(10))
    w2 = matrix(sizes[2], sizes[1], triples)

    triples = []
    b3 = np.zeros(sizes[3], dtype=np.float32)
    for c in range(nc):
        for q in range(nq):
            row = c * nq + q
            triples += [(row, c, 1), (row, nc + q, 1)]
            b3[row] = -1
    triples.extend((nc * nq + i, nc + nq + i, 1) for i in range(12))
    w3 = matrix(sizes[3], sizes[2], triples)

    tables = control_tables(penalty)
    triples = []
    for (t, old, new), c in classes.items():
        table = tables[old, new][1]
        for q, (p, previous) in enumerate(contexts):
            action = int(table[t, p, p - previous + 1])
            triples.append((action, c * nq + q, 80))
    triples.extend((6 + i, nc * nq + i, 1) for i in range(12))
    w4 = matrix(18, sizes[3], triples)
    initial = np.zeros(12, dtype=np.float32)
    initial[4] = initial[10] = 1
    weights = {'initial_memory': initial, 'sizes': np.asarray(sizes), 'penalty': np.asarray(penalty)}
    for i, (w, b) in enumerate(zip([w1, w2, w3, w4], [b1, np.zeros(sizes[2], dtype=np.float32), b3, np.zeros(18, dtype=np.float32)])):
        weights[f'w{i}_data'] = w.data
        weights[f'w{i}_indices'] = w.indices
        weights[f'w{i}_indptr'] = w.indptr
        weights[f'w{i}_shape'] = np.array(w.shape)
        weights[f'b{i}'] = b
    metadata = {'sizes': sizes, 'trajectory_features': nf, 'forecast_classes': nc, 'context_classes': nq, 'nonzero_parameters': sum(w.nnz + np.count_nonzero(b) for w, b in zip([w1, w2, w3, w4], [b1, weights['b1'], b3, weights['b3']])), 'dense_parameter_slots': sum(a * b + b for a, b in zip(sizes[:-1], sizes[1:]))}
    metadata['nonzero_parameters'] = int(metadata['nonzero_parameters'])
    return weights, metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=True, parents=True)
    reports = {}
    for penalty in [0.01, 0.05, 0.2]:
        weights, metadata = build(penalty)
        name = f"recurrent-{penalty:g}"
        path = args.output / (name + '.npz')
        np.savez_compressed(path, **weights)
        metadata['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
        metadata['penalty'] = penalty
        reports[name] = metadata
    (args.output / 'models.json').write_text(json.dumps(reports, indent=2))
    print(json.dumps(reports, indent=2))


if __name__ == '__main__':
    main()
