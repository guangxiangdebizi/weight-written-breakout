"""Generic sparse affine/ReLU recurrent network. No game logic in inference."""

import numpy as np
from scipy.sparse import csr_matrix
from benchmark import observation6


class CompiledNetwork:
    def __init__(self, path, episodes):
        with np.load(path, allow_pickle=False) as w:
            self.layers = []
            for i in range(len(w['sizes']) - 1):
                matrix = csr_matrix((w[f'w{i}_data'], w[f'w{i}_indices'], w[f'w{i}_indptr']), shape=tuple(w[f'w{i}_shape']))
                bias = w[f'b{i}'].copy()
                assert np.all(bias <= 0)
                self.layers.append((matrix, bias))
            self.memory = np.tile(w['initial_memory'], (episodes, 1))

    def forward(self, x):
        h = csr_matrix(x.T)
        for w, bias in self.layers:
            z = (w @ h).tocoo()
            z.sum_duplicates()
            values = z.data + bias[z.row]
            keep = values > 0
            h = csr_matrix((values[keep], (z.row[keep], z.col[keep])), shape=z.shape)
        return h.toarray().T

    def __call__(self, states, active):
        x = np.concatenate([observation6(states), self.memory[active]], axis=1)
        out = self.forward(x)
        self.memory[active] = out[:, 6:]
        return out[:, :6]
