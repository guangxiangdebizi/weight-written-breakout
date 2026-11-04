"""Inference contains only fixed affine transforms, ReLU and action argmax."""

import numpy as np
from scipy.sparse import csr_matrix


class NeuralPolicy:
    def __init__(self, path, zero=False):
        with np.load(path, allow_pickle=False) as weights:
            self.w1 = csr_matrix(weights["w1"])
            self.b1 = weights["b1"].copy()
            self.w2 = csr_matrix(weights["w2"])
            self.b2 = weights["b2"].copy()
            self.actions = weights["actions"].copy()
        if zero:
            self.w1.data.fill(0)
            self.b1.fill(0)
            self.w2.data.fill(0)
            self.b2.fill(0)

    def logits(self, observation):
        x = np.asarray(observation, dtype=np.float32).reshape(-1)
        h = np.maximum(self.w1 @ x + self.b1, 0)
        return self.w2 @ h + self.b2

    def __call__(self, observation):
        return int(self.actions[np.argmax(self.logits(observation))])
