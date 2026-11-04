"""Bit-parallel evaluation of verified binary ReLU circuits; no policy rules."""

import numpy as np
from network import CompiledNetwork
from benchmark import observation6


class FastCompiledNetwork(CompiledNetwork):
    def __init__(self, path, episodes):
        super().__init__(path, episodes)
        self.operations = []
        for layer, (w, bias) in enumerate(self.layers):
            operations = []
            for row in range(w.shape[0]):
                start, end = w.indptr[row:row + 2]
                columns, values = w.indices[start:end], w.data[start:end]
                if layer in [0, 2]:
                    assert np.all(np.isin(values, [-1, 1]))
                    positive = tuple(int(i) for i in columns[values == 1])
                    negative = tuple(int(i) for i in columns[values == -1])
                    assert bias[row] == 1 - len(positive)
                    operations.append((positive, negative))
                else:
                    assert bias[row] == 0
                    assert len(values) == 0 or (values[0] > 0 and np.all(values == values[0]))
                    operations.append((tuple(int(i) for i in columns), float(values[0]) if len(values) else 1.))
            self.operations.append(operations)

    def forward(self, x):
        assert np.all((x == 0) | (x == 1))
        batch = len(x)
        packed = np.packbits(x.astype(np.uint8).T, axis=1, bitorder='little')
        bits = [int.from_bytes(row.tobytes(), 'little') for row in packed]
        mask = (1 << batch) - 1
        scales = None
        for layer, operations in enumerate(self.operations):
            output = []
            if layer in [0, 2]:
                for positive, negative in operations:
                    value = mask
                    for col in positive:
                        value &= bits[col]
                        if value == 0:
                            break
                    if value:
                        for col in negative:
                            value &= ~bits[col]
                    output.append(value)
            else:
                scales = []
                for columns, scale in operations:
                    value = 0
                    for col in columns:
                        add = bits[col]
                        # A union equals an arithmetic sum only for disjoint events.
                        assert value & add == 0, (layer, 'non-binary activation')
                        value |= add
                    output.append(value)
                    scales.append(scale)
            bits = output
        length = (batch + 7) // 8
        rows = np.stack([np.frombuffer(value.to_bytes(length, 'little'), dtype=np.uint8) for value in bits])
        decoded = np.unpackbits(rows, axis=1, bitorder='little')[:, :batch].T.astype(np.float32)
        return decoded * np.asarray(scales, dtype=np.float32)
