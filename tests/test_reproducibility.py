import csv
import hashlib
import json
from pathlib import Path
import sys
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
V3 = ROOT / 'experiments/v3'
sys.path.insert(0, str(V3))
from build_weights import build


class ReproducibilityTests(unittest.TestCase):
    def test_v3_weights_rebuild_exactly(self):
        for penalty in [0.01, 0.05, 0.2]:
            rebuilt, _ = build(penalty)
            with np.load(V3 / f'recurrent-{penalty:g}.npz', allow_pickle=False) as saved:
                self.assertEqual(set(rebuilt), set(saved.files))
                for key, value in rebuilt.items():
                    np.testing.assert_array_equal(value, saved[key], err_msg=key)

    def test_v3_model_hashes(self):
        metadata = json.loads((V3 / 'models.json').read_text())
        for name, model in metadata.items():
            actual = hashlib.sha256((V3 / f'{name}.npz').read_bytes()).hexdigest()
            self.assertEqual(actual, model['sha256'])
        reference = hashlib.sha256((V3 / 'reference-hand.npz').read_bytes()).hexdigest()
        self.assertEqual(reference, '6e8d42658aed3704c7bf6fb67015e9468cc03f1d67cb8a4fd192e6534fcccb4a')

    def test_v2_v3_recorded_means_and_seed_counts(self):
        checked = 0
        for version in ['v2', 'v3']:
            for path in (ROOT / 'experiments' / version).rglob('episodes.csv'):
                with path.open(newline='', encoding='utf-8') as f:
                    rows = list(csv.DictReader(f))
                summary = json.loads(path.with_name('summary.json').read_text())
                cfg = summary['config']
                self.assertEqual(len(rows), cfg['episodes'])
                self.assertEqual([int(r['seed']) for r in rows], list(range(cfg['first_seed'], cfg['first_seed'] + cfg['episodes'])))
                self.assertAlmostEqual(np.mean([int(r['reward']) for r in rows]), summary['mean'], places=10)
                checked += 1
        self.assertEqual(checked, 18)

    def test_v1_results(self):
        base = ROOT / 'experiments/v1/results'
        summary = json.loads((base / 'summary.json').read_text())
        with (base / 'episodes.csv').open(newline='', encoding='utf-8') as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 4800)
        for condition in ['primary', 'stress']:
            for policy in ['anticipation', 'reactive', 'random', 'zero']:
                scores = [int(r['reward']) for r in rows if r['condition'] == condition and r['policy'] == policy]
                expected = summary['conditions'][condition][policy]
                self.assertEqual(len(scores), expected['episodes'])
                self.assertAlmostEqual(np.mean(scores), expected['mean'], places=10)

    def test_v3_paired_statistics(self):
        report = json.loads((V3 / 'comparison.json').read_text())
        for split in ['heldout', 'stress']:
            values = []
            for model in ['ours', 'reference']:
                with (V3 / f'{split}-{model}/episodes.csv').open() as f:
                    values.append(np.array([int(row['reward']) for row in csv.DictReader(f)]))
            difference = values[0] - values[1]
            self.assertAlmostEqual(difference.mean(), report[split]['paired_mean_difference'], places=10)
            rng = np.random.default_rng(20261104)
            means = difference[rng.integers(len(difference), size=(2000, len(difference)))].mean(axis=1)
            np.testing.assert_allclose(np.quantile(means, [0.025, 0.975]), report[split]['paired_ci95'], rtol=0, atol=1e-10)

    def test_development_and_heldout_do_not_overlap(self):
        protocol = json.loads((V3 / 'protocol.json').read_text())
        ranges = []
        for name in ['development', 'reproduction', 'heldout', 'stress']:
            cfg = protocol[name]
            seeds = set(range(cfg['first_seed'], cfg['first_seed'] + cfg['episodes']))
            for previous in ranges:
                self.assertFalse(seeds & previous)
            ranges.append(seeds)


if __name__ == '__main__':
    unittest.main()
