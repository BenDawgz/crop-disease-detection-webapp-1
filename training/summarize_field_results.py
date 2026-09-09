"""Summarize saved predictions on identical common classes and new negatives."""
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'instance/field_improvement'
sys.path.insert(0, str(ROOT))
from training.finetune_field_model import evaluate


def main():
    rows = [r for r in json.loads((OUT / 'manifest.json').read_text()) if r['split'] == 'test']
    scores = np.load(OUT / 'test_predictions.npz')
    mapping = json.loads((OUT / 'class_indices.json').read_text())
    names = sorted(mapping, key=mapping.get)
    baseline_map = json.loads((ROOT / 'instance/class_indices.json').read_text())
    baseline_names = sorted(baseline_map, key=baseline_map.get)
    report = {}
    for source in ('PlantWildV2', 'PlantDoc', 'UserDataset'):
        mask = np.array([r['source'] == source and r['label'] in baseline_names for r in rows])
        subset = [r for r, keep in zip(rows, mask) if keep]
        report[source] = dict(candidate=evaluate(subset, scores['candidate'][mask], names),
                              refined=evaluate(subset, scores['baseline'][mask], baseline_names))
    (OUT / 'common_class_comparison.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({source: {model: {k: v for k, v in result.items() if k not in ('per_class', 'confusion')}
                               for model, result in models.items()} for source, models in report.items()}, indent=2))


if __name__ == '__main__':
    main()
