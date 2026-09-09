"""Select disease threshold on validation only, then report the fixed test result."""
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from prediction_policy import classify_scores, has_leaf_evidence

OUT = Path(os.environ.get('RETRAIN_EXPERIMENT_DIR', ROOT / 'instance' / 'public_retraining'))


def summarize(rows, scores, names, threshold):
    decisions = [classify_scores(s, names, disease_confidence_threshold=threshold)[0] for s in scores]
    supported = np.array([r['label'] not in ('Not_A_Crop', 'Unsupported_Leaf') for r in rows])
    disease = np.array([d not in (None, 'Not_A_Crop', 'Unsupported_Leaf') for d in decisions])
    correct = np.array([d == r['label'] for d, r in zip(decisions, rows)])
    leaf = np.array([r['label'] != 'Not_A_Crop' for r in rows])
    field = np.array([r['source'] == 'PlantDoc' for r in rows])
    unknown = np.array([r['label'] == 'Unsupported_Leaf' for r in rows])
    accepted = supported & disease
    accepted_field = accepted & field
    def rate(values):
        return float(np.mean(values)) if len(values) else None
    return {
        'images': len(rows), 'supported_images': int(supported.sum()),
        'accepted_disease_count': int(accepted.sum()),
        'accepted_disease_accuracy': rate(correct[accepted]),
        'disease_coverage': rate(disease[supported]),
        'field_accepted_disease_count': int(accepted_field.sum()),
        'field_accepted_disease_accuracy': rate(correct[accepted_field]),
        'field_disease_coverage': rate(disease[supported & field]),
        'leaf_images_rejected_as_non_crop': sum(d == 'Not_A_Crop' for d, yes in zip(decisions, leaf) if yes),
        'non_leaf_images_given_disease_label': int((~leaf & disease).sum()),
        'non_leaf_images_with_grouped_leaf_evidence': sum(has_leaf_evidence(s, names) for s, yes in zip(scores, leaf) if not yes),
        'unsupported_leaf_images_given_disease_label': int((unknown & disease).sum()),
    }


def main():
    rows = json.loads((OUT / 'manifest.json').read_text(encoding='utf-8'))
    names = sorted(set(r['label'] for r in rows))
    validation = [r for r in rows if r['split'] == 'val']
    scores = np.load(OUT / 'validation_scores.npy')
    trials = []
    for threshold in (0.70, 0.75, 0.80, 0.85, 0.90, 0.925, 0.95, 0.975, 0.99):
        trials.append(dict(threshold=threshold, **summarize(validation, scores, names, threshold)))
    acceptable = [trial for trial in trials if trial['field_accepted_disease_count'] >= 20
                  and trial['field_accepted_disease_accuracy'] >= 0.90]
    if not acceptable:
        raise RuntimeError('Validation target was not met; do not activate automatically.')
    selected = acceptable[0]
    threshold = selected['threshold']
    test = [r for r in rows if r['split'] == 'test']
    test_scores = np.load(OUT / 'test_predictions.npz')['candidate']
    report = {
        'selection_rule': 'Lowest tested disease threshold with at least 20 accepted supported PlantDoc validation images and >=90% accuracy among those accepted images.',
        'disease_confidence_threshold': threshold, 'other_confidence_threshold': 0.70,
        'margin_threshold': 0.20, 'grouped_leaf_max_non_crop_score': 0.01,
        'validation_trials': trials, 'selected_validation': selected,
        'test': summarize(test, test_scores, names, threshold),
        'note': 'Model scores are not calibrated probabilities. Small accepted field subsets have substantial sampling uncertainty; abstention coverage is reported alongside accuracy.',
    }
    (OUT / 'calibrated_evaluation.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'validation_trials'}, indent=2))


if __name__ == '__main__':
    main()
