"""Compare local candidate with the installed public-data model at fixed policies."""
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'instance/local_retraining'
os.environ['RETRAIN_EXPERIMENT_DIR'] = str(OUT)
sys.path.insert(0, str(ROOT))

import numpy as np
import tensorflow as tf
from training.calibrate_public_model import summarize
from training.train_public_model import metrics


def main():
    refined = os.environ.get('RETRAIN_CANDIDATE_VARIANT') == 'refined'
    rows = json.loads((OUT / 'manifest.json').read_text())
    test = [r for r in rows if r['split'] == 'test']
    names = sorted(set(r['label'] for r in rows))
    new = tf.keras.models.load_model(OUT / ('refined_candidate.h5' if refined else 'candidate.h5'), compile=False)
    old = tf.keras.models.load_model(ROOT / 'instance/models/best_model.h5', compile=False)
    old_names = [k for k, v in sorted(json.loads((ROOT / 'instance/class_indices.json').read_text()).items(), key=lambda x: x[1])]
    assert names == old_names
    new_base = next(layer for layer in new.layers if layer.name.startswith('mobilenet'))
    old_base = next(layer for layer in old.layers if layer.name.startswith('mobilenet'))
    for a, b in zip(new_base.get_weights(), old_base.get_weights()):
        np.testing.assert_array_equal(a, b)
    features = np.load(next(OUT.glob('features_*_test.npz')))['x']
    old_scores = old.get_layer('public_leaf_classifier')(features, training=False).numpy()
    if refined:
        new_scores = np.load(OUT / 'refined_test_scores.npy')
        calibration = json.loads((OUT / 'refinement_selection.json').read_text())
        threshold = calibration['selected_policy']['threshold']
    else:
        new_scores = np.load(OUT / 'test_predictions.npz')['candidate']
        calibration = json.loads((OUT / 'calibrated_evaluation.json').read_text())
        threshold = calibration['disease_confidence_threshold']
    result = {'candidate_disease_threshold': threshold, 'current_disease_threshold': 0.95,
              'candidate': summarize(test, new_scores, names, threshold),
              'current': summarize(test, old_scores, names, 0.95), 'by_source': {}}
    for source in sorted(set(r['source'] for r in test)):
        mask = np.array([r['source'] == source for r in test])
        selected = [r for r in test if r['source'] == source]
        result['by_source'][source] = {
            'candidate': summarize(selected, new_scores[mask], names, threshold),
            'current': summarize(selected, old_scores[mask], names, 0.95),
            'candidate_top1': metrics(selected, new_scores[mask], names),
            'current_top1': metrics(selected, old_scores[mask], names),
        }
    np.save(OUT / 'previous_runtime_test_scores.npy', old_scores)
    (OUT / ('refined_runtime_comparison.json' if refined else 'runtime_comparison.json')).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
