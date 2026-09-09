"""Refine the installed head; select an epoch using validation only."""
import hashlib
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


def main():
    tf.keras.utils.set_random_seed(20260908)
    tf.config.threading.set_intra_op_parallelism_threads(8)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    rows = json.loads((OUT / 'manifest.json').read_text())
    names = sorted(set(r['label'] for r in rows))
    subsets = {s: [r for r in rows if r['split'] == s] for s in ('train', 'val', 'test')}
    x = {s: np.load(next(OUT.glob(f'features_*_{s}.npz')))['x'] for s in subsets}
    y = {s: np.array([names.index(r['label']) for r in subsets[s]]) for s in subsets}
    model = tf.keras.models.load_model(ROOT / 'instance/models/best_model.h5', compile=False)
    head = model.get_layer('public_leaf_classifier')
    head.compile(optimizer=tf.keras.optimizers.Adam(0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    counts = np.bincount(y['train'], minlength=len(names))
    weights = np.sqrt(counts.mean() / counts)[y['train']]
    weights *= np.array([3.0 if r['source'] == 'PlantDoc' else 1.0 for r in subsets['train']])
    supported = np.array([r['label'] not in ('Not_A_Crop', 'Unsupported_Leaf') for r in subsets['val']])

    def assess(scores):
        trials = [(t, summarize(subsets['val'], scores, names, t)) for t in (.70, .75, .80, .85, .90, .925, .95, .975, .99)]
        eligible = [(t, r) for t, r in trials if r['field_accepted_disease_count'] >= 20 and r['field_accepted_disease_accuracy'] >= .90]
        top1 = float(np.mean(np.argmax(scores, axis=1)[supported] == y['val'][supported]))
        if not eligible:
            return None, top1
        threshold, result = eligible[0]
        return dict(threshold=threshold, **result), top1

    initial, initial_top1 = assess(head(x['val'], training=False).numpy())
    best_quality = initial_top1
    best_weights, best_policy, best_epoch = None, None, None
    history = []
    for epoch in range(1, 21):
        fit = head.fit(x['train'], y['train'], sample_weight=weights, batch_size=128, epochs=1, verbose=0)
        policy, accuracy = assess(head(x['val'], training=False).numpy())
        record = dict(epoch=epoch, loss=float(fit.history['loss'][0]), validation_top1=accuracy,
                      eligible=policy is not None, policy=policy)
        history.append(record)
        print(json.dumps({k: v for k, v in record.items() if k != 'policy'}), flush=True)
        if policy is not None and accuracy > best_quality + 0.001:
            best_quality = accuracy
            best_weights = head.get_weights()
            best_policy, best_epoch = policy, epoch
    report = dict(initial_validation_top1=initial_top1, initial_policy=initial,
                  selected_epoch=best_epoch, selected_policy=best_policy, history=history)
    (OUT / 'refinement_selection.json').write_text(json.dumps(report, indent=2))
    if best_weights is None:
        print('No refinement met the validation requirement and improved supported-class accuracy. Runtime unchanged.', flush=True)
        return
    head.set_weights(best_weights)
    model.save(OUT / 'refined_candidate.h5')
    np.save(OUT / 'refined_validation_scores.npy', head(x['val'], training=False).numpy())
    test_scores = head(x['test'], training=False).numpy()
    np.save(OUT / 'refined_test_scores.npy', test_scores)
    report['test'] = summarize(subsets['test'], test_scores, names, best_policy['threshold'])
    report['model_sha256'] = hashlib.sha256((OUT / 'refined_candidate.h5').read_bytes()).hexdigest()
    (OUT / 'refinement_selection.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({'epoch': best_epoch, 'policy': best_policy, 'test': report['test']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
