"""Train a separate dominant-leaf locator; save a candidate and validation gate."""
import hashlib
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'instance/field_improvement'
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
sys.path.insert(0, str(ROOT))
import numpy as np
import tensorflow as tf
from leaf_locator import box_iou, valid_box
from training.train_public_model import batches


def summary(rows, leaf_scores, boxes, threshold):
    truth = np.array([r['leaf'] for r in rows], dtype=bool)
    accepted = (leaf_scores >= threshold) & np.array([valid_box(b) for b in boxes])
    ious = np.array([box_iou(b, r['box']) for b, r in zip(boxes, rows)])
    return dict(leaf_count=int(truth.sum()), negative_count=int((~truth).sum()),
                leaf_detection_recall=float(accepted[truth].mean()),
                non_leaf_false_detection_rate=float(accepted[~truth].mean()),
                localized_leaf_recall_iou50=float((accepted & (ious >= .5))[truth].mean()),
                mean_leaf_iou=float(ious[truth].mean()),
                full_image_baseline_mean_iou=float(np.mean([box_iou([0, 0, 1, 1], r['box']) for r in rows if r['leaf']])))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature-layer', default='block_16_project_BN')
    parser.add_argument('--variant', default='')
    args = parser.parse_args()
    prefix = args.variant + '_' if args.variant else ''
    tf.keras.utils.set_random_seed(20260908)
    tf.config.threading.set_intra_op_parallelism_threads(8)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    rows = json.loads((OUT / 'box_manifest.json').read_text())
    current = tf.keras.models.load_model(ROOT / 'instance/models/best_model.h5', compile=False)
    backbone = next(layer for layer in current.layers if isinstance(layer, tf.keras.Model) and 'mobilenet' in layer.name)
    encoder = tf.keras.Model(backbone.input, backbone.get_layer(args.feature_layer).output)
    encoder.trainable = False
    @tf.function(reduce_retracing=True)
    def encode(pixels):
        return encoder(pixels * 2 - 1, training=False)
    subsets = {s: [r for r in rows if r['split'] == s] for s in ('train', 'val', 'test')}
    digest = hashlib.sha256((OUT / 'box_manifest.json').read_bytes()).hexdigest()[:12]
    features = {}
    for split, subset in subsets.items():
        cache = OUT / f'locator_features_{args.feature_layer}_{digest}_{split}.npy'
        if not cache.exists():
            encoded = []
            for i, pixels in enumerate(batches(subset, 224)):
                encoded.append(encode(pixels).numpy())
                if i % 20 == 0:
                    print(f'Locator features {split}: {i * 24}/{len(subset)}', flush=True)
            np.save(cache, np.concatenate(encoded))
        features[split] = np.load(cache)
    inputs = tf.keras.Input(shape=features['train'].shape[1:])
    normalized = tf.keras.layers.LayerNormalization(axis=-1)(inputs)
    shared = tf.keras.layers.Conv2D(32, 1, activation='relu')(normalized)
    leaf = tf.keras.layers.Dense(1, activation='sigmoid', name='leaf')(tf.keras.layers.GlobalAveragePooling2D()(shared))
    spatial = tf.keras.layers.Dense(64, activation='relu')(tf.keras.layers.Flatten()(shared))
    spatial = tf.keras.layers.Dropout(.2)(spatial)
    box = tf.keras.layers.Dense(4, activation='sigmoid', name='box', kernel_initializer='zeros',
                                bias_initializer=tf.keras.initializers.Constant([-2., -2., 2., 2.]))(spatial)
    head = tf.keras.Model(inputs, {'leaf': leaf, 'box': box})
    head.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                 loss={'leaf': 'binary_crossentropy', 'box': 'mean_squared_error'},
                 loss_weights={'leaf': 1., 'box': 10.})
    y = {s: {'leaf': np.array([r['leaf'] for r in rs], dtype=np.float32)[:, None],
             'box': np.array([r['box'] for r in rs], dtype=np.float32)} for s, rs in subsets.items()}
    weights = {s: {'leaf': np.ones(len(rs)), 'box': y[s]['leaf'].ravel()} for s, rs in subsets.items()}
    history = head.fit(features['train'], y['train'], sample_weight=weights['train'],
                       validation_data=(features['val'], y['val'], weights['val']), epochs=60,
                       batch_size=64, verbose=2, callbacks=[tf.keras.callbacks.EarlyStopping(
                           monitor='val_loss', patience=7, restore_best_weights=True)])
    pixels = tf.keras.Input((224, 224, 3))
    model = tf.keras.Model(pixels, head(encoder(tf.keras.layers.Rescaling(2., offset=-1.)(pixels), training=False)))
    model.save(OUT / f'{prefix}leaf_locator.keras')
    predictions = {s: head.predict(x, batch_size=64, verbose=0) for s, x in features.items() if s != 'train'}
    trials = [dict(threshold=t, **summary(subsets['val'], predictions['val']['leaf'].ravel(), predictions['val']['box'], t))
              for t in (.5, .7, .8, .9, .95, .975, .99)]
    eligible = [r for r in trials if r['non_leaf_false_detection_rate'] <= .02 and r['localized_leaf_recall_iou50'] >= .7]
    threshold = eligible[0]['threshold'] if eligible else .9
    report = dict(validation_gate_passed=bool(eligible), threshold=threshold, trials=trials, manifest_digest=digest,
                  test=summary(subsets['test'], predictions['test']['leaf'].ravel(), predictions['test']['box'], threshold),
                  feature_layer=args.feature_layer, history=history.history, model_sha256=hashlib.sha256((OUT / f'{prefix}leaf_locator.keras').read_bytes()).hexdigest(),
                  limitations=['Dominant-leaf bounding box only, not instance segmentation.',
                               'PlantDoc is a reused web benchmark; Imagenette overlaps ImageNet pretraining.',
                               'A passed locator gate alone does not establish improved disease diagnosis.'])
    (OUT / f'{prefix}locator_evaluation.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k not in ('history', 'trials')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
