"""Train and evaluate a candidate with frozen ImageNet features; never activate it.

Input/output are compatible with app.py: RGB [0, 1], nearest-neighbor resize,
and a softmax over class_indices.json. All preprocessing is saved in the model.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get('RETRAIN_EXPERIMENT_DIR', ROOT / 'instance' / 'public_retraining'))
os.environ.setdefault('KERAS_HOME', str(ROOT / 'instance/public_retraining/keras_cache'))
USE_FLIPS = os.environ.get('RETRAIN_FEATURE_FLIPS', '1') == '1'
FIELD_WEIGHT = float(os.environ.get('RETRAIN_FIELD_WEIGHT', '1'))
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import json
import hashlib
import sys
import numpy as np
from PIL import Image
import tensorflow as tf

sys.path.insert(0, str(ROOT))
from prediction_policy import classify_scores


def batches(rows, size, batch_size=24, flip=False):
    for start in range(0, len(rows), batch_size):
        batch = []
        for row in rows[start:start + batch_size]:
            with Image.open(ROOT / row['path']) as image:
                im = image.convert('RGB').resize((size, size), Image.Resampling.NEAREST)
                arr = np.asarray(im, dtype=np.float32) / 255.0
                batch.append(arr[:, ::-1] if flip else arr)
        yield np.stack(batch)


def predictions(model, rows):
    result = []
    for index, batch in enumerate(batches(rows, model.input_shape[1])):
        result.append(model(batch, training=False).numpy())
        if index % 20 == 0:
            print(f'Evaluating: {min((index + 1) * 24, len(rows))}/{len(rows)}', flush=True)
    return np.concatenate(result)


def metrics(rows, scores, names):
    truth_leaf = np.array([r['label'] != 'Not_A_Crop' for r in rows])
    predicted = np.argmax(scores, axis=1)
    pred_labels = np.array(names)[predicted]
    predicted_leaf = pred_labels != 'Not_A_Crop'
    decisions = [classify_scores(score, names)[0] for score in scores]
    confident_crop = np.array([x not in (None, 'Not_A_Crop', 'Unsupported_Leaf') for x in decisions])
    supported = np.array([r['label'] in names and r['label'] not in ('Not_A_Crop', 'Unsupported_Leaf') for r in rows])
    correct = pred_labels == np.array([r['label'] for r in rows])
    accepted = confident_crop & supported
    def mean(values):
        return float(np.mean(values)) if len(values) else None
    return {
        'images': len(rows),
        'leaf_images': int(truth_leaf.sum()),
        'non_leaf_images': int((~truth_leaf).sum()),
        'leaf_false_rejection_rate_argmax': mean(~predicted_leaf[truth_leaf]),
        'non_leaf_false_acceptance_rate_argmax': mean(predicted_leaf[~truth_leaf]),
        'non_leaf_confident_disease_rate': mean(confident_crop[~truth_leaf]),
        'supported_disease_top1_accuracy': mean(correct[supported]),
        'supported_disease_accepted_count': int(accepted.sum()),
        'supported_disease_accepted_accuracy': mean(correct[accepted]),
        'supported_disease_coverage': mean(confident_crop[supported]),
    }


def main():
    tf.keras.utils.set_random_seed(20260908)
    tf.config.threading.set_intra_op_parallelism_threads(8)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    rows = json.loads((OUT / 'manifest.json').read_text(encoding='utf-8'))
    names = sorted(set(r['label'] for r in rows))
    indices = {name: index for index, name in enumerate(names)}
    for split in ('train', 'val', 'test'):
        for name in names:
            assert any(r['split'] == split and r['label'] == name for r in rows), (split, name)
    (OUT / 'class_indices.json').write_text(json.dumps(indices, indent=2), encoding='utf-8')
    backbone = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False,
                                                weights='imagenet', pooling='avg')
    backbone.trainable = False
    @tf.function(reduce_retracing=True)
    def encode(batch):
        return backbone(batch * 2.0 - 1.0, training=False)
    features, targets, split_rows = {}, {}, {}
    manifest_digest = hashlib.sha256((OUT / 'manifest.json').read_bytes()).hexdigest()[:12]
    for split in ('train', 'val', 'test'):
        subset = [r for r in rows if r['split'] == split]
        split_rows[split] = subset
        targets[split] = np.array([indices[r['label']] for r in subset])
        cache = OUT / f'features_{manifest_digest}_flips{int(USE_FLIPS)}_{split}.npz'
        if cache.exists():
            saved = np.load(cache)
            features[split] = saved['x']
        else:
            result = []
            for flip in ([False, True] if split == 'train' and USE_FLIPS else [False]):
                for index, batch in enumerate(batches(subset, 224, flip=flip)):
                    result.append(encode(batch).numpy())
                    if index % 20 == 0:
                        print(f'Extracting {split}, flip={flip}: {min((index + 1) * 24, len(subset))}/{len(subset)}', flush=True)
            features[split] = np.concatenate(result)
            np.savez_compressed(cache, x=features[split])
        if split == 'train' and USE_FLIPS:
            targets[split] = np.tile(targets[split], 2)

    head = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(1280,)),
        tf.keras.layers.Dense(192, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(len(names), activation='softmax'),
    ], name='public_leaf_classifier')
    head.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                 loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    counts = np.bincount(targets['train'], minlength=len(names))
    weights = {i: float(np.sqrt(counts.mean() / count)) for i, count in enumerate(counts)}
    source_weights = np.array([FIELD_WEIGHT if r['source'] == 'PlantDoc' else 1.0 for r in split_rows['train']])
    if USE_FLIPS:
        source_weights = np.tile(source_weights, 2)
    sample_weights = source_weights * np.array([weights[int(i)] for i in targets['train']])
    validation_weights = np.array([FIELD_WEIGHT if r['source'] == 'PlantDoc' else 1.0 for r in split_rows['val']])
    print('Training classifier on cached visual features', flush=True)
    history = head.fit(features['train'], targets['train'],
        validation_data=(features['val'], targets['val'], validation_weights),
        batch_size=128, epochs=60, sample_weight=sample_weights, verbose=2,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True),
                   tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.5, min_lr=0.00001)])
    (OUT / 'history.json').write_text(json.dumps(history.history, indent=2), encoding='utf-8')
    inputs = tf.keras.Input(shape=(224, 224, 3))
    normalized = tf.keras.layers.Rescaling(2.0, offset=-1.0)(inputs)
    outputs = head(backbone(normalized, training=False))
    candidate = tf.keras.Model(inputs, outputs, name='crop_public_transfer')
    candidate.save(OUT / 'candidate.h5')
    candidate = tf.keras.models.load_model(OUT / 'candidate.h5', compile=False)
    np.save(OUT / 'validation_scores.npy', head(features['val'], training=False).numpy())
    # Verify the saved end-to-end model reproduces the feature-based classifier.
    expected = head(features['test'][:24], training=False).numpy()
    actual = candidate(next(batches(split_rows['test'], 224)), training=False).numpy()
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)
    test_scores = head(features['test'], training=False).numpy()
    old_names = [k for k, v in sorted(json.loads((ROOT / 'class_indices.json').read_text()).items(), key=lambda item: item[1])]
    original = tf.keras.models.load_model(ROOT / 'best_model.h5', compile=False)
    original_scores = predictions(original, split_rows['test'])
    report = {
        'tensorflow': tf.__version__, 'seed': 20260908,
        'horizontal_flip_features': USE_FLIPS, 'field_training_weight': FIELD_WEIGHT,
        'manifest_sha256': hashlib.sha256((OUT / 'manifest.json').read_bytes()).hexdigest(),
        'candidate_sha256': hashlib.sha256((OUT / 'candidate.h5').read_bytes()).hexdigest(),
        'original_sha256': hashlib.sha256((ROOT / 'best_model.h5').read_bytes()).hexdigest(),
        'training_images': len(split_rows['train']), 'validation_images': len(split_rows['val']),
        'test_images': len(split_rows['test']), 'epochs_completed': len(history.history['loss']),
        'candidate': metrics(split_rows['test'], test_scores, names),
        'original': metrics(split_rows['test'], original_scores, old_names),
        'by_source': {}, 'user_examples': [],
        'limitations': [
            'Pilot dataset subset, not a field deployment accuracy guarantee.',
            'Imagenette contains ImageNet images; the pretrained backbone may have seen them. Negative test results are not an independent pretraining benchmark.',
            'PlantVillage grouping metadata and perceptual deduplication reduce leakage but do not prove all image independence.',
            'PlantDoc web labels may be noisy; unsupported leaves are other crop species, not comprehensive unknown disease coverage.',
            'The four user images have leaf-presence labels only; disease diagnosis accuracy cannot be scored on them.',
            'The original model training set is unknown and may overlap the public test images.',
        ],
    }
    for source in sorted(set(r['source'] for r in split_rows['test'])):
        mask = np.array([r['source'] == source for r in split_rows['test']])
        subset = [r for r in split_rows['test'] if r['source'] == source]
        report['by_source'][source] = {'candidate': metrics(subset, test_scores[mask], names),
                                       'original': metrics(subset, original_scores[mask], old_names)}
    common = np.array([r['label'] in old_names for r in split_rows['test']])
    report['common_classes'] = {
        'candidate': metrics([r for r, keep in zip(split_rows['test'], common) if keep], test_scores[common], names),
        'original': metrics([r for r, keep in zip(split_rows['test'], common) if keep], original_scores[common], old_names)}
    report['per_class'] = {}
    for label in names:
        mask = np.array([r['label'] == label for r in split_rows['test']])
        report['per_class'][label] = {
            'test_count': int(mask.sum()),
            'candidate_top1_recall': float(np.mean(np.argmax(test_scores[mask], axis=1) == indices[label])),
            'original_top1_recall': (float(np.mean(np.argmax(original_scores[mask], axis=1) == old_names.index(label)))
                                     if label in old_names else None),
        }
    for filename in ('images (1).jpg', 'images (3).jpg', 'images (4).jpg', 'tomato-disease-11-TSWV.jpg'):
        path = Path.home() / 'Downloads' / filename
        if not path.exists():
            continue
        one = [{'path': str(path)}]
        scores = predictions(candidate, one)[0]
        report['user_examples'].append({'file': filename, 'expected_leaf': True,
            'decision': classify_scores(scores, names),
            'non_crop_score': float(scores[indices['Not_A_Crop']]),
            'top3': [(names[int(i)], float(scores[i])) for i in np.argsort(scores)[-3:][::-1]]})
    np.savez_compressed(OUT / 'test_predictions.npz', candidate=test_scores, original=original_scores)
    (OUT / 'evaluation.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
