"""Fine-tune MobileNet visual layers on field photos; never activate automatically.

Select checkpoints on validation loss only. Test data is used after selection.
Export uses exactly the app's RGB [0,1] nearest-neighbor input convention.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'instance/field_improvement'
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
sys.path.insert(0, str(ROOT))
import numpy as np
from PIL import Image
import tensorflow as tf
from prediction_policy import classify_scores
from training.train_public_model import batches


def evaluate(rows, scores, names, threshold=.95, unlocalized=None):
    truth = np.array([r['label'] for r in rows])
    top = np.asarray(names)[scores.argmax(axis=1)]
    if unlocalized is not None:
        top = np.where(unlocalized, 'No leaf located', top)
    decisions = np.array([classify_scores(s, names, disease_confidence_threshold=threshold)[0] for s in scores], dtype=object)
    accepted = np.array([x not in (None, 'Not_A_Crop', 'Unsupported_Leaf') for x in decisions])
    supported = ~np.isin(truth, ['Not_A_Crop', 'Unsupported_Leaf'])
    correct = decisions == truth
    def rate(values):
        return float(np.mean(values)) if len(values) else None
    summary = dict(images=len(rows), supported_top1_accuracy=rate((top == truth)[supported]),
                   accepted_disease_count=int((accepted & supported).sum()),
                   accepted_disease_accuracy=rate(correct[accepted & supported]),
                   disease_coverage=rate(accepted[supported]),
                   non_leaf_given_disease=int((accepted & (truth == 'Not_A_Crop')).sum()),
                   leaf_rejected_as_non_leaf=int(((decisions == 'Not_A_Crop') & (truth != 'Not_A_Crop')).sum()))
    summary['per_class'] = {}
    for label in sorted(set(truth)):
        actual = truth == label
        predicted = top == label
        issued = decisions == label
        summary['per_class'][label] = dict(images=int(actual.sum()),
            top1_precision=rate((truth == label)[predicted]), top1_recall=rate((top == label)[actual]),
            accepted_count=int(issued.sum()), accepted_precision=rate((truth == label)[issued]),
            decision_recall=rate((decisions == label)[actual]))
    summary['confusion'] = {label: {str(other): int(((truth == label) & (top == other)).sum())
                                  for other in sorted(set(top[truth == label]))} for label in sorted(set(truth))}
    return summary


def main():
    global OUT
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment-dir', type=Path, default=OUT)
    parser.add_argument('--warmup-epochs', type=int, default=2)
    parser.add_argument('--finetune-epochs', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--smoke', action='store_true', help='Two training batches only; no disease accuracy claim.')
    args = parser.parse_args()
    OUT = args.experiment_dir.resolve()
    tf.keras.utils.set_random_seed(20260908)
    tf.config.threading.set_intra_op_parallelism_threads(8)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    rows = json.loads((OUT / 'manifest.json').read_text())
    names = [k for k, v in sorted(json.loads((OUT / 'class_indices.json').read_text()).items(), key=lambda kv: kv[1])]
    splits = {s: [r for r in rows if r['split'] == s] for s in ('train', 'val', 'test')}
    for key in ('group', 'pixel_sha256'):
        observed = {}
        for r in rows:
            identity = (r['source'], r[key]) if key == 'group' else r[key]
            if identity in observed and observed[identity] != r['split']:
                raise ValueError(f'Partition leakage detected for {key}')
            observed[identity] = r['split']
    original_path = ROOT / 'instance/models/best_model.h5'
    baseline = tf.keras.models.load_model(original_path, compile=False)
    baseline_names = [k for k, v in sorted(json.loads((ROOT / 'instance/class_indices.json').read_text()).items(), key=lambda kv: kv[1])]
    # Reload to preserve an immutable baseline during fine-tuning.
    source = tf.keras.models.load_model(original_path, compile=False)
    backbone = next(layer for layer in source.layers if isinstance(layer, tf.keras.Model) and 'mobilenet' in layer.name)
    old_head = source.get_layer('public_leaf_classifier')
    old_dense = [l for l in old_head.layers if isinstance(l, tf.keras.layers.Dense)]
    hidden = tf.keras.layers.Dense(192, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4))
    output = tf.keras.layers.Dense(len(names), activation='softmax')
    inputs = tf.keras.Input((224, 224, 3))
    # Keep BatchNorm statistics fixed even while selected visual weights learn.
    features = backbone(tf.keras.layers.Rescaling(2., offset=-1.)(inputs), training=False)
    prediction = output(tf.keras.layers.Dropout(.3)(hidden(features)))
    model = tf.keras.Model(inputs, prediction, name='field_crop_classifier')
    hidden.set_weights(old_dense[0].get_weights())
    weights, bias = output.get_weights()
    previous_weights, previous_bias = old_dense[-1].get_weights()
    for index, label in enumerate(names):
        if label in baseline_names:
            old_index = baseline_names.index(label)
            weights[:, index], bias[index] = previous_weights[:, old_index], previous_bias[old_index]
    output.set_weights([weights, bias])

    def read_image(encoded):
        path = (encoded.item() if hasattr(encoded, 'item') else encoded).decode('utf-8')
        with Image.open(path) as image:
            return np.asarray(image.convert('RGB').resize((224, 224), Image.Resampling.NEAREST), dtype=np.float32) / 255
    counts = np.bincount([names.index(r['label']) for r in splits['train']], minlength=len(names))
    class_weights = np.minimum(3., np.sqrt(counts.mean() / counts))
    def dataset(subset, training):
        paths = [str(ROOT / r['path']) for r in subset]
        labels = np.array([names.index(r['label']) for r in subset], dtype=np.int32)
        weights = np.array([(3. if r['source'] in ('PlantDoc', 'PlantWildV2') else 1.) for r in subset], dtype=np.float32)
        if training:
            weights *= class_weights[labels]
        ds = tf.data.Dataset.from_tensor_slices((paths, labels, weights))
        if training:
            ds = ds.shuffle(len(subset), seed=20260908)
        def decode(path, label, weight):
            im = tf.numpy_function(read_image, [path], tf.float32)
            im.set_shape((224, 224, 3))
            if training:
                im = tf.image.random_flip_left_right(im)
                im = tf.clip_by_value(tf.image.random_brightness(im, .08), 0., 1.)
            return im, label, weight
        ds = ds.map(decode, num_parallel_calls=4).batch(args.batch_size).prefetch(2)
        options = tf.data.Options()
        options.threading.private_threadpool_size = 4
        return ds.with_options(options)
    train, val = dataset(splits['train'], True), dataset(splits['val'], False)
    if args.smoke:
        train, val = train.take(2), val.take(1)
    class Progress(tf.keras.callbacks.Callback):
        def __init__(self, phase):
            super().__init__()
            self.phase, self.last = phase, 0
        def on_train_batch_end(self, batch, logs=None):
            if time.monotonic() - self.last > 25:
                state = dict(phase=self.phase, batch=batch, **{k: float(v) for k, v in (logs or {}).items()})
                (OUT / 'training_progress.json').write_text(json.dumps(state))
                print(json.dumps(state), flush=True)
                self.last = time.monotonic()
    checkpoint = OUT / ('smoke.weights.h5' if args.smoke else 'field_best.weights.h5')
    best_loss = float('inf')
    selected_phase = None
    histories = {}
    for phase, epochs, learning_rate in [('head', args.warmup_epochs, 1e-4), ('visual_layers', args.finetune_epochs, 1e-5)]:
        backbone.trainable = phase == 'visual_layers'
        if backbone.trainable:
            unfreeze = False
            for layer in backbone.layers:
                if layer.name == 'block_13_expand':
                    unfreeze = True
                layer.trainable = unfreeze and not isinstance(layer, tf.keras.layers.BatchNormalization)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate, clipnorm=1.), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        trainable_names = [l.name for l in backbone.layers if l.trainable] if backbone.trainable else []
        print(f'{phase}: {len(trainable_names)} visual layers trainable', flush=True)
        callback = tf.keras.callbacks.ModelCheckpoint(str(checkpoint), monitor='val_loss', save_best_only=True, save_weights_only=True, initial_value_threshold=best_loss)
        fit = model.fit(train, validation_data=val, epochs=1 if args.smoke else epochs, verbose=2,
                        callbacks=[Progress(phase), callback, tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)])
        histories[phase] = fit.history
        if min(fit.history['val_loss']) < best_loss:
            selected_phase = phase
        best_loss = min(best_loss, min(fit.history['val_loss']))
        model.load_weights(checkpoint)
    if args.smoke:
        model.save(OUT / 'smoke_candidate.keras')
        reloaded = tf.keras.models.load_model(OUT / 'smoke_candidate.keras', compile=False)
        sample = next(batches(splits['val'][:2], 224))
        np.testing.assert_allclose(model(sample, training=False), reloaded(sample, training=False), atol=1e-6)
        print('Smoke training, visual-layer updates, and export/reload passed. No accuracy claim.', flush=True)
        return
    model.save(OUT / 'field_candidate.keras')
    model = tf.keras.models.load_model(OUT / 'field_candidate.keras', compile=False)
    def predict(candidate, subset):
        scores = []
        for i, pixels in enumerate(batches(subset, 224)):
            scores.append(candidate(pixels, training=False).numpy())
            if i % 20 == 0:
                print(f'Evaluating {i*24}/{len(subset)}', flush=True)
        return np.concatenate(scores)
    val_scores = predict(model, splits['val'])
    np.save(OUT / 'validation_scores.npy', val_scores)
    # Keep production threshold fixed for a directly comparable evaluation.
    test_scores = predict(model, splits['test'])
    baseline_scores = predict(baseline, splits['test'])
    np.savez_compressed(OUT / 'test_predictions.npz', candidate=test_scores, baseline=baseline_scores)
    report = dict(threshold=.95, histories=histories, selected_phase=selected_phase, trainable_visual_layers=trainable_names,
                  candidate_sha256=hashlib.sha256((OUT / 'field_candidate.keras').read_bytes()).hexdigest(),
                  baseline_sha256=hashlib.sha256(original_path.read_bytes()).hexdigest(),
                  manifest_sha256=hashlib.sha256((OUT / 'manifest.json').read_bytes()).hexdigest(),
                  candidate=evaluate(splits['test'], test_scores, names),
                  baseline=evaluate(splits['test'], baseline_scores, baseline_names), by_source={})
    for source_name in sorted({r['source'] for r in splits['test']}):
        mask = np.array([r['source'] == source_name for r in splits['test']])
        subset = [r for r in splits['test'] if r['source'] == source_name]
        report['by_source'][source_name] = dict(candidate=evaluate(subset, test_scores[mask], names),
                                               baseline=evaluate(subset, baseline_scores[mask], baseline_names))
    report['limitations'] = json.loads((OUT / 'dataset_summary.json').read_text())['limitations']
    (OUT / 'field_evaluation.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({s: {k: {m: v for m, v in result.items() if m not in ('per_class', 'confusion')} for k, result in r.items()}
                      for s, r in report['by_source'].items()}, indent=2), flush=True)


if __name__ == '__main__':
    main()
