"""Compare whole-photo and located-leaf diagnosis without changing the runtime."""
import json
import argparse
import os
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'instance/field_improvement'
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
sys.path.insert(0, str(ROOT))
import numpy as np
from PIL import Image
import tensorflow as tf
from leaf_locator import crop_leaf, valid_box
from training.finetune_field_model import evaluate
from training.train_public_model import batches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--locator-variant', default='')
    args = parser.parse_args()
    prefix = args.locator_variant + '_' if args.locator_variant else ''
    tf.config.threading.set_intra_op_parallelism_threads(8)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    rows = [r for r in json.loads((OUT / 'manifest.json').read_text()) if r['split'] == 'test']
    locator = tf.keras.models.load_model(OUT / f'{prefix}leaf_locator.keras', compile=False)
    classifier = tf.keras.models.load_model(OUT / 'field_candidate.keras', compile=False)
    mapping = json.loads((OUT / 'class_indices.json').read_text())
    names = sorted(mapping, key=mapping.get)
    threshold = json.loads((OUT / f'{prefix}locator_evaluation.json').read_text())['threshold']
    raw_scores, crop_scores, detected = [], [], []
    for start in range(0, len(rows), 24):
        subset = rows[start:start+24]
        pixels = next(batches(subset, 224))
        located = locator(pixels, training=False)
        originals = classifier(pixels, training=False).numpy()
        crop_pixels = pixels.copy()
        found = []
        for i, row in enumerate(subset):
            box = np.asarray(located['box'])[i]
            score = float(np.asarray(located['leaf'])[i, 0])
            accepted = np.isfinite(score) and score >= threshold and valid_box(box)
            found.append(accepted)
            if accepted:
                with Image.open(ROOT / row['path']) as image:
                    cropped = crop_leaf(image.convert('RGB'), box).resize((224, 224), Image.Resampling.NEAREST)
                    crop_pixels[i] = np.asarray(cropped, dtype=np.float32) / 255
        scores = classifier(crop_pixels, training=False).numpy()
        # A failed locator leads to abstention, never a forced disease prediction.
        scores[~np.array(found)] = 1. / len(names)
        raw_scores.extend(originals)
        crop_scores.extend(scores)
        detected.extend(found)
        if start % 480 == 0:
            print(f'Two-stage evaluation: {start}/{len(rows)}', flush=True)
    raw_scores, crop_scores = np.asarray(raw_scores), np.asarray(crop_scores)
    np.savez_compressed(OUT / f'{prefix}two_stage_predictions.npz', whole_image=raw_scores,
                        cropped=crop_scores, detected=np.asarray(detected, dtype=bool))
    report = dict(locator_threshold=threshold, disease_threshold=.95, locator_found_count=int(sum(detected)),
                  whole_image=evaluate(rows, raw_scores, names),
                  two_stage=evaluate(rows, crop_scores, names, unlocalized=~np.asarray(detected, dtype=bool)), by_source={})
    for source in sorted({r['source'] for r in rows}):
        mask = np.array([r['source'] == source for r in rows])
        subset = [r for r in rows if r['source'] == source]
        report['by_source'][source] = dict(whole_image=evaluate(subset, raw_scores[mask], names),
                                           two_stage=evaluate(subset, crop_scores[mask], names,
                                                              unlocalized=~np.asarray(detected, dtype=bool)[mask]))
    report['note'] = 'No automatic activation. Public web benchmark; independent reviewed field testing remains outstanding.'
    (OUT / f'{prefix}two_stage_evaluation.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k not in ('whole_image', 'two_stage', 'by_source')}, indent=2))


if __name__ == '__main__':
    main()
