"""Merge a bounded PlantWild v2 crop subset with existing grouped partitions.

The authors report expert refinement, not laboratory confirmation per photo.
Public image grouping cannot establish independent plants/farms/photo sessions.
"""
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
import sys
import zipfile
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'instance/field_improvement'
sys.path.insert(0, str(ROOT))
from training.prepare_public_data import hashes, split_for

LABELS = {
    'corn gray leaf spot': 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'corn northern leaf blight': 'Corn_(maize)___Northern_Leaf_Blight',
    'corn rust': 'Corn_(maize)___Common_rust_',
    'grape black rot': 'Grape___Black_rot',
    'grape downy mildew': 'Grape___Downy_mildew',
    'grapevine leafroll disease': 'Grape___Leafroll_disease',
    'tomato bacterial leaf spot': 'Tomato___Bacterial_spot',
    'tomato early blight': 'Tomato___Early_blight',
    'tomato late blight': 'Tomato___Late_blight',
    'tomato leaf mold': 'Tomato___Leaf_Mold',
    'tomato mosaic virus': 'Tomato___Tomato_mosaic_virus',
    'tomato septoria leaf spot': 'Tomato___Septoria_leaf_spot',
    'tomato yellow leaf curl virus': 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
}


def similar(visual, references):
    if not len(references):
        return np.zeros(0, dtype=bool)
    distances = np.bitwise_count(references ^ np.asarray(visual, dtype=np.uint64))
    return (distances[:, 0] <= 5) & (distances[:, 1] <= 5)


def main():
    old = json.loads((ROOT / 'instance/local_retraining/manifest.json').read_text())
    reference_hashes = np.asarray([r['visual_hashes'] for r in old], dtype=np.uint64)
    reserved = []
    for filename in ('images (1).jpg', 'images (3).jpg', 'images (4).jpg', 'tomato-disease-11-TSWV.jpg'):
        path = Path.home() / 'Downloads' / filename
        if path.exists():
            with Image.open(path) as image:
                reserved.append(hashes(image.convert('RGB')))
    if reserved:
        reference_hashes = np.concatenate([reference_hashes, np.asarray(reserved, dtype=np.uint64)])
    old_pixels = {r['pixel_sha256'] for r in old}
    observed = defaultdict(set)
    for row in old:
        observed[(row['source'], row['group'])].add(row['split'])
    ambiguous = {key for key, values in observed.items() if len(values) > 1}
    excluded = [dict(image=r['path'], reason='Ambiguous legacy group crosses partitions') for r in old
                if (r['source'], r['group']) in ambiguous]
    old = [r for r in old if (r['source'], r['group']) not in ambiguous]
    # Keep every held-out legacy row. Cap lab training to unique leaf groups.
    selected, grouped = [], defaultdict(list)
    for row in old:
        if row['split'] != 'train' or row['source'] in ('PlantDoc', 'Imagenette'):
            selected.append(row)
        else:
            grouped[row['label']].append(row)
    for label, rows in grouped.items():
        groups = set()
        for row in sorted(rows, key=lambda r: hashlib.sha256(r['path'].encode()).hexdigest()):
            if row['group'] in groups:
                continue
            selected.append(row)
            groups.add(row['group'])
            if len(groups) >= 120:
                break
    out_images = OUT / 'field_images'
    out_images.mkdir(exist_ok=True, parents=True)
    candidates = []
    with zipfile.ZipFile(OUT / 'plantwild_v2.zip') as archive:
        for name in sorted(archive.namelist()):
            parts = name.split('/')
            if len(parts) != 3 or parts[1] not in LABELS or Path(name).suffix.lower() not in ('.jpg', '.jpeg', '.png'):
                continue
            with archive.open(name) as stream, Image.open(stream) as image:
                im = image.convert('RGB')
                pixel = hashlib.sha256(im.tobytes()).hexdigest()
                visual = hashes(im)
                if pixel in old_pixels or similar(visual, reference_hashes).any():
                    excluded.append(dict(image=name, reason='Exact/perceptual overlap with legacy data or reserved user examples'))
                    continue
                target = out_images / (hashlib.sha256(name.encode()).hexdigest() + '.jpg')
                im.save(target, quality=95)
            candidates.append(dict(source='PlantWildV2', label=LABELS[parts[1]], original_label=parts[1],
                                   path=str(target.relative_to(ROOT)), group=name, archive_member=name,
                                   pixel_sha256=pixel, visual_hashes=visual,
                                   label_review='Authors report expert-refined dataset; not independently reviewed here',
                                   license='CC-BY-NC-ND-4.0', plant_id=None, session_id=None))
    # Connected components ensure chains of near duplicates stay together.
    parent = list(range(len(candidates)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    all_hashes = np.asarray([r['visual_hashes'] for r in candidates], dtype=np.uint64)
    for i, row in enumerate(candidates):
        for j in np.flatnonzero(similar(row['visual_hashes'], all_hashes[:i])):
            parent[find(i)] = find(int(j))
    clusters = defaultdict(list)
    for i, row in enumerate(candidates):
        clusters[find(i)].append(row)
    for group in clusters.values():
        if len({r['label'] for r in group}) > 1:
            excluded.extend(dict(image=r['archive_member'], reason='Near-duplicate cluster has conflicting labels') for r in group)
            continue
        # One representative per visual group, so repeated copies don't inflate the test.
        row = min(group, key=lambda r: r['archive_member'])
        row['group'] = row['archive_member']
        row['split'] = split_for('pw2:' + row['group'])
        selected.append(row)
        excluded.extend(dict(image=r['archive_member'], reason='Additional near-duplicate view') for r in group if r is not row)
    names = sorted({r['label'] for r in selected})
    for label in names:
        for split in ('train', 'val', 'test'):
            if not any(r['label'] == label and r['split'] == split for r in selected):
                raise RuntimeError(f'Missing partition for {label}: {split}; review before training')
    (OUT / 'manifest.json').write_text(json.dumps(selected, indent=2))
    (OUT / 'field_unreviewed_manifest.json').write_text(json.dumps(selected, indent=2))
    (OUT / 'class_indices.json').write_text(json.dumps({n: i for i, n in enumerate(names)}, indent=2))
    (OUT / 'field_excluded.json').write_text(json.dumps(excluded, indent=2))
    report = dict(images=len(selected), classes=names, excluded=len(excluded),
                  by_source=dict(Counter(r['source'] for r in selected)),
                  partitions=dict(Counter(r['split'] for r in selected)),
                  new_field_per_class={n: dict(Counter(r['split'] for r in selected if r['source']=='PlantWildV2' and r['label']==n)) for n in names},
                  limitations=['Web photos can include non-leaf organs; per-image leaf review remains required.',
                               'Perceptual filtering reduces overlap; does not prove independence of plants/sessions.',
                               'Legacy test is reused; new public test is separate from fitting, not a prospective field trial.',
                               'Spotted wilt, leaf-miner damage and nutrient deficiency remain unfilled collection gaps.'])
    (OUT / 'dataset_summary.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
