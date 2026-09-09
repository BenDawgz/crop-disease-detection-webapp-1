"""Prepare the user's augmented dataset without modifying source images."""
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
import re
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from training.prepare_public_data import hashes, split_for

PUBLIC = ROOT / 'instance/public_retraining'
OUT = ROOT / 'instance/local_retraining'


def original_identifier(filename):
    stem = Path(filename).stem.split('___')[-1]
    stem = re.sub(r'_(?:(?:new)?\d+deg(?:FlipLR)?|flipLR|flipTB|final_masked)', '', stem, flags=re.I)
    return stem.split('copy')[0].strip().lower()


def group_for(filename, label, leaf_map):
    identifier = original_identifier(filename)
    options = leaf_map.get(identifier, [])
    if isinstance(options, str):
        options = [options]
    return str(next((x for x in options if label in x), options[0] if len(options) == 1 else identifier))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    public = json.loads((PUBLIC / 'manifest.json').read_text())
    leaf_map = json.loads((PUBLIC / 'pv_leaf_map.json').read_text())
    local_root = ROOT / 'training/dataset'
    labels = {d.name for d in local_root.iterdir() if d.is_dir()}
    assert labels and all(label.startswith(('Corn_', 'Grape_', 'Tomato_')) for label in labels)
    # Preserve the previous public held-out groups for a directly comparable benchmark.
    reserved = {r['group']: r['split'] for r in public if r['source'] == 'PlantVillage'}
    retained = [r for r in public if not (r['source'] == 'PlantVillage' and r['label'] in labels and r['split'] == 'train')]
    held_public_groups = {r['group'] for r in retained if r['source'] == 'PlantVillage' and r['split'] != 'train'}
    user_hashes = []
    for filename in ('images (1).jpg', 'images (3).jpg', 'images (4).jpg', 'tomato-disease-11-TSWV.jpg'):
        with Image.open(Path.home() / 'Downloads' / filename) as image:
            user_hashes.append(hashes(image))
    groups = defaultdict(list)
    excluded, blocked = [], set()
    files = sorted(p for p in local_root.rglob('*') if p.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp', '.webp'))
    for i, path in enumerate(files):
        label = path.parent.name
        group = group_for(path.name, label, leaf_map)
        row = dict(source='UserDataset', label=label, group=group, path=str(path.relative_to(ROOT)))
        try:
            with Image.open(path) as image:
                im = image.convert('RGB')
                hs = hashes(im)
                exact = hashlib.sha256(im.tobytes()).hexdigest()
            if any(min((hs[0] ^ x[0]).bit_count(), (hs[1] ^ x[1]).bit_count()) <= 6 for x in user_hashes):
                blocked.add(group)
            groups[group].append(dict(row, pixel_sha256=exact, visual_hashes=hs))
        except Exception as error:
            excluded.append(dict(row, reason=str(error)))
        if (i + 1) % 2000 == 0:
            print(f'Checking local images: {i + 1}/{len(files)}', flush=True)
    for group, items in groups.items():
        if group in blocked:
            excluded.extend(dict(r, reason='group similar to supplied failure example') for r in items)
            continue
        if len({r['label'] for r in items}) > 1:
            excluded.extend(dict(r, reason='conflicting class labels for leaf group') for r in items)
            continue
        split = reserved.get(group, split_for('pv:' + group))
        if group in held_public_groups:
            excluded.extend(dict(r, reason='public held-out leaf group retained separately') for r in items)
        elif split == 'train':
            retained.extend(dict(r, split=split) for r in items)
        else:
            # One view per held-out leaf, rather than scoring many augmented copies.
            items.sort(key=lambda r: (bool(re.search(r'flip|deg', r['path'], re.I)), r['path']))
            retained.append(dict(items[0], split=split))
            excluded.extend(dict(r, reason='additional augmented held-out view') for r in items[1:])
    seen, accepted = set(), []
    for row in sorted(retained, key=lambda r: ({'test': 0, 'val': 1, 'train': 2}[r['split']], r['path'])):
        exact = row['pixel_sha256']
        if exact in seen:
            excluded.append(dict(row, reason='exact duplicate'))
        else:
            seen.add(exact)
            accepted.append(row)
    split_groups = defaultdict(set)
    for row in accepted:
        if row['source'] in ('PlantVillage', 'UserDataset'):
            split_groups[row['group']].add(row['split'])
    assert all(len(splits) == 1 for splits in split_groups.values())
    summary = dict(local_images_found=len(files), counts=dict(Counter(r['split'] for r in accepted)),
        sources=dict(Counter(r['source'] for r in accepted)), excluded=len(excluded),
        class_counts={label: dict(Counter(r['split'] for r in accepted if r['label'] == label)) for label in sorted(set(r['label'] for r in accepted))})
    (OUT / 'manifest.json').write_text(json.dumps(accepted, indent=2))
    (OUT / 'excluded.json').write_text(json.dumps(excluded, indent=2))
    (OUT / 'dataset_summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
