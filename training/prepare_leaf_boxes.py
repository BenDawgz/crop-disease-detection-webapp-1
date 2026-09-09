"""Prepare original PlantDoc images with largest-leaf boxes and non-leaf negatives.

Preserves existing PlantDoc/Imagenette splits; the benchmark is reused, not an
independent field test. Published bounding boxes are not expert disease review.
"""
import csv
import hashlib
import io
import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
from urllib.parse import quote, unquote
import requests
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from training.prepare_public_data import hashes, split_for

OUT = ROOT / 'instance/field_improvement'
REV = '4730a233a555b30ee98e0879c63ad25d82407455'
BASE = f'https://raw.githubusercontent.com/pratikkayal/PlantDoc-Object-Detection-Dataset/{REV}/'


def main():
    OUT.mkdir(exist_ok=True, parents=True)
    (OUT / 'box_images').mkdir(exist_ok=True)
    old = json.loads((ROOT / 'instance/local_retraining/manifest.json').read_text())
    example_hashes = []
    for filename in ('images (1).jpg', 'images (3).jpg', 'images (4).jpg', 'tomato-disease-11-TSWV.jpg'):
        example = Path.home() / 'Downloads' / filename
        if example.exists():
            with Image.open(example) as image:
                example_hashes.append(hashes(image.convert('RGB')))
    splits = defaultdict(set)
    for row in old:
        if row['source'] == 'PlantDoc':
            splits[unquote(row['group']).casefold()].add(row['split'])
    tasks = []
    for part in ('train', 'test'):
        path = OUT / f'{part}_labels.csv'
        if not path.exists():
            response = requests.get(BASE + path.name, timeout=40)
            response.raise_for_status()
            path.write_bytes(response.content)
        grouped = defaultdict(list)
        for row in csv.DictReader(path.open(encoding='utf-8-sig')):
            grouped[row['filename']].append(row)
        for filename, annotations in grouped.items():
            existing = splits[unquote(filename).casefold()]
            # Conflicting legacy split assignments cannot enter training.
            if len(existing) > 1:
                continue
            split = next(iter(existing)) if existing else ('test' if part == 'test' else split_for('pd:' + filename))
            tasks.append((part, filename, annotations, split))

    def download(task):
        part, filename, annotations, split = task
        url = BASE + part.upper() + '/' + quote(filename, safe='')
        target = OUT / 'box_images' / (hashlib.sha256(url.encode()).hexdigest() + '.jpg')
        try:
            if not target.exists():
                response = requests.get(url, timeout=(15, 45))
                response.raise_for_status()
                with Image.open(io.BytesIO(response.content)) as im:
                    # Coordinates refer to stored pixels, so do not rotate annotated images.
                    image = im.convert('RGB')
                    image.thumbnail((1024, 1024))
                    image.save(target, quality=95)
            with Image.open(target) as image:
                visual = hashes(image)
                pixel = hashlib.sha256(image.convert('RGB').tobytes()).hexdigest()
            if any(all((a ^ b).bit_count() <= 8 for a, b in zip(visual, known)) for known in example_hashes):
                return dict(excluded=filename, reason='Overlap with user failure example reserved for review')
            boxes = []
            for row in annotations:
                w, h = float(row['width']), float(row['height'])
                if w <= 0 or h <= 0:
                    continue
                box = [max(0., min(1., float(row[key]) / scale)) for key, scale in
                       zip(('xmin', 'ymin', 'xmax', 'ymax'), (w, h, w, h))]
                if box[2] > box[0] and box[3] > box[1]:
                    boxes.append(box)
            if not boxes:
                raise ValueError('No valid annotation')
            largest = max(boxes, key=lambda b: (b[2]-b[0]) * (b[3]-b[1]))
            return dict(path=str(target.relative_to(ROOT)), source='PlantDocBoxes',
                        split=split, group=unquote(filename).casefold(), box=largest,
                        boxes=boxes, leaf=1, url=url, pixel_sha256=pixel, visual_hashes=visual)
        except (requests.RequestException, OSError, ValueError) as error:
            return dict(excluded=filename, reason=str(error))

    results = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        for i, result in enumerate(pool.map(download, tasks)):
            results.append(result)
            if i % 100 == 0:
                print(f'Leaf annotations/images: {i}/{len(tasks)}', flush=True)
    rows, excluded = [], []
    for result in results:
        if 'excluded' in result:
            excluded.append(result)
        else:
            rows.append(result)
    # If duplicate originals span partitions, exclude every occurrence.
    by_pixel = defaultdict(set)
    for row in rows:
        by_pixel[row['pixel_sha256']].add(row['split'])
    seen = set()
    clean = []
    for row in rows:
        key = row['pixel_sha256']
        if len(by_pixel[key]) > 1 or key in seen:
            excluded.append(dict(excluded=row['path'], reason='Duplicate or conflicting partition'))
        else:
            seen.add(key)
            clean.append(row)
    clean.extend(dict(row, box=[0, 0, 0, 0], boxes=[], leaf=0) for row in old if row['source'] == 'Imagenette')
    (OUT / 'box_manifest.json').write_text(json.dumps(clean, indent=2))
    (OUT / 'box_excluded.json').write_text(json.dumps(excluded, indent=2))
    summary = dict(images=len(clean), excluded=len(excluded), partitions=dict(Counter(r['split'] for r in clean)),
                   note='Largest annotated leaf only; public reused benchmark; Imagenette overlaps backbone pretraining.')
    (OUT / 'box_summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
