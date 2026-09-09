"""Download a reproducible, bounded public-data experiment (no model activation)."""
import hashlib
import io
import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import tarfile
from urllib.parse import quote

import numpy as np
from PIL import Image, ImageOps
import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'instance' / 'public_retraining'
OUT.mkdir(parents=True, exist_ok=True)
PV = 'spMohanty/PlantVillage-Dataset'
PD = 'pratikkayal/PlantDoc-Dataset'
UNKNOWN = 'Unsupported_Leaf'
NEGATIVE = 'Not_A_Crop'
PD_MAP = {
    'Corn Gray leaf spot': 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn leaf blight': 'Corn_(maize)___Northern_Leaf_Blight',
    'Corn rust leaf': 'Corn_(maize)___Common_rust_',
    'Tomato Early blight leaf': 'Tomato___Early_blight',
    'Tomato Septoria leaf spot': 'Tomato___Septoria_leaf_spot',
    'Tomato leaf': 'Tomato___healthy',
    'Tomato leaf bacterial spot': 'Tomato___Bacterial_spot',
    'Tomato leaf late blight': 'Tomato___Late_blight',
    'Tomato leaf mosaic virus': 'Tomato___Tomato_mosaic_virus',
    'Tomato leaf yellow virus': 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato mold leaf': 'Tomato___Leaf_Mold',
    'Tomato two spotted spider mites leaf': 'Tomato___Spider_mites Two-spotted_spider_mite',
    'grape leaf': 'Grape___healthy',
    'grape leaf black rot': 'Grape___Black_rot',
}


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def get(url):
    response = requests.get(url, timeout=90)
    response.raise_for_status()
    return response


def cached_json(name, url):
    path = OUT / name
    if not path.exists():
        path.write_text(json.dumps(get(url).json()), encoding='utf-8')
    return json.loads(path.read_text(encoding='utf-8'))


def split_for(group):
    value = int(digest(group)[:8], 16) % 100
    return 'train' if value < 70 else 'val' if value < 85 else 'test'


def leaf_identifier(filename):
    return Path(filename).stem.replace('_final_masked', '').split('___')[-1].split('copy')[0].strip().lower()


def hashes(im):
    gray = im.convert('L')
    a = np.asarray(gray.resize((8, 8)), dtype=np.float32)
    d = np.asarray(gray.resize((9, 8)), dtype=np.float32)
    return [int(''.join('1' if x else '0' for x in bits.ravel()), 2)
            for bits in (a > a.mean(), d[:, 1:] > d[:, :-1])]


def main():
    tasks = []
    pv_commit = cached_json('pv_commit.json', f'https://api.github.com/repos/{PV}/commits/master')['sha']
    pd_commit = cached_json('pd_commit.json', f'https://api.github.com/repos/{PD}/commits/master')['sha']
    pv_raw = cached_json('pv_raw.json', f'https://api.github.com/repos/{PV}/contents/raw?ref={pv_commit}')
    color_sha = next(x['sha'] for x in pv_raw if x['name'] == 'color')
    pv_tree = cached_json('pv_tree.json', f'https://api.github.com/repos/{PV}/git/trees/{color_sha}?recursive=1')
    if pv_tree.get('truncated'):
        directories = cached_json('pv_color_dirs.json', f'https://api.github.com/repos/{PV}/contents/raw/color?ref={pv_commit}')
        entries = []
        for directory in directories:
            if not directory['name'].startswith(('Corn_', 'Grape_', 'Tomato_')):
                continue
            subtree = cached_json('pv_' + directory['sha'] + '.json',
                                  f'https://api.github.com/repos/{PV}/git/trees/{directory["sha"]}')
            assert not subtree.get('truncated'), 'Incomplete class listing'
            entries.extend(dict(item, path=directory['name'] + '/' + item['path']) for item in subtree['tree'])
        pv_tree = {'tree': entries}
    leaf_map = cached_json('pv_leaf_map.json', f'https://raw.githubusercontent.com/{PV}/{pv_commit}/leaf_grouping/leaf-map.json')
    groups = defaultdict(list)
    for item in pv_tree['tree']:
        parts = item['path'].split('/')
        if item['type'] != 'blob' or len(parts) != 2 or not parts[0].startswith(('Corn_', 'Grape_', 'Tomato_')):
            continue
        label, filename = parts
        # Strip the extension BEFORE the copy marker: names such as
        # "R.S_HL 0606 copy.JPG" must retain their internal dot and leaf ID.
        identifier = leaf_identifier(filename)
        suggestions = leaf_map.get(identifier, [])
        if isinstance(suggestions, str):
            suggestions = [suggestions]
        group = next((x for x in suggestions if label in x), suggestions[0] if len(suggestions) == 1 else identifier)
        split = split_for('pv:' + str(group))
        path = 'raw/color/' + item['path']
        groups[(label, split)].append(dict(source='PlantVillage', label=label, split=split,
            group=str(group), url=f'https://raw.githubusercontent.com/{PV}/{pv_commit}/{quote(path)}'))
    for (label, split), rows in sorted(groups.items()):
        limit = 120 if split == 'train' else 30
        tasks.extend(sorted(rows, key=lambda r: digest(r['url']))[:limit])

    pd_tree = cached_json('pd_tree.json', f'https://api.github.com/repos/{PD}/git/trees/{pd_commit}?recursive=1')
    assert not pd_tree.get('truncated'), 'Incomplete PlantDoc listing'
    for item in pd_tree['tree']:
        parts = item['path'].split('/')
        if item['type'] != 'blob' or len(parts) != 3 or parts[0] not in ('train', 'test'):
            continue
        label = PD_MAP.get(parts[1], UNKNOWN)
        split = 'test' if parts[0] == 'test' else ('val' if int(digest(parts[2])[:8], 16) % 5 == 0 else 'train')
        tasks.append(dict(source='PlantDoc', label=label, split=split, group=parts[2],
            url=f'https://raw.githubusercontent.com/{PD}/{pd_commit}/{quote(item["path"])}'))

    # Keep upstream attribution and metadata alongside the local experiment.
    for name, url in {
        'PlantDoc_LICENSE.txt': f'https://raw.githubusercontent.com/{PD}/{pd_commit}/LICENSE.txt',
        'PlantVillage_loader.py.txt': f'https://raw.githubusercontent.com/{PV}/{pv_commit}/plant_village.py',
        'Imagenette_README.md': 'https://raw.githubusercontent.com/fastai/imagenette/master/README.md',
    }.items():
        path = OUT / name
        if not path.exists():
            path.write_bytes(get(url).content)

    images_dir = OUT / 'images'
    images_dir.mkdir(exist_ok=True)

    def download(row):
        path = images_dir / (digest(row['url']) + '.jpg')
        try:
            if not path.exists():
                data = get(row['url']).content
                with Image.open(io.BytesIO(data)) as image:
                    im = ImageOps.exif_transpose(image).convert('RGB')
                    im.save(path, quality=95)
            return dict(row, path=str(path.relative_to(ROOT)))
        except Exception as error:
            return dict(row, error=str(error))

    print(f'Downloading {len(tasks)} leaf images', flush=True)
    rows = []
    with ThreadPoolExecutor(max_workers=12) as pool:
        for index, row in enumerate(pool.map(download, tasks)):
            rows.append(row)
            if (index + 1) % 250 == 0:
                print(f'Leaf downloads: {index + 1}/{len(tasks)}', flush=True)

    archive = OUT / 'imagenette2-160.tgz'
    archive_url = 'https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz'
    if not archive.exists():
        print('Downloading Imagenette 160px archive', flush=True)
        response = requests.get(archive_url, stream=True, timeout=90)
        response.raise_for_status()
        pending = archive.with_suffix('.partial')
        with pending.open('wb') as handle:
            for chunk in response.iter_content(1024 * 1024):
                handle.write(chunk)
        pending.replace(archive)
    with tarfile.open(archive) as tar:
        members = defaultdict(list)
        for member in tar.getmembers():
            parts = member.name.split('/')
            if member.isfile() and len(parts) == 4 and parts[1] in ('train', 'val') and parts[-1].lower().endswith(('.jpg', '.jpeg')):
                members[(parts[1], parts[2])].append(member)
        selected = []
        for (upstream_split, category), items in sorted(members.items()):
            limit = 150 if upstream_split == 'train' else 40
            for member in sorted(items, key=lambda m: digest(m.name))[:limit]:
                selected.append((member, upstream_split))
        # Read gzip members in archive order, avoiding repeated decompression.
        for member, upstream_split in sorted(selected, key=lambda pair: pair[0].offset_data):
            split = 'test' if upstream_split == 'val' else ('val' if int(digest(member.name)[:8], 16) % 5 == 0 else 'train')
            url = archive_url + '#' + member.name
            path = images_dir / (digest(url) + '.jpg')
            if not path.exists():
                with Image.open(tar.extractfile(member)) as image:
                    ImageOps.exif_transpose(image).convert('RGB').save(path, quality=95)
            rows.append(dict(source='Imagenette', label=NEGATIVE, split=split, group=member.name,
                url=url, path=str(path.relative_to(ROOT))))

    # Exclude the user's examples and perceptually close copies from ALL splits.
    user_hashes = []
    for name in ('images (1).jpg', 'images (3).jpg', 'images (4).jpg', 'tomato-disease-11-TSWV.jpg'):
        path = Path.home() / 'Downloads' / name
        if path.exists():
            with Image.open(path) as image:
                user_hashes.append(hashes(ImageOps.exif_transpose(image)))
    # Keep test, then validation first. Remove duplicate/near-duplicate training
    # images across sources/splits; this is a heuristic, not proof of independence.
    accepted, excluded, seen, visual = [], [], set(), []
    for row in sorted(rows, key=lambda r: ({'test': 0, 'val': 1, 'train': 2}[r['split']], r['url'])):
        if 'error' in row:
            excluded.append(row)
            continue
        try:
            with Image.open(ROOT / row['path']) as image:
                im = image.convert('RGB')
                hs = hashes(im)
                exact = hashlib.sha256(im.tobytes()).hexdigest()
            if any(min((hs[0] ^ x[0]).bit_count(), (hs[1] ^ x[1]).bit_count()) <= 6 for x in user_hashes):
                excluded.append(dict(row, reason='user example or perceptually similar'))
            elif exact in seen or any(max((hs[0] ^ x[0]).bit_count(), (hs[1] ^ x[1]).bit_count()) <= 3 for x in visual):
                excluded.append(dict(row, reason='duplicate or perceptually similar'))
            else:
                seen.add(exact)
                visual.append(hs)
                accepted.append(dict(row, pixel_sha256=exact, visual_hashes=hs))
        except Exception as error:
            excluded.append(dict(row, error=str(error)))
    (OUT / 'manifest.json').write_text(json.dumps(accepted, indent=2), encoding='utf-8')
    (OUT / 'excluded.json').write_text(json.dumps(excluded, indent=2), encoding='utf-8')
    summary = dict(counts=dict(Counter(r['split'] for r in accepted)),
        sources=dict(Counter(r['source'] for r in accepted)), excluded=len(excluded),
        class_counts={label: dict(Counter(r['split'] for r in accepted if r['label'] == label))
                      for label in sorted(set(r['label'] for r in accepted))})
    (OUT / 'dataset_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
