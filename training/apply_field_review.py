"""Apply recorded visual suitability decisions without inventing disease labels."""
import json
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'instance/field_improvement'


def main():
    backup = OUT / 'field_unreviewed_manifest.json'
    if not backup.exists():
        backup.write_bytes((OUT / 'manifest.json').read_bytes())
    rows = json.loads(backup.read_text())
    review = json.loads((ROOT / 'training/field_visual_review.json').read_text())
    decisions = {r['archive_member']: r for r in review['records']}
    source_rows = [r for r in rows if r['source'] == 'PlantWildV2']
    if set(decisions) != {r['archive_member'] for r in source_rows}:
        raise RuntimeError('Source images changed; repeat visual review before training.')
    kept, excluded = [], []
    for row in rows:
        if row['source'] != 'PlantWildV2':
            kept.append(row)
            continue
        decision = decisions[row['archive_member']]
        if row['pixel_sha256'] != decision['pixel_sha256']:
            raise RuntimeError('Reviewed image pixels changed.')
        action = decision['decision']
        if action == 'exclude':
            excluded.append(dict(path=row['path'], archive_member=row['archive_member'], reason='Mixed subjects, composite/annotated image, or uncertain suitability'))
            continue
        if action not in ('leaf_photo', 'non_leaf_subject'):
            raise ValueError('Unknown review action')
        if action == 'non_leaf_subject':
            row['label'] = 'Not_A_Crop'
            row['label_review'] = 'AI visual review: fruit/stem subject; not a usable leaf diagnosis photo. Original disease label is not used.'
        row['suitability_review'] = action
        row['disease_independently_verified'] = False
        kept.append(row)
    names = sorted({r['label'] for r in kept})
    for name in names:
        for split in ('train', 'val', 'test'):
            if not any(r['label'] == name and r['split'] == split for r in kept):
                raise RuntimeError(f'Review left {name} without {split} examples')
    (OUT / 'manifest.json').write_text(json.dumps(kept, indent=2))
    (OUT / 'class_indices.json').write_text(json.dumps({n: i for i, n in enumerate(names)}, indent=2))
    (OUT / 'visual_excluded.json').write_text(json.dumps(excluded, indent=2))
    report = json.loads((OUT / 'dataset_summary.json').read_text())
    prior_excluded = report.get('pre_review_excluded', report['excluded'])
    report.update(pre_review_excluded=prior_excluded, excluded=prior_excluded + len(excluded))
    report.update(images=len(kept), by_source=dict(Counter(r['source'] for r in kept)),
                  partitions=dict(Counter(r['split'] for r in kept)),
                  visual_review=dict(Counter(r['decision'] for r in review['records'])),
                  new_field_per_class={n: dict(Counter(r['split'] for r in kept if r['source']=='PlantWildV2' and r['label']==n)) for n in names})
    report['limitations'][0] = 'AI visual suitability review removed mixed/composite examples and relabeled fruit/stem subjects as Not_A_Crop; it is not independent plant-pathologist disease verification.'
    (OUT / 'dataset_summary.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k not in ('classes', 'new_field_per_class', 'limitations')}, indent=2))


if __name__ == '__main__':
    main()
