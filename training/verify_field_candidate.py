"""Exercise candidate uploads in an isolated Flask instance; do not infer ground truth."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'instance/field_improvement'
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--with-locator', action='store_true')
    args = parser.parse_args()
    os.environ['MODEL_PATH'] = str(OUT / 'field_candidate.keras')
    os.environ['RUNTIME_CLASS_INDICES_PATH'] = str(OUT / 'class_indices.json')
    os.environ['DATABASE_PATH'] = str(OUT / 'route_verification.db')
    os.environ['DISEASE_CONFIDENCE_THRESHOLD'] = '.95'
    os.environ['CONFIDENCE_THRESHOLD'] = '.70'
    os.environ['PREDICTION_MARGIN_THRESHOLD'] = '.20'
    if args.with_locator:
        os.environ['LEAF_LOCATOR_PATH'] = str(OUT / 'final_leaf_locator.keras')
        os.environ['LEAF_LOCATOR_THRESHOLD'] = str(json.loads((OUT / 'final_locator_evaluation.json').read_text())['threshold'])
    else:
        os.environ.pop('LEAF_LOCATOR_PATH', None)
    import app as application
    assert application.MODEL is not None
    assert application.active_model_path().resolve() == (OUT / 'field_candidate.keras').resolve()
    assert len(application.CLASS_NAMES) == len(json.loads((OUT / 'class_indices.json').read_text()))
    assert application.MODEL_IMG_SIZE == (224, 224)
    results = []
    with tempfile.TemporaryDirectory(dir=OUT) as temporary:
        application.UPLOAD_FOLDER = Path(temporary)
        application.app.config.update(UPLOAD_FOLDER=temporary, TESTING=True)
        client = application.app.test_client()
        for filename in ('images (1).jpg', 'images (3).jpg', 'images (4).jpg', 'tomato-disease-11-TSWV.jpg'):
            path = Path.home() / 'Downloads' / filename
            if not path.exists():
                continue
            response = client.post('/upload', data={'file': (io.BytesIO(path.read_bytes()), filename)},
                                   content_type='multipart/form-data', follow_redirects=True)
            assert response.status_code == 200
            with client.session_transaction() as session:
                saved = session['last_filename']
            label, confidence, treatment = application.processing(saved)
            assert not label.startswith('Could not process image:')
            assert label in application.CLASS_NAMES + [application.UNCERTAIN_IMAGE_LABEL, application.INVALID_IMAGE_LABEL]
            results.append(dict(file=filename, http_status=200, label=label, confidence=confidence,
                                message=treatment, disease_ground_truth=None))
        assert client.get('/').status_code == 200
        assert client.get('/input').status_code == 200
    report = dict(with_locator=args.with_locator, results=results,
                  candidate_sha256=hashlib.sha256((OUT / 'field_candidate.keras').read_bytes()).hexdigest(),
                  note='These are route checks and recorded predictions, not verified disease diagnoses.')
    suffix = '_with_locator' if args.with_locator else ''
    (OUT / f'upload_verification{suffix}.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
