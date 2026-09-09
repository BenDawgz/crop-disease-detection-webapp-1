"""Exercise real Flask uploads against the candidate without changing active data."""
import io
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get('RETRAIN_EXPERIMENT_DIR', ROOT / 'instance' / 'public_retraining'))
sys.path.insert(0, str(ROOT))
os.environ['MODEL_PATH'] = str(OUT / os.environ.get('RETRAIN_CANDIDATE_FILE', 'candidate.h5'))
os.environ['RUNTIME_CLASS_INDICES_PATH'] = str(OUT / 'class_indices.json')
os.environ['DATABASE_PATH'] = str(OUT / 'smoke_test.db')
os.environ['DISEASE_CONFIDENCE_THRESHOLD'] = '0.95'
os.environ['CONFIDENCE_THRESHOLD'] = '0.70'
os.environ['PREDICTION_MARGIN_THRESHOLD'] = '0.20'

import app as application


def main():
    assert application.MODEL is not None
    assert len(application.CLASS_NAMES) == 20
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
            html = response.get_data(as_text=True)
            assert response.status_code == 200
            assert 'data-label="Uncertain image"' in html
            assert 'A leaf was recognized' in html
            results.append({'file': filename, 'http_status': response.status_code,
                            'result': 'Leaf recognized; condition uncertain'})
        assert client.get('/').status_code == 200
        assert client.get('/input').status_code == 200
        assert client.get('/admin').status_code == 404
        assert client.get('/history').status_code == 200
    (OUT / 'upload_verification.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
