import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import requests
from PIL import Image
from crop_vision import CropVision, AnalysisError


def sample():
    return dict(status='possible_condition', crop='tomato', condition='leaf miner damage',
                symptoms=['Winding pale trails'], possible_causes=['Leaf miner damage'],
                explanation='Visual evidence is provisional.', next_steps=['Inspect the underside.'])


class VisionTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.photo = self.root / 'photo.jpg'
        Image.new('RGB', (100, 100), 'green').save(self.photo)
        self.vision = CropVision(self.root)
        self.env = patch.dict(os.environ, {'OPENAI_API_KEY': 'test-secret'})
        self.env.start()
        self.addCleanup(self.env.stop)

    def response(self, data=None):
        return Mock(status_code=200, json=Mock(return_value={'status': 'completed', 'output': [
            {'type': 'message', 'content': [{'type': 'output_text', 'text': json.dumps(data or sample())}]}]}))

    @patch('crop_vision.requests.post')
    def test_image_payload_and_persistent_cache(self, post):
        post.return_value = self.response()
        self.assertEqual(self.vision.analyze(self.photo), sample())
        self.assertEqual(CropVision(self.root).analyze(self.photo), sample())
        post.assert_called_once()
        kwargs = post.call_args.kwargs
        self.assertFalse(kwargs['json']['store'])
        self.assertTrue(kwargs['json']['text']['format']['strict'])
        self.assertTrue(kwargs['json']['input'][0]['content'][1]['image_url'].startswith('data:image/jpeg;base64,'))
        self.assertNotIn('test-secret', json.dumps(kwargs['json']))

    @patch('crop_vision.requests.post')
    def test_corrupt_image_never_sent(self, post):
        self.photo.write_text('broken')
        with self.assertRaises(AnalysisError):
            self.vision.analyze(self.photo)
        post.assert_not_called()

    @patch('crop_vision.requests.post')
    def test_missing_key_never_sent(self, post):
        with patch.dict(os.environ, {'OPENAI_API_KEY': ''}):
            with self.assertRaisesRegex(AnalysisError, 'not configured'):
                self.vision.analyze(self.photo)
        post.assert_not_called()

    @patch('crop_vision.requests.post')
    def test_failures_are_sanitized_not_cached(self, post):
        for status in [401, 403, 429, 500, 302]:
            post.return_value = Mock(status_code=status, text='test-secret')
            with self.assertRaises(AnalysisError) as error:
                self.vision.analyze(self.photo)
            self.assertNotIn('test-secret', str(error.exception))
        post.side_effect = requests.Timeout('test-secret')
        with self.assertRaises(AnalysisError) as error:
            self.vision.analyze(self.photo)
        self.assertNotIn('test-secret', str(error.exception))
        self.assertFalse((self.root / 'vision_cache').exists())

    @patch('crop_vision.requests.post')
    def test_invalid_and_refused_responses(self, post):
        for body in [{'status': 'incomplete'}, {'status': 'completed', 'output': [
            {'type': 'message', 'content': [{'type': 'refusal', 'refusal': 'No'}]}]}]:
            post.return_value = Mock(status_code=200, json=Mock(return_value=body))
            with self.assertRaises(AnalysisError):
                self.vision.analyze(self.photo)
        data = sample(); data['crop'] = 'other'
        post.return_value = self.response(data)
        with self.assertRaises(AnalysisError):
            self.vision.analyze(self.photo)


class PageTests(unittest.TestCase):
    def test_upload_display_and_errors_without_local_model(self):
        import app as module
        self.assertEqual(module.ANALYSIS_PROVIDER, 'openai')
        self.assertIsNone(module.MODEL)
        module.app.config['TESTING'] = True
        data = sample(); data['explanation'] = '<script>alert(1)</script>'
        with tempfile.TemporaryDirectory() as folder, patch.object(module, 'UPLOAD_FOLDER', Path(folder)), patch.object(module, 'log_prediction'), patch.object(module.VISION, 'analyze', return_value=data):
            client = module.app.test_client()
            image = io.BytesIO(); Image.new('RGB', (50, 50)).save(image, format='PNG'); image.seek(0)
            result = client.post('/upload', data={'file': (image, 'leaf.png')}, follow_redirects=True)
            text = result.get_data(as_text=True)
            self.assertEqual(result.status_code, 200)
            self.assertIn('Possible leaf miner damage', text)
            self.assertIn('&lt;script&gt;alert(1)&lt;/script&gt;', text)
            self.assertNotIn('const guide', text)
            self.assertNotIn('id="bar"', text)
            with patch.object(module.VISION, 'analyze', side_effect=AnalysisError('Please configure the key.')):
                text = client.get('/upload').get_data(as_text=True)
                self.assertIn('Please configure the key.', text)


if __name__ == '__main__':
    unittest.main()
