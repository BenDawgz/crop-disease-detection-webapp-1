import ast
from contextlib import nullcontext
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np
from PIL import Image

from prediction_policy import classify_scores, has_leaf_evidence


NAMES = ['Corn___healthy', 'Not_A_Crop', 'Tomato___disease']


class DecisionTests(unittest.TestCase):
    def test_confident_non_crop_is_never_replaced_by_crop(self):
        self.assertEqual(classify_scores([0.12, 0.85, 0.03], NAMES),
                         ('Not_A_Crop', 85.0))

    def test_confident_leaf_is_accepted(self):
        label, confidence = classify_scores([0.04, 0.03, 0.93], NAMES)
        self.assertEqual(label, 'Tomato___disease')
        self.assertAlmostEqual(confidence, 93.0)

    def test_ambiguous_leaf_and_non_crop_abstain_in_both_directions(self):
        for scores in ([0.48, 0.49, 0.03], [0.49, 0.48, 0.03]):
            with self.subTest(scores=scores):
                self.assertIsNone(classify_scores(scores, NAMES)[0])

    def test_margin_includes_non_crop_competitor(self):
        self.assertIsNone(classify_scores([0.55, 0.44, 0.01], NAMES,
                                         confidence_threshold=0.5)[0])

    def test_invalid_or_incompatible_output_abstains(self):
        for scores in ([np.nan, 0, 1], [np.inf, 0, 1], [0.5, 0.5],
                       [[0.1, 0.8, 0.1]]):
            with self.subTest(scores=scores):
                self.assertEqual(classify_scores(scores, NAMES), (None, 0.0))
        self.assertEqual(classify_scores([0.9, 0.1], ['Corn', 'Tomato']),
                         (None, 0.0))

    def test_logits_are_normalized_without_overflow(self):
        label, confidence = classify_scores([1000, 1010, 1001], NAMES)
        self.assertEqual(label, 'Not_A_Crop')
        self.assertGreater(confidence, 99)

    def test_disease_threshold_does_not_weaken_non_crop_rejection(self):
        self.assertIsNone(classify_scores([0.85, 0.10, 0.05], NAMES,
                          disease_confidence_threshold=0.95)[0])
        self.assertEqual(classify_scores([0.10, 0.85, 0.05], NAMES,
                         disease_confidence_threshold=0.95)[0], 'Not_A_Crop')

    def test_grouped_leaf_evidence_requires_new_model_and_valid_scores(self):
        self.assertTrue(has_leaf_evidence([0.45, 0.005, 0.25, 0.295], NAMES + ['Unsupported_Leaf']))
        self.assertFalse(has_leaf_evidence([0.25] * 4, NAMES + ['Unsupported_Leaf']))
        self.assertFalse(has_leaf_evidence([0.45, 0.005, 0.545], NAMES))
        self.assertFalse(has_leaf_evidence([np.nan] * 4, NAMES + ['Unsupported_Leaf']))


class ProcessingTests(unittest.TestCase):
    """Exercise the actual request processing function without loading TensorFlow.

    Model outputs are controlled: these are pipeline regression tests, not an
    accuracy evaluation of the bundled neural network.
    """

    def setUp(self):
        tree = ast.parse(Path('app.py').read_text(encoding='utf-8'))
        function = next(node for node in tree.body
                        if isinstance(node, ast.FunctionDef) and node.name == 'processing')
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.model = Mock()
        self.treatment = Mock(return_value='Treatment')

        def load_img(path, target_size):
            with Image.open(path) as source:
                return source.convert('RGB').resize(target_size[::-1])

        self.namespace = dict(
            MODEL=self.model, MODEL_LOCK=nullcontext(), MODEL_IMG_SIZE=(256, 256),
            LEAF_LOCATOR=None, LEAF_LOCATOR_THRESHOLD=.9, Image=Image,
            CLASS_NAMES=NAMES, CONFIDENCE_THRESHOLD=0.7,
            DISEASE_CONFIDENCE_THRESHOLD=0.7,
            PREDICTION_MARGIN_THRESHOLD=0.2, NON_CROP_CLASS_NAME='Not_A_Crop',
            INVALID_IMAGE_LABEL='image not supported', UNCERTAIN_IMAGE_LABEL='Uncertain image',
            np=np, classify_scores=classify_scores, has_leaf_evidence=has_leaf_evidence,
            treatment_for=self.treatment,
            upload_path=lambda name: Path(self.folder.name) / name,
            tf=SimpleNamespace(keras=SimpleNamespace(preprocessing=SimpleNamespace(
                image=SimpleNamespace(load_img=load_img, img_to_array=np.asarray)))),
        )
        exec(compile(ast.Module(body=[function], type_ignores=[]), 'app.py', 'exec'),
             self.namespace)

    def predict_image(self, color, scores):
        Image.new('RGB', (64, 64), color).save(Path(self.folder.name) / 'sample.png')
        self.model.predict.return_value = np.array([scores])
        return self.namespace['processing']('sample.png')

    def test_brown_image_reaches_model_instead_of_color_rejection(self):
        result = self.predict_image((110, 65, 35), [0.04, 0.03, 0.93])
        self.assertEqual(result[0], 'Tomato___disease')
        self.model.predict.assert_called_once()
        batch = self.model.predict.call_args.args[0]
        self.assertEqual(batch.shape, (1, 256, 256, 3))
        self.assertAlmostEqual(float(batch[0, 0, 0, 0]), 110 / 255, places=6)

    def test_green_image_cannot_override_non_crop_output(self):
        result = self.predict_image((35, 180, 55), [0.12, 0.85, 0.03])
        self.assertEqual(result[0], 'image not supported')
        self.treatment.assert_not_called()

    def test_ambiguous_non_crop_output_is_uncertain(self):
        result = self.predict_image((110, 65, 35), [0.48, 0.49, 0.03])
        self.assertEqual(result[0], 'Uncertain image')
        self.treatment.assert_not_called()

    def test_corrupt_image_does_not_reach_model(self):
        (Path(self.folder.name) / 'broken.jpg').write_text('not an image')
        result = self.namespace['processing']('broken.jpg')
        self.assertTrue(result[0].startswith('Could not process image:'))
        self.model.predict.assert_not_called()

    def test_unsupported_leaf_is_not_reported_as_non_crop_or_given_treatment(self):
        self.namespace['CLASS_NAMES'] = NAMES + ['Unsupported_Leaf']
        result = self.predict_image((110, 65, 35), [0.02, 0.01, 0.02, 0.95])
        self.assertEqual(result[0], 'Uncertain image')
        self.assertIn('A leaf was recognized', result[2])
        self.treatment.assert_not_called()

    def test_recognized_leaf_with_uncertain_condition_has_no_treatment(self):
        self.namespace['CLASS_NAMES'] = NAMES + ['Unsupported_Leaf']
        self.namespace['DISEASE_CONFIDENCE_THRESHOLD'] = 0.95
        result = self.predict_image((110, 65, 35), [0.73, 0.005, 0.20, 0.065])
        self.assertEqual(result[0], 'Uncertain image')
        self.assertIn('A leaf was recognized', result[2])
        self.treatment.assert_not_called()

    def test_failed_leaf_localization_does_not_issue_disease_or_treatment(self):
        self.namespace.update(LEAF_LOCATOR=Mock(), locate_leaf=Mock(return_value=None))
        result = self.predict_image((35, 180, 55), [0.01, 0.01, .98])
        self.assertEqual(result[0], 'Uncertain image')
        self.assertIn('could not be located', result[2])
        self.model.predict.assert_not_called()
        self.treatment.assert_not_called()

    def test_locator_crop_is_the_actual_classifier_input(self):
        from leaf_locator import crop_leaf
        self.namespace.update(LEAF_LOCATOR=Mock(), locate_leaf=Mock(return_value=[.25, .25, .75, .75]), crop_leaf=crop_leaf)
        image = Image.new('RGB', (100, 100), (255, 0, 0))
        image.paste((0, 255, 0), (20, 20, 80, 80))
        image.save(Path(self.folder.name) / 'sample.png')
        self.model.predict.return_value = np.array([[.01, .01, .98]])
        result = self.namespace['processing']('sample.png')
        self.assertEqual(result[0], 'Tomato___disease')
        pixels = self.model.predict.call_args.args[0]
        self.assertTrue(np.all(pixels[0, :, :, 0] == 0))
        self.assertTrue(np.all(pixels[0, :, :, 1] == 1))


if __name__ == '__main__':
    unittest.main()
