import unittest
import numpy as np
from PIL import Image
from leaf_locator import box_iou, crop_leaf, locate_leaf, valid_box


class LocatorTests(unittest.TestCase):
    def test_crop_keeps_leaf_with_padding_and_clamps_edges(self):
        image = Image.new('RGB', (100, 200))
        self.assertEqual(crop_leaf(image, [0, 0, .5, .5], padding=1).size, (100, 200))
        self.assertEqual(crop_leaf(image, [.25, .25, .75, .75], padding=0).size, (50, 100))

    def test_invalid_boxes_cannot_crop_images(self):
        for box in ([.8, .1, .2, .8], [0, 0, 0, 1], [float('nan'), 0, 1, 1], [-.1, 0, 1, 1]):
            self.assertFalse(valid_box(box))
            with self.assertRaises(ValueError):
                crop_leaf(Image.new('RGB', (20, 20)), box)

    def test_low_confidence_and_invalid_predictions_abstain(self):
        image = Image.new('RGB', (100, 100))
        def fake(score, box):
            return lambda pixels, training: {'leaf': np.array([[score]]), 'box': np.array([box])}
        self.assertIsNone(locate_leaf(fake(.5, [0, 0, 1, 1]), image))
        self.assertIsNone(locate_leaf(fake(.99, [.9, 0, .1, 1]), image))
        self.assertEqual(locate_leaf(fake(.99, [.1, .1, .9, .9]), image), [.1, .1, .9, .9])

    def test_iou_handles_overlap_and_disjoint_boxes(self):
        self.assertEqual(box_iou([0, 0, 1, 1], [0, 0, 1, 1]), 1)
        self.assertEqual(box_iou([0, 0, .2, .2], [.8, .8, 1, 1]), 0)
        self.assertAlmostEqual(box_iou([0, 0, 1, 1], [0, 0, .5, .5]), .25)
