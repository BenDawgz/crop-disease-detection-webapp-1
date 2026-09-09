import unittest
from training.prepare_local_data import original_identifier


class LocalGroupTests(unittest.TestCase):
    def test_augmented_views_keep_the_same_original_leaf(self):
        for name in ('uuid___R.S_HL 0606 copy 2_flipLR.jpg',
                     'uuid___R.S_HL 0606_270deg.JPG',
                     'uuid___R.S_HL 0606_new30degFlipLR.JPG',
                     'uuid___R.S_HL 0606.JPG'):
            self.assertEqual(original_identifier(name), 'r.s_hl 0606')

    def test_different_leaf_ids_remain_different(self):
        self.assertNotEqual(original_identifier('RS_Rust 1563_flipLR.JPG'),
                            original_identifier('RS_Rust 1564.JPG'))
