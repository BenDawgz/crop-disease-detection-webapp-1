import unittest

from training.prepare_public_data import leaf_identifier, split_for


class LeafGroupingTests(unittest.TestCase):
    def test_copy_and_original_share_the_full_leaf_identifier(self):
        original = leaf_identifier('uuid___R.S_HL 0606.JPG')
        copied = leaf_identifier('other_uuid___R.S_HL 0606 copy.JPG')
        self.assertEqual(original, 'r.s_hl 0606')
        self.assertEqual(original, copied)
        self.assertEqual(split_for('pv:' + original), split_for('pv:' + copied))

    def test_distinct_maize_leaves_do_not_collapse_to_r(self):
        self.assertNotEqual(leaf_identifier('uuid___R.S_HL 0606 copy.JPG'),
                            leaf_identifier('uuid___R.S_HL 0607 copy.JPG'))


if __name__ == '__main__':
    unittest.main()
