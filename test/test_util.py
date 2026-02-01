import src.util as util
import unittest


class TestUtil(unittest.TestCase):
    def test_mean(self):
        self.assertEqual(util.mean([0]), 0)
        self.assertEqual(util.mean([1, 2, 3, 4, 5]), 3)
        self.assertEqual(util.mean([-2, 2, 4, 8, 20]), 6.4)
        with self.assertRaises(ZeroDivisionError):
            util.mean([])

    def test_median(self):
        self.assertEqual(util.median([1, 2, 3, 4, 5]), 3)
        self.assertEqual(util.median([1, 2]), 1.5)
        self.assertEqual(util.median([0]), 0)
        with self.assertRaises(ValueError):
            util.median([])

    def test_mode(self):
        self.assertEqual(util.mode([1, 2, 3, 4, 5, 5]), 5)
        self.assertEqual(util.mode([1, 1]), 1)
        self.assertEqual(util.mode([99]), 99)
        self.assertEqual(util.mode([1, 1, 2, 2, 3, 3]), 1)
        self.assertEqual(util.mode([3, 3, 2, 2, 1, 1]), 1)
        with self.assertRaises(ValueError):
            util.mode([])

    def test_mode_v2(self):
        self.assertEqual(util.mode_v2([1, 2, 3, 4, 5, 5]), 5)
        self.assertEqual(util.mode_v2([1, 1]), 1)
        self.assertEqual(util.mode_v2([99]), 99)
        self.assertEqual(util.mode_v2([1, 1, 2, 2, 3, 3]), [1, 2, 3])
        self.assertEqual(util.mode_v2([3, 3, 2, 2, 1, 1]), [3, 2, 1])
        with self.assertRaises(ValueError):
            util.mode_v2([])

    def test_variance(self):
        self.assertEqual(util.variance([1, 1, 1], 1), 0)
        self.assertAlmostEqual(util.variance([2, 8, 10, 16, 32],
                                           util.mean([2, 8, 10, 16, 32])), 105, 0)
        self.assertAlmostEqual(util.variance([2, 8, 10, 16, 32],
                                           util.mean([2, 8, 10, 16, 32]), grouping_type='population'), 105, 0)
        with self.assertRaises(ZeroDivisionError):
            util.variance([], 1)
        self.assertEqual(util.variance([1, 5], 3, grouping_type='population'), 4)
        self.assertEqual(util.variance([1, 5], 3, grouping_type='sample'), 8)

    def test_covariance(self):
        l1 = [1, 2, 3]
        l2 = [2, 4, 6]
        self.assertAlmostEqual(util.covariance(l1, util.mean(l1), l2, util.mean(l2)), 1.333, 3)
        self.assertEqual(util.covariance(l1, util.mean(l1), l2, util.mean(l2), grouping_type='sample'), 2)
        with self.assertRaises(ZeroDivisionError):
            util.covariance([], 0, [], 0)
        with self.assertRaises(AssertionError):
            util.covariance([1, 2], 1.5, [2], 2)
        self.assertAlmostEqual(util.covariance([10, 50, 90, 150],
                                             util.mean([10, 50, 90, 150]),
                                             [26, 72, 94, 68],
                                             util.mean([26, 72, 94, 68]), grouping_type='sample'), 1006.667, 3)
        self.assertEqual(util.covariance([10, 50, 90, 150],
                                       util.mean([10, 50, 90, 150]),
                                       [26, 72, 94, 68],
                                       util.mean([26, 72, 94, 68]), grouping_type='population'), 755)
