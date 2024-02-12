import src.CalculationReference as Cr
import unittest


class TestCalculationReference(unittest.TestCase):
    def test_mean(self):
        self.assertEqual(Cr.mean([0]), 0)
        self.assertEqual(Cr.mean([1, 2, 3, 4, 5]), 3)
        self.assertEqual(Cr.mean([-2, 2, 4, 8, 20]), 6.4)
        with self.assertRaises(ZeroDivisionError):
            Cr.mean([])

    def test_median(self):
        self.assertEqual(Cr.median([1, 2, 3, 4, 5]), 3)
        self.assertEqual(Cr.median([1, 2]), 1.5)
        self.assertEqual(Cr.median([0]), 0)
        with self.assertRaises(ValueError):
            Cr.median([])

    def test_mode(self):
        self.assertEqual(Cr.mode([1, 2, 3, 4, 5, 5]), 5)
        self.assertEqual(Cr.mode([1, 1]), 1)
        self.assertEqual(Cr.mode([99]), 99)
        self.assertEqual(Cr.mode([1, 1, 2, 2, 3, 3]), 1)
        self.assertEqual(Cr.mode([3, 3, 2, 2, 1, 1]), 1)
        self.assertRaises(ValueError, lambda: Cr.mode([]))

    def test_mode_v2(self):
        self.assertEqual(Cr.mode_v2([1, 2, 3, 4, 5, 5]), 5)
        self.assertEqual(Cr.mode_v2([1, 1]), 1)
        self.assertEqual(Cr.mode_v2([99]), 99)
        self.assertEqual(Cr.mode_v2([1, 1, 2, 2, 3, 3]), [1, 2, 3])
        self.assertEqual(Cr.mode_v2([3, 3, 2, 2, 1, 1]), [3, 2, 1])
        self.assertRaises(ValueError, lambda: Cr.mode_v2([]))

    def test_variance(self):
        self.assertEqual(Cr.variance([1, 1, 1], 1), 0)
        self.assertAlmostEqual(Cr.variance([2, 8, 10, 16, 32],
                                           Cr.mean([2, 8, 10, 16, 32])), 105, 0)
        self.assertAlmostEqual(Cr.variance([2, 8, 10, 16, 32],
                                           Cr.mean([2, 8, 10, 16, 32]), grouping_type='population'), 105, 0)
        self.assertRaises(ZeroDivisionError, lambda: Cr.variance([],1))
        self.assertEqual(Cr.variance([1, 5], 3, grouping_type='population'), 4)
        self.assertEqual(Cr.variance([1, 5], 3, grouping_type='sample'), 8)
