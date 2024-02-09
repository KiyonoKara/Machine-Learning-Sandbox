import src.CalculationReference as Cr
import unittest
import statistics


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
        with self.assertRaises(statistics.StatisticsError):
            Cr.median([])