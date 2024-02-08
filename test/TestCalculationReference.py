import src.CalculationReference as Cr
import unittest


class TestCalculationReference(unittest.TestCase):
    def test_mean(self):
        self.assertEqual(Cr.mean([0]), 0)
        self.assertEqual(Cr.mean([1, 2, 3, 4, 5]), 3)
        self.assertEqual(Cr.mean([-2, 2, 4, 8, 20]), 6.4)
        with self.assertRaises(ZeroDivisionError):
            Cr.mean([])
