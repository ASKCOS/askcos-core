import unittest

from . import preprocess_reagent_group


class Test(unittest.TestCase):
    def test_01(self):
        r = ['CO.[Na+].[OH-]','CO.O']
        result = preprocess_reagent_group.preprocess_reagents(r)
        result.sort()
        expected = ['CO', 'O', 'O[Na]']
        self.assertEqual(result, expected)

    def test_02(self):
        r = ['[Li+].[AlH4-]']
        result = preprocess_reagent_group.preprocess_reagents(r)
        result.sort()
        expected = ['[Li][AlH4]']
        self.assertEqual(result, expected)

    def test_03(self):
        r = ["[Cl-].[Cl-].[Ce+2]"]
        result = preprocess_reagent_group.preprocess_reagents(r)
        result.sort()
        expected = ["Cl[Ce]Cl"]
        self.assertEqual(result, expected)

    def test_04(self):
        r = ["[OH-].[NH4+]"]
        result = preprocess_reagent_group.preprocess_reagents(r)
        result.sort()
        expected = ["N", "O"]
        self.assertEqual(result, expected)


if __name__ == '__main__':
    res = unittest.main(verbosity=3, exit=False)
