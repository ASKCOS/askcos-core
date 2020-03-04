import gzip
import os
import unittest

from makeit.utilities.buyable.pricer import Pricer


class TestPricer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """This method is run once before every test in this module."""
        cls.pricer = Pricer()
        # Set some arbitrary prices
        cls.pricer.prices['CCCCCO'] = 1.0
        cls.pricer.prices['c1ccccc1'] = 2.0

    def test_lookup_existing_smiles(self):
        """Test that we can lookup a SMILES that exists."""
        result = self.pricer.lookup_smiles('CCCCCO')
        expected = 1.0
        self.assertAlmostEqual(expected, result)

    def test_lookup_nonexisting_smiles(self):
        """Test that we can lookup a SMILES that does not exist."""
        result = self.pricer.lookup_smiles('CCCCXCCO')
        expected = 0.0
        self.assertAlmostEqual(expected, result)

    def test_dump_and_load(self):
        """Test that we can dump the prices to a file and load again."""
        filename = 'temp.json.gz'
        self.pricer.dump_to_file(filename)
        self.assertTrue(os.path.isfile(filename))

        with gzip.open(filename, 'rt', encoding='utf-8') as f:
            # Just check the length of the string, since dict order is non-deterministic
            self.assertEqual(len(f.read()), 70)

        new_pricer = Pricer()
        new_pricer.load_from_file(filename)

        self.assertEqual(self.pricer.prices, new_pricer.prices)

        os.remove(filename)


if __name__ == '__main__':
    res = unittest.main(verbosity=3, exit=False)
