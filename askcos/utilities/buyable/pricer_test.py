import gzip
import os
import unittest

import pandas as pd
from pymongo import MongoClient, collection, errors

import askcos.global_config as gc
from askcos.utilities.buyable.pricer import Pricer


def db_available():
    """Check if a mongo db instance is available."""
    db_client = MongoClient(serverSelectionTimeoutMS=1000, **gc.MONGO)

    try:
        db_client.server_info()
    except errors.ServerSelectionTimeoutError:
        return False
    else:
        return True


class TestPricer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """This method is run once before every test in this module."""
        cls.pricer = Pricer(use_db=False)
        # Set some arbitrary prices
        data = [
            {'smiles': 'CCCCCO', 'source': 'test1', 'ppg': 1.0},
            {'smiles': 'c1ccccc1', 'source': 'test1', 'ppg': 2.0},
            {'smiles': 'c1ccccc1', 'source': 'test2', 'ppg': 3.0},
        ]
        cls.pricer.prices = pd.DataFrame(data)

    def test_lookup_existing_smiles(self):
        """Test that we can lookup a SMILES that exists."""
        result = self.pricer.lookup_smiles('CCCCCO')
        expected = 1.0
        self.assertAlmostEqual(expected, result)

    def test_lookup_existing_smiles_source(self):
        """Test that we can lookup a SMILES that exists."""
        result = self.pricer.lookup_smiles('c1ccccc1', source='test2')
        expected = 3.0
        self.assertAlmostEqual(expected, result)

    def test_lookup_existing_smiles_source_2(self):
        """Test that we can lookup a SMILES that exists."""
        result = self.pricer.lookup_smiles('c1ccccc1', source=['test1', 'test2'])
        expected = 2.0
        self.assertAlmostEqual(expected, result)

    def test_lookup_nonexisting_smiles(self):
        """Test that we can lookup a SMILES that does not exist."""
        result = self.pricer.lookup_smiles('CCCCCN')
        expected = 0.0
        self.assertAlmostEqual(expected, result)

    def test_lookup_nonexisting_smiles_source(self):
        """Test that we can lookup a SMILES that does not exist."""
        result = self.pricer.lookup_smiles('CCCCCO', source='test2')
        expected = 0.0
        self.assertAlmostEqual(expected, result)

    def test_lookup_invalid_smiles(self):
        """Test that we can lookup an invalid SMILES."""
        result = self.pricer.lookup_smiles('CCCCCX')
        expected = 0.0
        self.assertAlmostEqual(expected, result)

    def test_dump_and_load(self):
        """Test that we can dump the prices to a file and load again."""
        filename = os.path.join(os.path.dirname(__file__), 'temp.json.gz')
        self.pricer.dump_to_file(filename)
        self.assertTrue(os.path.isfile(filename))

        with gzip.open(filename, 'rt', encoding='utf-8') as f:
            # Just check the length of the string, since dict order is non-deterministic
            self.assertEqual(len(f.read()), 146)

        new_pricer = Pricer()
        new_pricer.load_from_file(filename)

        self.assertTrue(self.pricer.prices.equals(new_pricer.prices))

        os.remove(filename)

    @unittest.skipIf(db_available(), 'Skipping because mongo db is available.')
    def test_db_fallback(self):
        """Test that we load from file if db is not available."""
        filename = os.path.join(os.path.dirname(__file__), 'temp.json.gz')
        self.pricer.dump_to_file(filename)
        self.assertTrue(os.path.isfile(filename))

        new_pricer = Pricer(use_db=True)
        new_pricer.load(filename)

        self.assertFalse(new_pricer.use_db)
        self.assertIsNone(new_pricer.BUYABLES_DB)
        self.assertTrue(self.pricer.prices.equals(new_pricer.prices))

        os.remove(filename)

    @unittest.skipIf(not db_available(), 'Skipping because mongo db is not available.')
    def test_db_lookup(self):
        """Test that we can load pricing data from mongo db."""
        new_pricer = Pricer(use_db=True)
        new_pricer.load()

        self.assertFalse(new_pricer.prices)  # prices should be empty
        self.assertTrue(new_pricer.BUYABLES_DB)
        self.assertIsInstance(new_pricer.BUYABLES_DB, collection.Collection)

        self.assertEqual(new_pricer.lookup_smiles('CCCCCO'), 1.0)

    @unittest.skipIf(not db_available(), 'Skipping because mongo db is not available.')
    def test_db_lookup_nonexisting(self):
        """Test that we can return price for chemical not in mongo db."""
        new_pricer = Pricer(use_db=True)
        new_pricer.load()

        self.assertFalse(new_pricer.prices)  # prices should be empty
        self.assertTrue(new_pricer.BUYABLES_DB)
        self.assertIsInstance(new_pricer.BUYABLES_DB, collection.Collection)

        self.assertEqual(new_pricer.lookup_smiles('C#C'), 0.0)


if __name__ == '__main__':
    res = unittest.main(verbosity=3, exit=False)
