import os
import unittest
from pymongo import MongoClient, errors

import askcos.global_config as gc
from askcos.utilities.historian.chemicals import ChemHistorian

def db_available():
    """Check if a mongo db instance is available."""
    db_client = MongoClient(
        gc.MONGO['path'],
        gc.MONGO['id'],
        connect=gc.MONGO['connect'],
        serverSelectionTimeoutMS=1000,
    )

    try:
        db_client.server_info()
    except errors.ServerSelectionTimeoutError:
        return False
    else:
        return True

class TestChemHistorian(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """This method is run once before each test in this class."""
        cls.chemhistorian = ChemHistorian()
        cls.chemhistorian.load_from_file(os.path.join(os.path.dirname(__file__), 'test_data', 'chemicals.json.gz'))

    def test_01_lookup_smiles(self):
        """Test that we can look up a SMILES string in chemhistorian."""
        result = self.chemhistorian.lookup_smiles('CCCCO')
        expected = {'as_product': 2726, 'as_reactant': 17450, 'template_set': 'reaxys'}
        self.assertEqual(expected, result)


@unittest.skipIf(not db_available(), 'Skipping because mongo db is not available.')
class TestDBChemHistorian(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """This method is run once before each test in this class."""
        cls.chemhistorian = ChemHistorian(use_db=True)
        cls.chemhistorian.load()
        cls.new_doc = {'_id': 'test_id','smiles': 'CCCCO', 'as_product': 1, 'as_reactant': 1, 'template_set': 'test_template_set'}

    @classmethod
    def tearDownClass(cls):
        """This is run after each test in this class."""
        cls.chemhistorian.CHEMICALS_DB.delete_one(cls.new_doc)

    def test_01_lookup_smiles(self):
        """Test that we can look up a SMILES string in chemhistorian."""
        result = self.chemhistorian.lookup_smiles('CCCCO')
        result.pop('_id')
        expected = {'as_product': 2726, 'as_reactant': 17450}
        self.assertEqual(expected, result)

    def test_02_use_db_template_set(self):
        """Test that when using the mongoDB, we can lookup in different template sets"""
        self.chemhistorian.CHEMICALS_DB.insert_one(self.new_doc)
        reaxys_result = self.chemhistorian.lookup_smiles('CCCCO', template_set='reaxys')
        reaxys_result.pop('_id')
        expected_reaxys = {'as_product': 2726, 'as_reactant': 17450}
        self.assertEqual(expected_reaxys, reaxys_result)
        new_result = self.chemhistorian.lookup_smiles('CCCCO', template_set='test_template_set')
        expected_new = {'as_product': self.new_doc['as_product'], 'as_reactant': self.new_doc['as_reactant'], '_id': self.new_doc['_id']}
        self.assertEqual(expected_new, new_result)

if __name__ == '__main__':
    res = unittest.main(verbosity=3, exit=False)
