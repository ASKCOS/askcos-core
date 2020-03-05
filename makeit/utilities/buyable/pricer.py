import gzip
import json
import os
from collections import defaultdict

import rdkit.Chem as Chem
from pymongo import MongoClient, errors

import makeit.global_config as gc
from makeit.utilities.io.logger import MyLogger

pricer_loc = 'pricer'


class Pricer:
    """
    The Pricer class is used to look up the ppg of chemicals if they
    are buyable.
    """

    def __init__(self, use_db=True, BUYABLES_DB=None):

        self.BUYABLES_DB = BUYABLES_DB
        self.use_db = use_db
        self.prices = defaultdict(float)  # default 0 ppg means not buyable

    def load(self, file_name=gc.BUYABLES['file_name']):
        """
        Load pricer information. Either create connection to MongoDB or load from local file.
        """
        if self.use_db:
            self.load_databases(file_name)
        else:
            self.load_from_file(file_name)

    def load_databases(self, file_name=gc.BUYABLES['file_name']):
        """
        Load the pricing data from the online database

        If connection to MongoDB cannot be made, fallback and try to load from local file.
        """
        db_client = MongoClient(
            gc.MONGO['path'],
            gc.MONGO['id'],
            connect=gc.MONGO['connect'],
            serverSelectionTimeoutMS=1000
        )

        try:
            db_client.server_info()
        except errors.ServerSelectionTimeoutError:
            MyLogger.print_and_log('Cannot connect to mongodb to load prices', pricer_loc)
            self.use_db = False
            self.load(file_name=file_name)
        else:
            db = db_client[gc.BUYABLES['database']]
            self.BUYABLES_DB = db[gc.BUYABLES['collection']]

    def dump_to_file(self, file_path):
        """
        Write prices to a local file
        """
        prices = [{'smiles': s, 'ppg': p} for s, p in self.prices.items()]

        with gzip.open(file_path, 'wt', encoding='utf-8') as f:
            json.dump(prices, f)

    def load_from_file(self, file_name):
        """
        Load buyables information from local file
        """
        if os.path.isfile(file_name):
            with gzip.open(file_name, 'rt', encoding='utf-8') as f:
                prices = json.load(f)

            self.prices.update({p['smiles']: p['ppg'] for p in prices if p.get('smiles')})

            MyLogger.print_and_log('Loaded prices from flat file', pricer_loc)
        else:
            MyLogger.print_and_log('Buyables file does not exist: {}'.format(file_name), pricer_loc)

    def lookup_smiles(self, smiles, alreadyCanonical=False, isomericSmiles=True):
        """
        Looks up a price by SMILES. Canonicalize smiles string unless 
        the user specifies that the smiles string is definitely already 
        canonical. If the DB connection does not exist, look up from 
        prices dictionary attribute, otherwise lookup from DB.
        If multiple entries exist in the DB, return the lowest price.
        """
        if not alreadyCanonical:
            mol = Chem.MolFromSmiles(smiles)
            if not mol:
                return 0.
            smiles = Chem.MolToSmiles(mol, isomericSmiles=isomericSmiles)

        if self.use_db:
            cursor = self.BUYABLES_DB.find({
                'smiles': smiles,
                'source': {'$ne': 'LN'}
            })
            if cursor.count():
                return min([doc['ppg'] for doc in cursor])
            else:
                return 0.
        else:
            return self.prices[smiles]


if __name__ == '__main__':
    pricer = Pricer()
    pricer.load()
    print(pricer.lookup_smiles('CCCCCO'))
    print(pricer.lookup_smiles('CCCCXCCO'))
