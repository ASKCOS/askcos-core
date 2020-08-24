import gzip
import json
import os
from collections import defaultdict

import rdkit.Chem as Chem
import pandas as pd
from pymongo import MongoClient, errors

import askcos.global_config as gc
from askcos.utilities.io.logger import MyLogger

pricer_loc = 'pricer'


class Pricer:
    """
    The Pricer class is used to look up the ppg of chemicals if they
    are buyable.
    """

    def __init__(self, use_db=True, BUYABLES_DB=None):

        self.BUYABLES_DB = BUYABLES_DB
        self.use_db = use_db
        self.prices = None

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
        self.prices.to_json(file_path, orient='records', compression='gzip')

    def load_from_file(self, file_name):
        """
        Load buyables information from local file
        """
        if os.path.isfile(file_name):
            self.prices = pd.read_json(
                file_name,
                orient='records',
                dtype={'smiles': 'object', 'source': 'object', 'ppg': 'float'},
                compression='gzip',
            )
            MyLogger.print_and_log('Loaded prices from flat file', pricer_loc)
        else:
            MyLogger.print_and_log('Buyables file does not exist: {}'.format(file_name), pricer_loc)

    def lookup_smiles(self, smiles, source=None, alreadyCanonical=False, isomericSmiles=True):
        """
        Looks up a price by SMILES. Canonicalize smiles string unless 
        the user specifies that the smiles string is definitely already 
        canonical. If the DB connection does not exist, look up from 
        prices dictionary attribute, otherwise lookup from DB.
        If multiple entries exist in the DB, return the lowest price.

        Args:
            smiles (str): SMILES string to look up
            source (list or str, optional): buyables sources to consider;
                if ``None`` (default), include all sources, otherwise
                must be single source or list of sources to consider;
            alreadyCanonical (bool, optional): whether SMILES string is already
                canonical; if ``False`` (default), SMILES will be canonicalized
            isomericSmiles (bool, optional): whether to generate isomeric
                SMILES string when performing canonicalization
        """
        if not alreadyCanonical:
            mol = Chem.MolFromSmiles(smiles)
            if not mol:
                return 0.
            smiles = Chem.MolToSmiles(mol, isomericSmiles=isomericSmiles)

        if source == []:
            # If no sources are allowed, there is no need to perform lookup
            # Empty list is checked explicitly here, since None means source
            # will not be included in query, and '' is a valid source value
            return 0.0

        if self.use_db:
            query = {'smiles': smiles}

            if source is not None:
                if isinstance(source, list):
                    query['source'] = {'$in': source}
                else:
                    query['source'] = source

            cursor = self.BUYABLES_DB.find(query)
            return min([doc['ppg'] for doc in cursor], default=0.0)
        elif self.prices is not None:
            query = self.prices['smiles'] == smiles

            if source is not None:
                if isinstance(source, list):
                    query = query & (self.prices['source'].isin(source))
                else:
                    query = query & (self.prices['source'] == source)

            results = self.prices.loc[query]
            return min(results['ppg'], default=0.0)
        else:
            return 0.0


if __name__ == '__main__':
    pricer = Pricer()
    pricer.load()
    print(pricer.lookup_smiles('CCCCCO'))
    print(pricer.lookup_smiles('CCCCXCCO'))
