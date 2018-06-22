import rdkit.Chem as Chem
import rdkit.Chem.AllChem as AllChem
from pymongo import MongoClient
from bson.objectid import ObjectId
import numpy as np
from EFGs_match import *
from tqdm import tqdm 
import six; from six.moves import cPickle as pickle
import json
import os
from copy import copy

'''This script is meant to test out an EFG-based filtering system
for retrosynthesis. Instead of applying all templates that match, 
we can develop a 'conservative' system by not applying templates in
cases where we have not seen a certain functional group co-present with
the reaction core (in the product).'''

import makeit.global_config as gc
client = MongoClient(gc.MONGO['path'], gc.MONGO[ 'id'], connect=gc.MONGO['connect'])
db = client['askcos_transforms']
EFG_DB = db['EFGs']
library = [doc for doc in EFG_DB.find({'redux': True})]

# Define matching func
matching_func = precompile_matching_funcs(library)

def analyze_products(existence=None, analyzed_rxids=None):
    '''Analyze one transform document'''
    
    # Default values - have not started analysis before
    if existence is None:
        existence = np.array([0.0 for i in range(len(library))])
    if analyzed_rxids is None:
        analyzed_rxids = set()
    total_count = len(analyzed_rxids)
    counter = 0

    try:
        for rx_doc in tqdm(REACTION_DB.find({}, ['_id', 'RXN_SMILES'])):
            if 'RXN_SMILES' not in rx_doc: 
                print('Reaction document does not contain SMILES')
                continue
            if rx_doc['_id'] in analyzed_rxids: continue

            if counter % 2000 == 0:
                with open(analyzed_rxid_fpath, 'w') as fid:
                    json.dump(list(analyzed_rxids), fid, sort_keys=True)
                with open(annotated_library, 'w') as fid:
                    library_copy = copy(library)
                    for i in range(len(library_copy)):
                        library_copy[i]['match_count'] = existence[i]
                        library_copy[i]['match_freq'] = existence[i] / total_count
                        library_copy[i]['total_count_tested'] = total_count
                    json.dump(library_copy, fid, indent=4, sort_keys=True,
                        separators=(',', ': '))
                print('Saved!')

            # Create molecule object from the product
            product = Chem.MolFromSmiles(str(rx_doc['RXN_SMILES']).split('>')[-1])
            if not product: 
                print('Could not load product')
                continue 
            product.UpdatePropertyCache()
            
            # Update tracker with results of matching_func
            EFG = np.array(matching_func(product), dtype=bool)
            existence += EFG
            total_count += 1.0
            analyzed_rxids.add(rx_doc['_id'])
            counter += 1   

            # Encode bit array into uint8 using apckbits
            PROD_EFG_DB.insert_one({
                '_id': rx_doc['_id'],
                'efgs': pickle.dumps(np.packbits(EFG)),
            })

    finally:
        
        # One more log
        with open(analyzed_rxid_fpath, 'w') as fid:
            json.dump(list(analyzed_rxids), fid, sort_keys=True)
        with open(annotated_library, 'w') as fid:
            library_copy = copy(library)
            for i in range(len(library_copy)):
                library_copy[i]['match_count'] = existence[i]
                library_copy[i]['match_freq'] = existence[i] / total_count
                library_copy[i]['total_count_tested'] = total_count
            json.dump(library_copy, fid, indent=4, sort_keys=True,
                separators=(',', ': '))
        print('Saved!')

    return existence, total_count

if __name__ == '__main__':


    # Get retro transforms and instance DB
    db = client['reaxys']
    CHEMICAL_DB = db['chemicals']
    REACTION_DB = db['reactions']
    PROD_EFG_DB = db['reactions_product_efgs']

    FROOT = os.path.join(os.path.dirname(__file__), 'output')
    if not os.path.isdir(FROOT):
        os.mkdir(FROOT)

    analyzed_rxid_fpath = os.path.join(FROOT, 'EFGs_analyzed_rxids.json')
    annotated_library = os.path.join(FROOT, 'EFGs_analysis_of_products.json')

    existence = None
    analyzed_rxids = None
    if os.path.isfile(analyzed_rxid_fpath) and os.path.isfile('EFG_existence.pickle'):
        with open(analyzed_rxid_fpath, 'r') as fid:
            analyzed_rxids = set(json.load(fid))
        with open(annotated_library, 'r') as fid:
            annotated_library = json.load(fid)
        print('Reloaded {} previously read RXIDs'.format(len(analyzed_rxids)))
    existence = np.array([doc['match_count'] for doc in annotated_library])
    existence, total_count = analyze_products(existence=existence, analyzed_rxids=analyzed_rxids)
    print(existence)

    # Save to file
    with open(os.path.join(FROOT, 'EFGs_analysis_of_products.csv'), 'w') as fid:
        fid.write('{}\t{}\t{}\t{}\t{}\t{}\n'.format(
            'EFG Index', 'Name', 'SMARTS', 'In reduced set?', 'Number of products with group', 
            'Fraction of products with group'
        ))

        for i, efg in enumerate(library):
            fid.write('{}\t{}\t{}\t{}\t{}\t{}\n'.format(
                efg['_id'], efg['name'], efg['SMARTS'], efg['redux'], existence[i], existence[i] / total_count
            ))