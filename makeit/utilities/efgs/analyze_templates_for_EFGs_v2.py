import rdkit.Chem as Chem
import rdkit.Chem.AllChem as AllChem
from pymongo import MongoClient
from bson.objectid import ObjectId
import numpy as np
from EFGs_match import *
from tqdm import tqdm 
import os
import json 

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

# TEST: load annotated library
with open(os.path.join(os.path.dirname(__file__), 'output', 'EFGs_analysis_of_products (copy).json'), 'r') as fid:
    library = json.load(fid)

# Define matching func
matching_func = precompile_matching_funcs(library)

# Get retro transforms and instance DB
db = client['reaxys']
TRANSFORM_DB = db['transforms_retro_v4']
INSTANCE_DB = db['instances']
CHEMICAL_DB = db['chemicals']
REACTION_DB = db['reactions']

def analyze_template(tform_doc):
    '''Analyze one transform document'''

    reactant_smarts = str(tform_doc['reaction_smarts'].split('>')[0])
    existence = np.array([0.0 for i in range(len(library))])
    total_count = 0.0

    done_rxids = set()

    for rxd_id in tqdm(tform_doc['references']):
        rxid = int(rxd_id.split('-')[0])
        if rxid in done_rxids:
            continue

        # Create molecule object from the product
        rx_doc = REACTION_DB.find_one({'_id': rxid}, ['RXN_SMILES'])
        product = Chem.MolFromSmiles(str(rx_doc['RXN_SMILES']).split('>')[-1])
        if not product: 
            print('Could not load product')
            continue 
        product.UpdatePropertyCache()

        # Get EFG signature
        EFG = matching_func(product)
        existence += np.array(EFG)
        total_count += 1.0
        done_rxids.add(rx_doc['_id'])

    np.seterr(divide='ignore', invalid='ignore')

    # Save to file
    with open(os.path.join(FROOT, 'EFG_template_analysis_{}.csv'.format(tform_doc['_id'])), 'w') as fid:
        fid.write('_id: {}\n'.format(tform_doc['_id']))
        fid.write('SMARTS: {}\n'.format(tform_doc['reaction_smarts']))
        fid.write('number of examples (rxd ids): {}\n'.format(len(tform_doc['references'])))
        fid.write('number of examples (rx ids): {}\n'.format(len(done_rxids)))
        fid.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(
            'EFG Index', 'Name', 'SMARTS', 'In reduced set?', 'Number of products with group', 
            'Fraction of products with group', 'Fraction of products (in general) with group',
            'Total number of products (in general) tested', 'Z-score', 'Fractional Boost'
        ))

        for i, efg in enumerate(library):

            p1 = library[i]['match_freq']
            n1 = library[i]['total_count_tested']
            p2 = existence[i] / total_count 
            n2 = total_count 
            p = (n1*p1 + n2*p2) / (n1+n2)
            z = (p1 - p2) / np.sqrt(p*(1-p) * (1/float(n1) + 1/float(n2)))

            fid.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(
                efg['_id'], efg['name'], efg['SMARTS'], efg['redux'], existence[i], 
                p2, p1, n1, z, p2 - p1,
            ))

    return existence / total_count

if __name__ == '__main__':

    FROOT = os.path.join(os.path.dirname(__file__), 'output')
    if not os.path.isdir(FROOT):
        os.mkdir(FROOT)

    # Get ONE example for now
    tform_doc = TRANSFORM_DB.find_one({'count': {'$gt': 20000, '$lt': 50000}})
    if not tform_doc: quit(1)
    print('Template SMARTS: {}'.format(str(tform_doc['reaction_smarts'])))
    print('id: {}'.format(tform_doc['_id']))
    coexistence = analyze_template(tform_doc)
