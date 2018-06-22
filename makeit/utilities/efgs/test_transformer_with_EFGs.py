import rdkit.Chem as Chem
import rdkit.Chem.AllChem as AllChem
from pymongo import MongoClient
from bson.objectid import ObjectId
import numpy as np
from EFGs_match import *
from tqdm import tqdm 
import os
import json 
import makeit.utilities.io.pickle as pickle 

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

# Get retro transforms and instance DB
db = client['reaxys']
TRANSFORM_DB = db['transforms_retro_v4']
INSTANCE_DB = db['instances']
CHEMICAL_DB = db['chemicals']
REACTION_DB = db['reactions']


if __name__ == '__main__':

    FROOT = os.path.join(os.path.dirname(__file__), 'output')
    if not os.path.isdir(FROOT):
        os.mkdir(FROOT)

    import makeit.retro.transformer as transformer 
    RetroTransformer = transformer.Transformer()
    mincount_retro = 50
    print('Loading retro templates...')
    RetroTransformer.load(TRANSFORM_DB, mincount=mincount_retro, get_retro=True, 
        get_synth=False, efgs=True)
    print('Loaded {} retro templates'.format(RetroTransformer.num_templates))

    # Calculate global occurrences of EFG groups
    total_existence = np.array([0.0 for i in range(len(library))])
    total_total_count = 0.0
    for i, template in enumerate(RetroTransformer.templates):
        if template['efgs'] is None:
            template_doc = TRANSFORM_DB.find_one({'_id': template['_id']})

            existence = np.array([0.0 for j in range(len(library))])
            total_count = 0.0
            done_rxids = set()
            for rxd_id in tqdm(template_doc['references']):
                rxid = int(rxd_id.split('-')[0])
                if rxid in done_rxids:
                    continue
                # Create molecule object from the product
                rx_doc = REACTION_DB.find_one({'_id': rxid}, ['RXN_SMILES', 'RX_NVAR'])
                nvar = rx_doc['RX_NVAR']
                product = Chem.MolFromSmiles(str(rx_doc['RXN_SMILES']).split('>')[-1])
                if not product: 
                    print('Could not load product')
                    continue 
                product.UpdatePropertyCache()

                # Get EFG signature
                EFG = matching_func(product)
                existence += np.array(EFG) * nvar
                total_count += 1.0 * nvar
                done_rxids.add(rx_doc['_id'])

            np.seterr(divide='ignore', invalid='ignore')
            existence = existence / total_count

            RetroTransformer.templates[i]['efgs'] = existence

            # Save in DB
            TRANSFORM_DB.update_one({'_id': template['_id']},
                {'$set':
                    {'efgs': pickle.dumps(existence),}
                }
            )

        else:
            existence = pickle.loads(str(template['efgs']))
            RetroTransformer.templates[i]['efgs'] = existence
            total_count = template['count']

        # Add to totals
        total_existence += existence * total_count
        total_total_count += total_count

    print('Went through all templates in transformer and characterized EFGs')

    # Now look for applicability?
    smiles = 'Cc1ccnc2c1NC(=O)c1cccnc1N2C1CC1'
    label = 'nevirapine'
    mol = Chem.MolFromSmiles(smiles)
    EFG = np.array(matching_func(mol), dtype=np.float32)

    applicability = [0.0 for i in range(len(RetroTransformer.templates))]
    for i, template in enumerate(RetroTransformer.templates):
        applicability[i] = np.dot(EFG, template['efgs'] - total_existence/total_total_count)

    with open(os.path.join(os.path.dirname(__file__), 'output', 
        'test_applicability_{}.csv'.format(label)), 'w') as fid:
        fid.write('{}\n'.format(smiles))
        fid.write('{}\t{}\t{}\t{}\t{}\t{}\n'.format('count', 'template', 
            'applicability score', 'num outcomes', 'example', 'example_csf'))
        for i, template in enumerate(RetroTransformer.templates):
            outcomes = template['rxn'].RunReactants([mol])
            outcome = ''
            outcome_csf = 0.
            if len(outcomes) > 0:
                outcome = Chem.MolToSmiles(outcomes[0][0])
                from makeit.retro.transformer import RetroPrecursor
                precursor = RetroPrecursor(smiles_list = [outcome])
                precursor.score()
                outcome_csf = precursor.retroscore
            fid.write('{}\t{}\t{}\t{}\t{}\t{}\n'.format(template['count'],
                template['reaction_smarts'], applicability[i], len(outcomes), 
                outcome, outcome_csf))