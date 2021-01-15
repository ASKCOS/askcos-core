#!/usr/bin/env python

import sys
import numpy as np

from . import preprocess_reagent_group
from . import smiles_util

def generate_reagents_encoder2(all_reagents):
    enc = {None:0}
    cnt = 0
    for r in all_reagents:
        s = smiles_util.canonicalize_smiles(r)
        if s is None:
            raise RuntimeError("generate_reagents_encoder2: cannot load reagent list.")
        if enc.get(s, None) is None:
            cnt += 1
            enc[s] = cnt
    return enc

def prepare_reagents2(encoder, reagents):
    '''
        encoder: {None:0, 'smiles1':1}
        reagents: [{'smiles':'xx', 'conc':1}, None, None]
    '''
    if not isinstance(reagents, list):
        reagents = [reagents]
    
    valid_reagents = []
    for r in reagents:
        if r['smiles'] is None or r['smiles'] is 'missing':
            continue
        valid_reagents.append(r['smiles'])
    
    valid_reagents = preprocess_reagent_group.preprocess_reagents(valid_reagents)
    res = np.zeros((len(valid_reagents), len(encoder)), dtype=np.float32)
    for i in range(len(valid_reagents)):
        idx = encoder.get(smiles_util.canonicalize_smiles(valid_reagents[i]), None)
        if idx is None:
            sys.stderr.write('prepare_reagents2(): encoder missing smiles='+valid_reagents[i]+'\n')
        else:
            res[i, idx] = 1
    
    return res

def convert_onehots_to_multiclass(onehots):
    '''
        onehots: (nclass, len_onehots)
        return: (len_onehots)
    '''
    res = np.sum(onehots, axis=0, keepdims=False, dtype=np.float32)
    res = np.where(res > 0, np.float32(1.0), np.float32(0.0))
    return res
