#!/usr/bin/env python

import sys
import json
import copy
import math
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger

from . import db
from . import smiles_util

# These rules convert ions into their connected neutral molecular form.
# This is neccessary for spiltting reagents.
reagent_conv_rules = None
if reagent_conv_rules is None:
    with open(db.config_reaction_condition['reagent_conv_rules'], 'r') as f:
        _reagent_conv_rules = json.load(f)
    reagent_conv_rules = {}
    for k, v in _reagent_conv_rules.items():
        s = smiles_util.canonicalize_smiles(k)
        if k is not None:
            reagent_conv_rules[s] = v

def canonicalize_smiles_reagent_conv_rules(s):
    s_can = smiles_util.canonicalize_smiles(s)
    r = reagent_conv_rules.get(s_can, None)
    if r is not None:
        return r
    else:
        return s

# SMILES cannot be parsed by RDKit. If RDKit can parse them correctly, they can be removed.
SMILES2CHARGE = {
    'Br[Br-]Br':-1,
    '[BH]1234[H][BH]561[BH]173[BH]34([H]2)[BH]247[BH]761[BH]165[H][BH]586[BH]32([BH]4715)[H]8':0,
    '[BH3][S](C)C':0,
    '[BH3-][S+](C)C':0,
    '[BH3][O]1CCCC1':0,
    'CC1CCCCN1[BH3]':0,
}

def get_smiles_charge(s):
    q = 0
    for i in s.split('.'):
        q += get_smiles_charge_single_mol(i)
    return q

def get_smiles_charge_single_mol(s):
    q = SMILES2CHARGE.get(s, None)
    if q is None:
        try:
            mol = Chem.MolFromSmiles(canonicalize_smiles_reagent_conv_rules(s), sanitize=False)
        except:
            raise ValueError('get_smiles_charge(): MolFromSmiles() fails, s='+s)
        if mol is None:
            raise ValueError('get_smiles_charge(): MolFromSmiles() fails, s='+s)
        q = Chem.rdmolops.GetFormalCharge(mol)
    return q

def split_neutral_fragment(reagents):
    reagents_neutral = set()
    reagents_charged = []
    for r in reagents:
        r_split = r.split('.')
        r_remaining = []
        for s in r_split:
            q = get_smiles_charge(s)
            if int(q) == 0:
                reagents_neutral.add(s)
            else:
                r_remaining.append(s)
        if len(r_remaining) > 0:
            r_remaining = '.'.join(r_remaining)
            q = get_smiles_charge(r_remaining)
            if int(q) == 0:
                reagents_neutral.add(r_remaining)
            else:
                reagents_charged.append(r_remaining)
    return reagents_neutral, reagents_charged

def preprocess_reagents(reagents):
    '''
        inputs: list of str, smiles
        outputs: list of str, smiles
        Rules:
        1. Neutral molecules are splitted from compounds
        2. Try to combine separated charged species
        3. Canonicalization using hardcoded rules
    '''
    assert isinstance(reagents, list)
    for i in range(len(reagents)):
        reagents[i] = canonicalize_smiles_reagent_conv_rules(reagents[i])
    
    # Rule 1, split neutral
    reagents_neutral, reagents_charged = split_neutral_fragment(reagents)
    
    # Rule 2, combine charged, reagents_charged --> reagents_neutral
    # q for smiles in reagents_charged
    charges = [get_smiles_charge(s) for s in reagents_charged]
    # sanity check
    # check 1, total charge 0
    total_charge = sum(charges)
    if total_charge != 0:
        print('reagents: ', reagents)
        print('reagents_neutral: ', reagents_neutral)
        print('reagents_charged: ', reagents_charged)
        raise ValueError('preprocess_reagents(): total charge is not zero, q='+str(total_charge))
    if len(reagents_charged) > 0:
        reagents_neutral.add(smiles_util.canonicalize_smiles('.'.join(reagents_charged)))
    reagents_neutral = list(reagents_neutral)
    
    # Rule 3, Canonicalization, replace using reagent_conv_rules.json
    res = set()
    for i in reagents_neutral:
        tmp = canonicalize_smiles_reagent_conv_rules(i)
        tmp1, tmp2 = split_neutral_fragment([tmp])
        if len(tmp2) != 0:
            sys.stderr.write('preprocess_reagents(): error: charged fragment, s='+str(reagents)+'\n')
        for s in tmp1:
            res.add(s)
    
    return list(res)
