#!/usr/bin/env python

import json
import os
import sys

import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem

import numpy as np

from . import db

def get_morgan_fp(s, fp_radius, fp_length):
    return np.array(AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s, sanitize=True), fp_radius, nBits=fp_length), dtype='float32')

def canonicalize_smiles_rdkit(s):
    try:
        s_can = Chem.MolToSmiles(Chem.MolFromSmiles(s, sanitize=False)) # avoid 'Br[Br-]Br' problem
    except:
        sys.stderr.write('canonicalize_smiles_rdkit(): fail s='+s+'\n')
        s_can = None
    return s_can

def canonicalize_smiles(s):
    s = canonicalize_smiles_rdkit(s)
    return s
