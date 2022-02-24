#!/usr/bin/env python

import sys

import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger

import numpy as np

def remove_atommap(mol):
    for atom in mol.GetAtoms():
        #atom.ClearProp('molAtomMapNumber')
        atom.SetAtomMapNum(0)

def remove_redundant_mapno(smiles):
    '''
        make map no continuous starting from 1
        remove map no for non-mapped atoms
    '''
    r, rg, p = smiles.split(' ')[0].split('>')
    mol_r = Chem.MolFromSmiles(r)
    mol_p = Chem.MolFromSmiles(p)
    
    mapno2atomidx_r = {0:[]}
    mapno2atomidx_p = {0:[]}
    for atom in mol_r.GetAtoms():
        mapno = atom.GetAtomMapNum()
        idx = atom.GetIdx()
        if mapno == 0:
            mapno2atomidx_r[mapno].append(idx)
        else:
            assert mapno2atomidx_r.get(mapno, None) is None, 'duplicate mapno ({}) in reactants'.format(mapno)
            mapno2atomidx_r[mapno] = idx
    for atom in mol_p.GetAtoms():
        mapno = atom.GetAtomMapNum()
        idx = atom.GetIdx()
        if mapno == 0:
            mapno2atomidx_p[mapno].append(idx)
        else:
            assert mapno2atomidx_p.get(mapno, None) is None, 'duplicate mapno ({}) in products'.format(mapno)
            mapno2atomidx_p[mapno] = idx
    
    remove_atommap(mol_r)
    remove_atommap(mol_p)
    
    mapno_cnt = 1
    for mapno in mapno2atomidx_p:
        if mapno == 0:
            continue
        
        idx_p = mapno2atomidx_p[mapno]
        idx_r = mapno2atomidx_r.get(mapno, None)
        if idx_r is None:
            raise ValueError('no mapno ({}) in reactant'.format(mapno))
        
        mol_r.GetAtomWithIdx(idx_r).SetAtomMapNum(mapno_cnt)
        mol_p.GetAtomWithIdx(idx_p).SetAtomMapNum(mapno_cnt)
        
        mapno_cnt += 1
    
    return Chem.MolToSmiles(mol_r) + '>' + rg + '>' + Chem.MolToSmiles(mol_p)

_nbond = 10

def bond_order2int(a):
    return round(a*10)

def check_int_continuous_from_zero(a):
    b = list(sorted(a))
    for i in range(len(b)):
        if b[i] != i:
            return False
    return True

def get_atom_map_as_dict(f):
    res = {0:[]}
    for i in f['atom_features']:
        n = i['atom_map_id']
        if n == 0:
            res[0].append(i['atom_idx'])
        else:
            if res.get(n) is not None:
                raise Exception('Duplicate atom map num')
            res[n] = i['atom_idx']
    return res

def smiles2features(s):
    '''
        atom_features: {atom_idx, atom_map_id, atom_num, aromaticity, formal_charge, explicit/implicit valence, degree of connectivity, ring_size:[]}
        bond_features: {'i':atom_idx1, 'j':atom_idx2, 'order':bond_order}
        
        May throw exception
    '''
    mol = Chem.MolFromSmiles(s)
    if mol is None:
        sys.stderr.write('rdkit Chem.MolFromSmiles() fails with sanitizer, s='+s+'\n')
        raise ValueError('rdkit Chem.MolFromSmiles() fails with sanitizer')
    atom_features = []
    atom_ring_info = mol.GetRingInfo().AtomRings() # e.g. ((0, 5, 4, 3, 2, 1),)
    for i in mol.GetAtoms():
        atom_idx = i.GetIdx()
        f = {
            'atom_idx': i.GetIdx(),
            'atom_map_id': i.GetAtomMapNum(),
            'atom_num': i.GetAtomicNum(),
            'aromaticity': i.GetIsAromatic(),
            'formal_charge': i.GetFormalCharge(),
            'explicit_valence': i.GetExplicitValence(), 
            'implicit_valence': i.GetImplicitValence(),
            'degree_conn': i.GetDegree(), 
            'degree_total_conn': i.GetTotalDegree(), 
            'in_ring': i.IsInRing(),
            'ring_size': [],
            'bonds_idx': [],
            }
        if f['in_ring']:
            for j in atom_ring_info:
                if atom_idx in j:
                    f['ring_size'].append(len(j))
            f['ring_size'].sort()
        atom_features.append(f)
    
    bond_features = []
    all_bonds = mol.GetBonds()
    for i in range(len(all_bonds)):
        b = all_bonds[i]
        f = {
            'i': b.GetBeginAtomIdx(),
            'j': b.GetEndAtomIdx(),
            'order': bond_order2int(b.GetBondTypeAsDouble()),
            'is_conj': b.GetIsConjugated(),
            'in_ring': b.IsInRing(),
            }
        bond_features.append(f)
        assert atom_features[f['i']]['atom_idx'] == f['i']
        assert atom_features[f['j']]['atom_idx'] == f['j']
        atom_features[f['i']]['bonds_idx'].append(i)
        atom_features[f['j']]['bonds_idx'].append(i)
    
    return {'atom_features':atom_features, 'bond_features':bond_features}

def rxn2features(r):
    '''
        {
            reactants: {atom_features:[], bond_features:[]},
            products:  {atom_features:[], bond_features:[]}
        }
    '''
    r = remove_redundant_mapno(r)
    r_split = r.split(' ')[0].split('>')
    reactants = r_split[0]
    products = r_split[2]
    f_r = smiles2features(reactants)
    f_p = smiles2features(products)
    return {'reactants':f_r, 'products':f_p}

def swap_idxval(d):
    res = {}
    d = list(d)
    for i in range(len(d)):
        res[d[i]] = i
    res[None] = len(d)
    return res

def swap_idxval_dict(d):
    res = {}
    for k, v in d.items():
        res[v] = k
    return res

def feature_statistics_perfeature(s=None, features=None):
    '''
        First time: s = feature_statistics_perfeature()
        After: s = feature_statistics_perfeature(s, [f1, f2, f3])
        Final: s = feature_statistics_perfeature_convert(s) # cannot add features after this
    '''
    if s is None:
        s = {
            'max_atoms': 0,
            'max_bonds': 0,
            'max_ring_size': 0,
            'max_rings': 0,
            'max_bonds_per_atom': 0,
            'enumerate_list_atom': {
                'atom_num': set(),
                'formal_charge': set(),
                'explicit_valence': set(),
                'implicit_valence': set(),
                'degree_conn': set(),
                'degree_total_conn': set(),
                },
            'bond_order': set(),
            }
        return s
    if not isinstance(features, list):
        features = [features]
    for chemical in features:
        # atom mapped atoms count as 1, not 2
        try:
            map_r = get_atom_map_as_dict(chemical['reactants'])
            map_p = get_atom_map_as_dict(chemical['products'])
        except:
            continue
        l = len(map_r)-1+len(map_r[0])+len(map_p[0])
        s['max_atoms'] = max(s['max_atoms'], l)
        for k, f in chemical.items():
            # k= 'reactants', 'products'
            # s['max_atoms'] = max(s['max_atoms'], len(f['atom_features']))
            s['max_bonds'] = max(s['max_bonds'], len(f['bond_features']))
            for b in f['bond_features']:
                s['bond_order'].add(b['order'])
            for a in f['atom_features']:
                s['max_bonds_per_atom'] = max(s['max_bonds_per_atom'], len(a['bonds_idx']))
                s['max_rings'] = max(s['max_rings'], len(a['ring_size']))
                for i in a['ring_size']:
                    s['max_ring_size'] = max(i, s['max_ring_size'])
                for k in s['enumerate_list_atom']:
                    s['enumerate_list_atom'][k].add(a[k])
    return s

def feature_statistics_perfeature_convert(s):
    s['bond_order'] = swap_idxval(s['bond_order'])
    s['enumerate_list_atom']['atom_num'].add(0)
    s['enumerate_list_atom']['ring_size'] = list(range(3,11)) # 3-10
    for k in s['enumerate_list_atom']:
        s['enumerate_list_atom'][k] = swap_idxval(s['enumerate_list_atom'][k])
    return s

def feature_statistics(features):
    s = {
        'max_atoms': 0,
        'max_bonds': 0,
        'max_ring_size': 0,
        'max_rings': 0,
        'max_bonds_per_atom': 0,
        'enumerate_list_atom': {
            'atom_num': set(),
            'formal_charge': set(),
            'explicit_valence': set(),
            'implicit_valence': set(),
            'degree_conn': set(),
            'degree_total_conn': set(),
            },
        'bond_order': set(),
        }
    for chemical in features:
        # atom mapped atoms count as 1, not 2
        try:
            map_r = get_atom_map_as_dict(chemical['reactants'])
            map_p = get_atom_map_as_dict(chemical['products'])
        except:
            continue
        l = len(map_r)-1+len(map_r[0])+len(map_p[0])
        s['max_atoms'] = max(s['max_atoms'], l)
        for k, f in chemical.items():
            # k= 'reactants', 'products'
            # s['max_atoms'] = max(s['max_atoms'], len(f['atom_features']))
            s['max_bonds'] = max(s['max_bonds'], len(f['bond_features']))
            for b in f['bond_features']:
                s['bond_order'].add(b['order'])
            for a in f['atom_features']:
                s['max_bonds_per_atom'] = max(s['max_bonds_per_atom'], len(a['bonds_idx']))
                s['max_rings'] = max(s['max_rings'], len(a['ring_size']))
                for i in a['ring_size']:
                    s['max_ring_size'] = max(i, s['max_ring_size'])
                for k in s['enumerate_list_atom']:
                    s['enumerate_list_atom'][k].add(a[k])
    s['bond_order'] = swap_idxval(s['bond_order'])
    s['enumerate_list_atom']['atom_num'].add(0)
    s['enumerate_list_atom']['ring_size'] = list(range(3,11)) # 3-10
    for k in s['enumerate_list_atom']:
        s['enumerate_list_atom'][k] = swap_idxval(s['enumerate_list_atom'][k])
    return s

def encode_onehot(val, num_class):
    '''
        num_class: {0: 0, 6: 1, 7: 2, 8: 3, 53: 4, None: 5}
        len of return is num_class[None]+1
    '''
    res = [0]*(num_class[None]+1)
    idx = num_class.get(val)
    if idx is None:
        idx = num_class[None]
    res[idx] = 1
    return res

def __test_encode_onehot():
    num_class = {8: 0, 53: 1, 6: 2, 7: 3, None: 4}
    assert encode_onehot(8, num_class) == [1, 0, 0, 0, 0]
    assert encode_onehot(0, num_class) == [0, 0, 0, 0, 1]
    assert encode_onehot(7, num_class) == [0, 0, 0, 1, 0]

def build_oneatom_feature(a, s, property_list_onehot = ['atom_num', 'formal_charge', 'explicit_valence', 'implicit_valence', 'degree_conn', 'degree_total_conn'], bool_list_onehot = ['aromaticity', 'in_ring'], nrings_encode = 3):
    res = []
    for p in property_list_onehot:
        res += encode_onehot(a[p], s['enumerate_list_atom'][p])
    for p in bool_list_onehot:
        res.append(int(a[p]))
    if nrings_encode is None:
        nrings_encode = 0
    if nrings_encode != 0:
        length = min(len(a['ring_size']), nrings_encode)
        for i in range(length):
            res += encode_onehot(a['ring_size'][i], s['enumerate_list_atom']['ring_size'])
            nrings_encode -= 1
        res += [0]*(s['enumerate_list_atom']['ring_size'][None]+1)*nrings_encode
    return np.array(res, dtype=np.float32)

def __test_encode_atom():
    s = {'max_atoms': 34, 'max_bonds': 35, 'max_ring_size': 6, 'max_rings': 1, 'max_bonds_per_atom': 4, 'enumerate_list_atom': {'atom_num': {0: 0, 6: 1, 7: 2, 8: 3, 53: 4, None: 5}, 'formal_charge': {0: 0, None: 1}, 'explicit_valence': {1: 0, 2: 1, 3: 2, 4: 3, None: 4}, 'implicit_valence': {0: 0, None: 1}, 'degree_conn': {1: 0, 2: 1, 3: 2, 4: 3, None: 4}, 'degree_total_conn': {1: 0, 2: 1, 3: 2, 4: 3, None: 4}, 'ring_size': {3: 0, 4: 1, 5: 2, 6: 3, 7: 4, 8: 5, 9: 6, 10: 7, None: 8}}, 'bond_order': {10: 0, 20: 1, 15: 2, None: 3}}
    a = {
        "atom_idx": 8,
        "atom_map_id": 26,
        "atom_num": 6,
        "aromaticity": True,
        "formal_charge": 0,
        "explicit_valence": 4,
        "implicit_valence": 0,
        "degree_conn": 2,
        "degree_total_conn": 3,
        "in_ring": True,
        "ring_size": [
            6
            ],
        "bonds_idx": [
            7,
            8
            ]
        }
    assert build_oneatom_feature(a, s).tolist() == [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

def build_allatom_feature(atom_features, s):
    f = build_oneatom_feature(atom_features[0], s)
    len_feature = len(f)
    res = np.zeros((s['max_atoms'], len_feature), dtype=np.float32)
    for a in atom_features:
        i = a['atom_idx']
        f = build_oneatom_feature(a, s)
        res[i, :] = f
    return res

def build_onebond_feature(b, s):
    '''
        b: {'order', 'is_conj', 'in_ring'}
        s: {'bond_order': {10: 0, 20: 1, 15: 2, None: 3}}
    '''
    res = [int(b['is_conj']), int(b['in_ring'])]
    res += encode_onehot(b['order'], s['bond_order'])
    return np.array(res, dtype=np.float32)

def __test_encode_bond():
    s = {'bond_order': {10: 0, 20: 1, 15: 2, None: 3}}
    b = {'order':20, 'is_conj':True, 'in_ring':False}
    assert build_onebond_feature(b, s).tolist() == [1, 0, 0, 1, 0, 0]

def build_allbond_featrue(bond_features, s):
    '''
        bond_features: features
        s: return from feature_statistics
        return:
            bond: s['max_atoms']*s['max_atoms']*nfeature
    '''
    if len(bond_features) > 0:
        f = build_onebond_feature(bond_features[0], s)
    else:
        f = build_onebond_feature({'order':20, 'is_conj':True, 'in_ring':False}, s)
    len_feature = len(f)
    res = np.zeros((s['max_atoms'], s['max_atoms'], len_feature), dtype=np.float32)
    for b in bond_features:
        i = b['i']
        j = b['j']
        f = build_onebond_feature(b, s)
        res[i, j, :] = f
        res[j, i, :] = f
    return res

def build_allbond_featrue_mapped(bond_features, s, index_map):
    '''
        bond_features: features
        s: return from feature_statistics
        index_map: atom_idx->atom_map_num
        return:
            bond: s['max_atoms']*s['max_atoms']*nfeature
    '''
    if len(bond_features) > 0:
        f = build_onebond_feature(bond_features[0], s)
    else:
        f = build_onebond_feature({'order':20, 'is_conj':True, 'in_ring':False}, s)
    len_feature = len(f)
    res = np.zeros((s['max_atoms'], s['max_atoms'], len_feature), dtype=np.float32)
    for b in bond_features:
        i = index_map[b['i']]
        j = index_map[b['j']]
        f = build_onebond_feature(b, s)
        res[i, j, :] = f
        res[j, i, :] = f
    return res

def build_conn_mat(bond_features, s):
    res = np.zeros((s['max_atoms'],s['max_atoms']), dtype=np.float32)
    for b in bond_features:
        i = b['i']
        j = b['j']
        res[i][j] = 1
        res[j][i] = 1
    return res

def build_conn_mat_mapped(bond_features, s, index_map):
    '''
        index_map: atom_idx->atom_map_num
    '''
    res = np.zeros((s['max_atoms'],s['max_atoms']), dtype=np.float32)
    for b in bond_features:
        i = index_map[b['i']]
        j = index_map[b['j']]
        res[i][j] = 1
        res[j][i] = 1
    return res

def build_mol_from_encoding(atom, bond, conn, s):
    '''
        Note: Not work for NH in aromatic rings
        assume atom is [atom_num, ...]
        assume bond is [bool, bool, bond_order, ...]
    '''
    atom_dict = swap_idxval_dict(s['enumerate_list_atom']['atom_num'])
    atom_start = 0
    atom_end = atom_start+len(atom_dict)
    bond_dict = swap_idxval_dict(s['bond_order'])
    rdkit_bond = {10:Chem.BondType.SINGLE, 15:Chem.BondType.AROMATIC, 20:Chem.BondType.DOUBLE, 25:Chem.BondType.TWOANDAHALF, 30:Chem.BondType.TRIPLE}
    bond_start = 2
    bond_end = bond_start+len(bond_dict)
    m = Chem.RWMol()
    atom_idx = []
    # add atoms
    for i in range(atom.shape[0]):
        a = atom[i,atom_start:atom_end]
        idx = None
        if np.any(a):
            atom_num = atom_dict[np.nonzero(a)[0][0]]
            if atom_num is not None:
                idx = m.AddAtom(Chem.Atom(atom_num))
        atom_idx.append(idx)
    bond_idx = np.nonzero(conn)
    for i in range(bond_idx[0].shape[0]):
        idx_a = bond_idx[0][i]
        idx_b = bond_idx[1][i]
        if idx_a > idx_b:
            b = bond[idx_a, idx_b, bond_start:bond_end]
            bond_type = rdkit_bond[bond_dict[np.nonzero(b)[0][0]]]
            is_conjugate = bond[idx_a, idx_b, 0]
            rd_idx_a = atom_idx[idx_a]
            rd_idx_b = atom_idx[idx_b]
            idx = m.AddBond(rd_idx_a,rd_idx_b,bond_type)
    m = m.GetMol()
    return Chem.MolToSmiles(m)

def encode_features_atommapped(f, s, isrand=False):
    '''
        f: a feature
        s: return from feature_statistics
        return:
            atom: s['max_atoms']*nfeature
            bond: s['max_atoms']*s['max_atoms']*nfeature
            conn: s['max_atoms']*s['max_atoms']
    '''
    # collect all atom_map_num for reactants and products
    atom_map_reactant = get_atom_map_as_dict(f['reactants'])
    atom_map_product  = get_atom_map_as_dict(f['products'])
    # check for continuity of atom_map_num
    assert check_int_continuous_from_zero(atom_map_reactant.keys())
    assert check_int_continuous_from_zero(atom_map_product.keys())
    assert len(atom_map_reactant) == len(atom_map_product)
    # generate np array for input: atom, bond, conn
    # randomize atom map
    # Note: atom_map_reactant is {0:[atom_idx], atom_map_num: atom_idx}
    # total_atoms = len(atom_map_reactant) - 1 + len(atom_map_reactant[0]) + len(atom_map_product[0])
    total_atoms = s['max_atoms']
    # old atom_map_num -> new atom_map_num
    if isrand:
        random_idx = np.random.permutation(total_atoms)
    else:
        random_idx = np.arange(total_atoms)
    # atom_map_new[atom_map_idx, :] = ['reactant_atom_idx', 'product_atom_idx']
    # mapped atoms, 0: reactant, 1: product
    # atom_map_idx is the idx in the inputs
    atom_map_new = -np.ones((total_atoms,2), dtype=np.int32)
    # build universal map for reactants and products
    for i in range(len(atom_map_reactant)-1):
        a = atom_map_reactant[i+1]
        b = atom_map_product[i+1]
        assert a is not None
        assert b is not None
        atom_map_new[random_idx[i], 0] = a
        atom_map_new[random_idx[i], 1] = b
    # non-mapped
    istart = len(atom_map_reactant)-1
    for i in range(len(atom_map_reactant[0])):
        a = atom_map_reactant[0][i]
        assert a is not None
        atom_map_new[random_idx[i+istart], 0] = a
    istart = len(atom_map_reactant)-1+len(atom_map_reactant[0])
    for i in range(len(atom_map_product[0])):
        b = atom_map_product[0][i]
        assert b is not None
        atom_map_new[random_idx[i+istart], 1] = b
    # build reverse idx lookup table
    atom_idx2atom_map_num_reactant = swap_idxval(atom_map_new[:,0].flatten())
    atom_idx2atom_map_num_product = swap_idxval(atom_map_new[:,1].flatten())
    # build atom features
    property_list_onehot_header = ['atom_num']
    property_list_onehot = ['formal_charge', 'explicit_valence', 'implicit_valence', 'degree_conn', 'degree_total_conn']
    bool_list_onehot = ['aromaticity', 'in_ring']
    nrings_encode = 0
    f1 = build_oneatom_feature(f['reactants']['atom_features'][0], s, property_list_onehot_header, bool_list_onehot=[], nrings_encode=0)
    f2 = build_oneatom_feature(f['products']['atom_features'][0], s, property_list_onehot, nrings_encode=nrings_encode)
    len_feature_headers = len(f1)
    len_feature_chemicals  = len(f2)
    len_feature = len_feature_headers + len_feature_chemicals*2
    atom = np.zeros((s['max_atoms'], len_feature), dtype=np.float32)
    # build atom feature
    for i in range(s['max_atoms']):
        features_header = None
        features_reactant = None
        features_product = None
        if atom_map_new[i, 0] != -1:
            features_header = build_oneatom_feature(
                f['reactants']['atom_features'][atom_map_new[i, 0]],
                s,
                property_list_onehot_header,
                bool_list_onehot=[],
                nrings_encode=0
                )
            features_reactant = build_oneatom_feature(
                f['reactants']['atom_features'][atom_map_new[i, 0]],
                s,
                property_list_onehot,
                bool_list_onehot,
                nrings_encode
                )
        else:
            features_reactant = np.zeros(len_feature_chemicals)
        if atom_map_new[i, 1] != -1:
            if features_header is None:
                features_header = build_oneatom_feature(
                f['products']['atom_features'][atom_map_new[i, 1]],
                s,
                property_list_onehot_header,
                bool_list_onehot=[],
                nrings_encode=0
                )
            features_product = build_oneatom_feature(
                f['products']['atom_features'][atom_map_new[i, 1]],
                s,
                property_list_onehot,
                bool_list_onehot,
                nrings_encode
                )
        else:
            features_product = np.zeros(len_feature_chemicals)
        if features_header is None:
            features_header = np.zeros(len_feature_headers)
        atom[i, :] = np.append(features_header, [features_reactant, features_product])
    bond1 = build_allbond_featrue_mapped(f['reactants']['bond_features'], s, atom_idx2atom_map_num_reactant)
    bond2 = build_allbond_featrue_mapped(f['products']['bond_features'], s, atom_idx2atom_map_num_product)
    conn1 = build_conn_mat_mapped(f['reactants']['bond_features'], s, atom_idx2atom_map_num_reactant)
    conn2 = build_conn_mat_mapped(f['products']['bond_features'], s, atom_idx2atom_map_num_product)
    return atom, bond1, bond2, conn1, conn2

def __test_encode_features():
    s2='[CH3:12][C:11]([CH3:13])([CH3:14])[O:10][C:8](=[O:9])[N:3]1[CH2:2][CH2:1][CH:6]([CH2:5][C:4]1=[O:7])OS(=O)(=O)C.[CH:20]1=[C:19]([N:18]=[CH:17][C:16]2=[C:15]1[CH:24]=[N:23][NH:22]2)[Cl:21]>>[CH3:12][C:11]([CH3:13])([CH3:14])[O:10][C:8](=[O:9])[N:3]1[CH2:2][CH2:1][CH:6]([CH2:5][C:4]1=[O:7])[N:22]2[C:16]3=[C:15]([CH:20]=[C:19]([N:18]=[CH:17]3)[Cl:21])[CH:24]=[N:23]2'
    s3='[CH3:1][CH2:2][CH2:3][CH2:4][CH2:5][CH2:6][CH2:7][CH2:8][CH2:9][CH2:10][CH2:11][CH2:12][CH2:13][CH2:14][CH2:15][CH2:16][CH2:17][C:18](=[O:19])O>CCCCCCCCCCCCCCCCCC(=O)O[C@H](C1=CC=CC=C1)C(=O)OC2=CC[C@]3([C@H]4CC5=CC=C(C6=C5[C@]3([C@H]2O6)CCN4C)O)O.CN(C)C=O.C1CCOC1.C(=O)(C(=O)Cl)Cl.C(=O)(C(F)(F)F)O>[CH3:1][CH2:2][CH2:3][CH2:4][CH2:5][CH2:6][CH2:7][CH2:8][CH2:9][CH2:10][CH2:11][CH2:12][CH2:13][CH2:14][CH2:15][CH2:16][CH2:17][C:18](=[O:19])Cl |f:1.5|'
    s4='O[C:18]([CH2:17][CH2:16][CH2:15][CH2:14][CH2:13][CH2:12][CH2:11][CH2:10][CH2:9][CH2:8][CH2:7][CH2:6][CH2:5][CH2:4][CH2:3][CH2:2][CH3:1])=[O:19]>CCCCCCCCCCCCCCCCCC(=O)O[C@H](C1=CC=CC=C1)C(=O)OC2=CC[C@]3([C@H]4CC5=CC=C(C6=C5[C@]3([C@H]2O6)CCN4C)O)O.CN(C)C=O.C1CCOC1.C(=O)(C(=O)Cl)Cl.C(=O)(C(F)(F)F)O>Cl[C:18]([CH2:17][CH2:16][CH2:15][CH2:14][CH2:13][CH2:12][CH2:11][CH2:10][CH2:9][CH2:8][CH2:7][CH2:6][CH2:5][CH2:4][CH2:3][CH2:2][CH3:1])=[O:19]'
    f2 = rxn2features(s2)
    f3 = rxn2features(s3)
    f4 = rxn2features(s4)
    s = feature_statistics([f2,f3])
    atom, bond1, bond2, conn1, conn2 = encode_features_atommapped(f3, s, isrand=True)
    assert build_mol_from_encoding(atom, bond1, conn1, s) == 'CCCCCCCCCCCCCCCCCC(=O)O.Cl'
    assert build_mol_from_encoding(atom, bond2, conn2, s) == 'CCCCCCCCCCCCCCCCCC(=O)Cl.O'
    atom, bond1, bond2, conn1, conn2 = encode_features_atommapped(f4, s, isrand=False)
    assert build_mol_from_encoding(atom, bond1, conn1, s) == 'CCCCCCCCCCCCCCCCCC(=O)O.Cl'
    assert build_mol_from_encoding(atom, bond2, conn2, s) == 'CCCCCCCCCCCCCCCCCC(=O)Cl.O'

def merge_dense_graph(atom, bond1, bond2, conn1, conn2):
    conn = conn1 + conn2
    conn = np.where(conn > 0, 1, 0)
    bond = np.concatenate((bond1,bond2), axis=-1)
    assert bond.shape[-1] == bond1.shape[-1]+bond2.shape[-1]
    return atom, bond, conn

def encode_features_atommapped_dense_graph(f, s, isrand=False):
    atom, bond1, bond2, conn1, conn2 = encode_features_atommapped(f, s, isrand)
    return merge_dense_graph(atom, bond1, bond2, conn1, conn2)

def encode_features_onemol(f, s, isrand=False):
    '''
        f: a feature: f['reactants'] or f['products']
        s: return from feature_statistics
        return:
            atom: s['max_atoms']*nfeature
            bond: s['max_atoms']*s['max_atoms']*nfeature
            conn: s['max_atoms']*s['max_atoms']
    '''
    # generate np array for input: atom, bond, conn
    total_atoms = s['max_atoms']
    # old atom_num -> new atom_num
    if isrand:
        atom_map_new = np.random.permutation(total_atoms)
    else:
        atom_map_new = np.arange(total_atoms)
    # build reverse idx lookup table
    new_atom_idx2old_atom_idx = swap_idxval(atom_map_new)
    # build atom features
    property_list_onehot = ['atom_num', 'formal_charge', 'explicit_valence', 'implicit_valence', 'degree_conn', 'degree_total_conn']
    bool_list_onehot = ['aromaticity', 'in_ring']
    nrings_encode = 0
    f_atom = build_oneatom_feature(f['atom_features'][0], s, property_list_onehot, nrings_encode=nrings_encode)
    len_feature_atom = len(f_atom)
    atom = np.zeros((s['max_atoms'], len_feature_atom), dtype=np.float32)
    # build atom feature
    for i in range(s['max_atoms']):
        features_atom = None
        if atom_map_new[i] < len(f['atom_features']):
            features_atom = build_oneatom_feature(
                f['atom_features'][atom_map_new[i]],
                s,
                property_list_onehot,
                bool_list_onehot,
                nrings_encode
                )
        else:
            features_atom = np.zeros(len_feature_atom)
        atom[i, :] = np.array(features_atom)
    bond = build_allbond_featrue_mapped(f['bond_features'], s, new_atom_idx2old_atom_idx)
    conn = build_conn_mat_mapped(f['bond_features'], s, new_atom_idx2old_atom_idx)
    return atom, bond, conn

def encode_features_non_mapped(f, s, isrand=False):
    atom1, bond1, conn1 = encode_features_onemol(f['reactants'], s, isrand=isrand)
    atom2, bond2, conn2 = encode_features_onemol(f['products'], s, isrand=isrand)
    return atom1, bond1, conn1, atom2, bond2, conn2

def __test_encode_features_non_mapped():
    s2='[CH3:12][C:11]([CH3:13])([CH3:14])[O:10][C:8](=[O:9])[N:3]1[CH2:2][CH2:1][CH:6]([CH2:5][C:4]1=[O:7])OS(=O)(=O)C.[CH:20]1=[C:19]([N:18]=[CH:17][C:16]2=[C:15]1[CH:24]=[N:23][NH:22]2)[Cl:21]>>[CH3:12][C:11]([CH3:13])([CH3:14])[O:10][C:8](=[O:9])[N:3]1[CH2:2][CH2:1][CH:6]([CH2:5][C:4]1=[O:7])[N:22]2[C:16]3=[C:15]([CH:20]=[C:19]([N:18]=[CH:17]3)[Cl:21])[CH:24]=[N:23]2'
    s3='[CH3:1][CH2:2][CH2:3][CH2:4][CH2:5][CH2:6][CH2:7][CH2:8][CH2:9][CH2:10][CH2:11][CH2:12][CH2:13][CH2:14][CH2:15][CH2:16][CH2:17][C:18](=[O:19])O>CCCCCCCCCCCCCCCCCC(=O)O[C@H](C1=CC=CC=C1)C(=O)OC2=CC[C@]3([C@H]4CC5=CC=C(C6=C5[C@]3([C@H]2O6)CCN4C)O)O.CN(C)C=O.C1CCOC1.C(=O)(C(=O)Cl)Cl.C(=O)(C(F)(F)F)O>[CH3:1][CH2:2][CH2:3][CH2:4][CH2:5][CH2:6][CH2:7][CH2:8][CH2:9][CH2:10][CH2:11][CH2:12][CH2:13][CH2:14][CH2:15][CH2:16][CH2:17][C:18](=[O:19])Cl |f:1.5|'
    f2 = rxn2features(s2)
    f3 = rxn2features(s3)
    s = feature_statistics([f2,f3])
    atom1, bond1, conn1, atom2, bond2, conn2 = encode_features_non_mapped(f3, s, isrand=True)
    assert build_mol_from_encoding(atom1, bond1, conn1, s) == 'CCCCCCCCCCCCCCCCCC(=O)O'
    assert build_mol_from_encoding(atom2, bond2, conn2, s) == 'CCCCCCCCCCCCCCCCCC(=O)Cl'

def __test_encode_features_non_mapped2():
    s='[CH3:1][OH:2]>>[CH2:1][O:2]'
    encoder = {'max_atoms': 4, 'max_bonds': 4, 'max_ring_size': 0, 'max_rings': 0, 'max_bonds_per_atom': 4, 'enumerate_list_atom': {'atom_num': {6: 0, 8: 1, None: 2}, 'formal_charge': {0: 0, None: 1}, 'explicit_valence': {1: 0, 2: 1, None: 2}, 'implicit_valence': {0: 0, None: 1}, 'degree_conn': {1: 0, 2: 1, 3: 2, 4: 3, None: 4}, 'degree_total_conn': {1: 0, 2: 1, 3: 2, 4: 3, None: 4}, 'ring_size': {None: 0}}, 'bond_order': {10: 0, 20: 1, 15: 2, None: 3}}
    f = rxn2features(s)
    print('__test_encode_features_non_mapped2')
    print(f)
    print(encode_features_onemol(f['reactants'], encoder, isrand=False))
    print(encode_features_onemol(f['reactants'], encoder, isrand=True))


if __name__ == "__main__":
    import json
    import sys
    np.set_printoptions(threshold=sys.maxsize)
    s1="[CH3:23][C:22]1=[C:18]([N:19]=[C:20]([O:21]1)[C:24]2=[CH:25][CH:26]=[CH:27][CH:28]=[CH:29]2)[CH2:17][CH2:16][O:15][C:12]3=[CH:11][C:10](=[C:9]([CH:14]=[CH:13]3)[CH2:8][CH2:7][C:6](=[O:32])[O:5][C:1]([CH3:2])([CH3:3])[CH3:4])[CH2:30][OH:31].[CH3:33]I>[H-].CN(C)C=O.[Na+]>[CH3:23][C:22]1=[C:18]([N:19]=[C:20]([O:21]1)[C:24]2=[CH:25][CH:26]=[CH:27][CH:28]=[CH:29]2)[CH2:17][CH2:16][O:15][C:12]3=[CH:11][C:10](=[C:9]([CH:14]=[CH:13]3)[CH2:8][CH2:7][C:6](=[O:32])[O:5][C:1]([CH3:2])([CH3:3])[CH3:4])[CH2:30][O:31][CH3:33] |f:2.4|"
    s2='[CH3:12][C:11]([CH3:13])([CH3:14])[O:10][C:8](=[O:9])[N:3]1[CH2:2][CH2:1][CH:6]([CH2:5][C:4]1=[O:7])OS(=O)(=O)C.[CH:20]1=[C:19]([N:18]=[CH:17][C:16]2=[C:15]1[CH:24]=[N:23][NH:22]2)[Cl:21]>>[CH3:12][C:11]([CH3:13])([CH3:14])[O:10][C:8](=[O:9])[N:3]1[CH2:2][CH2:1][CH:6]([CH2:5][C:4]1=[O:7])[N:22]2[C:16]3=[C:15]([CH:20]=[C:19]([N:18]=[CH:17]3)[Cl:21])[CH:24]=[N:23]2'
    s3='[CH3:1][CH2:2][CH2:3][CH2:4][CH2:5][CH2:6][CH2:7][CH2:8][CH2:9][CH2:10][CH2:11][CH2:12][CH2:13][CH2:14][CH2:15][CH2:16][CH2:17][C:18](=[O:19])O>CCCCCCCCCCCCCCCCCC(=O)O[C@H](C1=CC=CC=C1)C(=O)OC2=CC[C@]3([C@H]4CC5=CC=C(C6=C5[C@]3([C@H]2O6)CCN4C)O)O.CN(C)C=O.C1CCOC1.C(=O)(C(=O)Cl)Cl.C(=O)(C(F)(F)F)O>[CH3:1][CH2:2][CH2:3][CH2:4][CH2:5][CH2:6][CH2:7][CH2:8][CH2:9][CH2:10][CH2:11][CH2:12][CH2:13][CH2:14][CH2:15][CH2:16][CH2:17][C:18](=[O:19])Cl |f:1.5|'
    f = rxn2features(s3)
    print(json.dumps(f, indent=4))
    s = feature_statistics([f])
    print(s)
    atom, bond1, bond2, conn1, conn2 = encode_features_atommapped(f, s)
    #print(conn1)
    #print(atom, bond1, bond2, conn1, conn2)
    #print(build_conn_mat(f['reactants']['bond_features'], s))
    #print(build_allbond_featrue(f['reactants']['bond_features'], s))
    __test_encode_features_non_mapped()
    __test_encode_features()
    __test_encode_onehot()
    __test_encode_bond()
    __test_encode_atom()
    __test_encode_features_non_mapped2()
