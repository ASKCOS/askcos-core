from rdkit import Chem
import numpy as np

elem_list = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'W', 'Ru', 'Nb', 'Re', 'Te', 'Rh', 'Tc', 'Ba', 'Bi', 'Hf', 'Mo', 'U', 'Sm', 'Os', 'Ir', 'Ce','Gd','Ga','Cs', 'unknown']
atom_fdim = len(elem_list) + 6 + 6 + 6 + 2
bond_fdim = 6
max_nb = 10
binary_fdim = 5 + 6

from .electronegs import electronegs 


def onek_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def atom_features(atom, include_electronegs=True):
    if include_electronegs:
        last_feature = [atom.GetIsAromatic(), electronegs[atom.GetAtomicNum()]]
    else:
        last_feature = [atom.GetIsAromatic()]

    return np.array(onek_encoding_unk(atom.GetSymbol(), elem_list)
                    + onek_encoding_unk(atom.GetDegree(), [0,1,2,3,4,5])
                    + onek_encoding_unk(atom.GetExplicitValence(), [1,2,3,4,5,6])
                    + onek_encoding_unk(atom.GetImplicitValence(), [0,1,2,3,4,5])
                    + last_feature, dtype=np.float32)

def bond_features(bond):
    bt = bond.GetBondType()
    return np.array([bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE, bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC, bond.GetIsConjugated(), bond.IsInRing()], dtype=np.float32)


def smiles2graph(smiles, idxfunc=lambda x: x.GetIdx()):
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        raise ValueError("Could not parse smiles string:", smiles)

    fatoms, fbonds, atom_nb, bond_nb, num_nbs, _ = _mol2graph(mol, idxfunc)

    return fatoms, fbonds, atom_nb, bond_nb, num_nbs


def _mol2graph(mol, idxfunc=lambda x: x.GetIdx(), core=[], include_electroneg=True, atom_fdim=atom_fdim):
    n_atoms = mol.GetNumAtoms()
    n_bonds = max(mol.GetNumBonds(), 1)
    fatoms = np.zeros((n_atoms, atom_fdim))
    fbonds = np.zeros((n_bonds, bond_fdim))
    atom_nb = np.zeros((n_atoms, max_nb), dtype=np.int32)
    bond_nb = np.zeros((n_atoms, max_nb), dtype=np.int32)
    num_nbs = np.zeros((n_atoms,), dtype=np.int32)
    core_mask = np.zeros((n_atoms,), dtype=np.int32)

    for atom in mol.GetAtoms():
        idx = idxfunc(atom)
        if idx >= n_atoms:
            raise Exception(Chem.MolToSmiles(mol))
        fatoms[idx] = atom_features(atom, include_electroneg)
        if idx in core:
            core_mask[idx] = 1

    for bond in mol.GetBonds():
        a1 = idxfunc(bond.GetBeginAtom())
        a2 = idxfunc(bond.GetEndAtom())
        idx = bond.GetIdx()
        if num_nbs[a1] == max_nb or num_nbs[a2] == max_nb:
            raise Exception(Chem.MolToSmiles(mol))
        atom_nb[a1,num_nbs[a1]] = a2
        atom_nb[a2,num_nbs[a2]] = a1
        bond_nb[a1,num_nbs[a1]] = idx
        bond_nb[a2,num_nbs[a2]] = idx
        num_nbs[a1] += 1
        num_nbs[a2] += 1
        fbonds[idx] = bond_features(bond)
    return fatoms, fbonds, atom_nb, bond_nb, num_nbs, core_mask


def smiles2graph_pr(p_smiles, r_smiles, idxfunc=lambda x: x.GetIdx(), core_buffer=0, atom_fdim=atom_fdim):
    """

    :param p_smiles: smiles for products
    :param r_smiles: smiles for reactants
    :param idxfunc: index function
    :param core_buffer: radius to look for neighbors
    :param atom_fdim: feature dimension for atoms
    :return: graph representation for reactants and products
    """
    p = Chem.MolFromSmiles(p_smiles)
    rs = Chem.MolFromSmiles(r_smiles)

    if not p:
        raise ValueError("Could not parse smiles string:", p_smiles)

    if not rs:
        raise ValueError("Could not parse smiles string:", r_smiles)

    rs_core, p_core = _get_reacting_core(rs, p, core_buffer)

    rs_features = _mol2graph(rs, idxfunc, core=rs_core, include_electroneg=False, atom_fdim=atom_fdim)
    p_features = _mol2graph(p, idxfunc, core=p_core, include_electroneg=False, atom_fdim=atom_fdim)

    return rs_features, p_features


def _get_reacting_core(rs, p, buffer):
    '''
    use molAtomMapNumber of molecules
    buffer: neighbor to be cosidered as reacting center
    return: atomidx of reacting core
    '''
    rs_dict = {a.GetIntProp('molAtomMapNumber'):a for a in rs.GetAtoms()}
    p_dict = {a.GetIntProp('molAtomMapNumber'):a for a in p.GetAtoms()}

    corer_mapnum = set()
    corep_mapnum = set()
    for a_map in p_dict:

        if a_map not in rs_dict:
            corep_mapnum.add(a_map)
        else:
            a_neighbor_in_p = set([a.GetIntProp('molAtomMapNumber') for a in p_dict[a_map].GetNeighbors()])
            a_neighbor_in_rs = set([a.GetIntProp('molAtomMapNumber') for a in rs_dict[a_map].GetNeighbors()])
            if a_neighbor_in_p != a_neighbor_in_rs:
                corer_mapnum.add(a_map)
                corep_mapnum.add(a_map)
                if a_neighbor_in_rs - a_neighbor_in_p:
                    corer_mapnum.update(a_neighbor_in_rs - a_neighbor_in_p)
            else:
                for a_neighbor in a_neighbor_in_p:
                    b_in_p = p.GetBondBetweenAtoms(p_dict[a_neighbor].GetIdx(), p_dict[a_map].GetIdx())
                    b_in_r = rs.GetBondBetweenAtoms(rs_dict[a_neighbor].GetIdx(), rs_dict[a_map].GetIdx())
                    if b_in_p.GetBondType() != b_in_r.GetBondType():
                        corer_mapnum.add(a_map)
                        corep_mapnum.add(a_map)

    core_rs = _get_buffer(rs, [rs_dict[a].GetIdx() for a in corer_mapnum], buffer)
    core_p = _get_buffer(p, [p_dict[a].GetIdx() for a in corep_mapnum], buffer)

    return core_rs, core_p


def _get_buffer(m, cores, buffer):
    neighbors = set(cores)

    for i in range(buffer):
        neighbors_temp = list(neighbors)
        for c in neighbors_temp:
            neighbors.update([n.GetIdx() for n in m.GetAtomWithIdx(c).GetNeighbors()])

    return neighbors


def pack2D(arr_list):
    N = max([x.shape[0] for x in arr_list])
    M = max([x.shape[1] for x in arr_list])
    a = np.zeros((len(arr_list), N, M))
    for i, arr in enumerate(arr_list):
        n = arr.shape[0]
        m = arr.shape[1]
        a[i,0:n,0:m] = arr
    return a

def pack2D_withidx(arr_list):
    N = max([x.shape[0] for x in arr_list])
    M = max([x.shape[1] for x in arr_list])
    a = np.zeros((len(arr_list), N, M, 2))
    for i, arr in enumerate(arr_list):
        n = arr.shape[0]
        m = arr.shape[1]
        a[i,0:n,0:m,0] = i
        a[i,0:n,0:m,1] = arr
    return a

def pack1D(arr_list):
    N = max([x.shape[0] for x in arr_list])
    a = np.zeros((len(arr_list), N))
    for i, arr in enumerate(arr_list):
        n = arr.shape[0]
        a[i,0:n] = arr
    return a

def get_mask(arr_list):
    N = max([x.shape[0] for x in arr_list])
    a = np.zeros((len(arr_list), N))
    for i, arr in enumerate(arr_list):
        for j in range(arr.shape[0]):
            a[i][j] = 1
    return a


def get_bin_feature(r, max_natoms):
    '''
    This function is used to generate descriptions of atom-atom relationships, including
    the bond type between the atoms (if any) and whether they belong to the same molecule.
    It is used in the global attention mechanism.
    '''
    comp = {a.GetIdx(): 0 for a in Chem.MolFromSmiles(r).GetAtoms()}
    n = 0
    for i, s in enumerate(r.split('.')):
        mol = Chem.MolFromSmiles(s)
        for _ in mol.GetAtoms():
            comp[n] = i
            n += 1

    n_comp = len(r.split('.'))
    rmol = Chem.MolFromSmiles(r)
    n_atoms = rmol.GetNumAtoms()
    bond_map = {}
    for bond in rmol.GetBonds():
        a1 = bond.GetBeginAtom().GetIdx()
        a2 = bond.GetEndAtom().GetIdx()
        bond_map[(a1,a2)] = bond_map[(a2,a1)] = bond

    features = []

    for i in range(max_natoms):
        for j in range(max_natoms):
            f = np.zeros((binary_fdim,))
            if i >= n_atoms or j >= n_atoms or i == j:
                features.append(f)
                continue
            if (i,j) in bond_map:
                bond = bond_map[(i,j)]
                f[1:1+6] = bond_features(bond)
            else:
                f[0] = 1.0
            f[-4] = 1.0 if comp[i] != comp[j] else 0.0
            f[-3] = 1.0 if comp[i] == comp[j] else 0.0
            f[-2] = 1.0 if n_comp == 1 else 0.0
            f[-1] = 1.0 if n_comp > 1 else 0.0
            features.append(f)
    return np.vstack(features).reshape((max_natoms,max_natoms,binary_fdim))


def binary_features_batch(r_list):
    mol_list = []
    max_natoms = 0
    for r in r_list:
        rmol = Chem.MolFromSmiles(r)
        if rmol.GetNumAtoms() > max_natoms:
            max_natoms = rmol.GetNumAtoms()
    features = []
    for r in r_list:
        features.append(get_bin_feature(r,max_natoms))
    return np.array(features)


def smiles2graph_list(smiles_list, idxfunc=lambda x:x.GetIdx()):
    res = list(map(lambda x:smiles2graph(x,idxfunc), smiles_list))
    fatom_list, fbond_list, gatom_list, gbond_list, nb_list = zip(*res)
    return pack2D(fatom_list), pack2D(fbond_list), pack2D_withidx(gatom_list), pack2D_withidx(gbond_list), pack1D(nb_list), get_mask(fatom_list)
