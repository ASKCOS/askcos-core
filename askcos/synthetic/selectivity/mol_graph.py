from rdkit import Chem
import numpy as np

elem_list = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K',
             'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In',
             'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'W', 'Ru', 'Nb', 'Re', 'Te', 'Rh', 'Tc', 'Ba', 'Bi', 'Hf', 'Mo', 'U',
             'Sm', 'Os', 'Ir', 'Ce','Gd','Ga','Cs', 'unknown']
qm_elem_list = ['C', 'O', 'N', 'F', 'Br', 'Cl', 'S', 'Si', 'B', 'I','P','H']
atom_fdim = len(elem_list) + 6 + 6 + 6 + 1
bond_fdim = 6
max_nb = 10
binary_fdim = 5 + 6

from .electronegs import electronegs 


def onek_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def atom_features(atom, elem_list):
    return np.array(onek_encoding_unk(atom.GetSymbol(), elem_list)
                    + onek_encoding_unk(atom.GetDegree(), [0, 1, 2, 3, 4, 5])
                    + onek_encoding_unk(atom.GetExplicitValence(), [1, 2, 3, 4, 5, 6])
                    + onek_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5])
                    + [atom.GetIsAromatic()], dtype=np.float32)


def bond_features(bond):
    bt = bond.GetBondType()
    return np.array([bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE, bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC, bond.GetIsConjugated(), bond.IsInRing()], dtype=np.float32)


def smiles2graph(smiles, idxfunc=lambda x: x.GetIdx()):
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        raise ValueError("Could not parse smiles string:", smiles)

    fatoms, fbonds, atom_nb, bond_nb, num_nbs, _ = _mol2graph(mol, idxfunc)

    return fatoms, fbonds, atom_nb, bond_nb, num_nbs


def _mol2graph(mol, idxfunc=lambda x: x.GetIdx(), core=[], atom_fdim=atom_fdim):
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
        fatoms[idx] = atom_features(atom, elem_list)
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

    rs_features = _mol2graph(rs, idxfunc, core=rs_core, atom_fdim=atom_fdim)
    p_features = _mol2graph(p, idxfunc, core=p_core, atom_fdim=atom_fdim)

    return rs_features, p_features


def rbf_expansion(expanded, mu=0, delta=0.01, kmax=8):
    k = np.arange(0, kmax)
    return np.exp(-(expanded - (mu + delta * k))**2 / delta)


def _mol2graph_qm(rxn_smiles, qm_descriptors=None, qf_feature=1):
    rt_atom_fdim_geo = len(qm_elem_list) + 6 + 6 + 6 + 1
    rg_atom_fdim_geo = len(elem_list) + 6 + 6 + 6 + 1
    atom_fdim_qm = (10 + 10 + 10 + 10) * qf_feature

    bond_fdim_geo = 6
    bond_fdim_qm = (25 + 40) * qf_feature
    max_nb = 10

    bond_changes = get_changed_bonds(rxn_smiles)
    rs, rg, _ = rxn_smiles.split('>')
    mol_rs = Chem.MolFromSmiles(rs)
    mol_rg = Chem.MolFromSmiles(rg)

    if not mol_rs:
        raise ValueError("Could not parse smiles string:", rxn_smiles)

    if not mol_rg:
        raise ValueError("Could not parse smiles string:", rxn_smiles)

    # rs
    fatom_index = {a.GetIntProp('molAtomMapNumber'): a.GetIdx() for a in mol_rs.GetAtoms()}
    fbond_index = {'{}-{}'.format(*sorted([b.GetBeginAtom().GetIntProp('molAtomMapNumber'),
                                           b.GetEndAtom().GetIntProp('molAtomMapNumber')])): b.GetIdx()
                   for b in mol_rs.GetBonds()}

    n_atoms = mol_rs.GetNumAtoms()
    n_bonds = max(mol_rs.GetNumBonds(), 1)
    fatoms_geo = np.zeros((n_atoms, rt_atom_fdim_geo))
    fatoms_qm = np.zeros((n_atoms, atom_fdim_qm))
    fbonds_geo = np.zeros((n_bonds, bond_fdim_geo))
    fbonds_qm = np.zeros((n_bonds, bond_fdim_qm))

    atom_nb = np.zeros((n_atoms, max_nb), dtype=np.int32)
    bond_nb = np.zeros((n_atoms, max_nb), dtype=np.int32)
    num_nbs = np.zeros((n_atoms,), dtype=np.int32)
    cores = np.zeros((len(bond_changes), 2), dtype=np.int32)

    connect_change = np.zeros((len(bond_changes), 10), dtype=np.float32)

    for i, bc in enumerate(bond_changes):
        try:
            cores[i, 0] = fatom_index[bc[0]]
            cores[i, 1] = fatom_index[bc[1]]
        except:
            print(fatom_index)
            print(bc)
            raise
        connect_change[i, :5] = onek_encoding_unk(bc[2], [0, 1, 1.5, 2, 3])
        connect_change[i, 5:] = onek_encoding_unk(bc[3], [0, 1, 1.5, 2, 3])

    for smiles in rs.split('.'):

        mol = Chem.MolFromSmiles(smiles)
        fatom_index_mol = {a.GetIntProp('molAtomMapNumber'): a.GetIdx() for a in mol.GetAtoms()}

        qm_series = qm_descriptors.loc[smiles]

        partial_charge = qm_series['partial_charge'].reshape(-1, 1)
        partial_charge = np.apply_along_axis(rbf_expansion, -1, partial_charge, 0.2, 0.05 / qf_feature, 10 * qf_feature)

        fukui_elec = qm_series['fukui_elec'].reshape(-1, 1)
        fukui_elec = np.apply_along_axis(rbf_expansion, -1, fukui_elec, 0, 0.02 / qf_feature, 10 * qf_feature)

        fukui_neu = qm_series['fukui_neu'].reshape(-1, 1)
        fukui_neu = np.apply_along_axis(rbf_expansion, -1, fukui_neu, 0, 0.03 / qf_feature, 10 * qf_feature)

        nmr = qm_series['NMR'].reshape(-1, 1)
        nmr = np.apply_along_axis(rbf_expansion, -1, nmr, 0.2, 0.08 / qf_feature, 10 * qf_feature)

        bond_index = np.expand_dims(qm_series['bond_order_matrix'], -1)
        bond_index = np.apply_along_axis(rbf_expansion, -1, bond_index, 0.5, 0.1 / qf_feature, 25 * qf_feature)

        bond_distance = np.expand_dims(qm_series['distance_matrix'], -1)
        bond_distance = np.apply_along_axis(rbf_expansion, -1, bond_distance, 0.5, 0.05 / qf_feature, 40 * qf_feature)

        atom_qm_descriptor = np.concatenate([partial_charge, fukui_elec, fukui_neu, nmr], axis=-1)

        for map_idx in fatom_index_mol:
            fatoms_geo[fatom_index[map_idx], :] = atom_features(mol_rs.GetAtomWithIdx(fatom_index[map_idx]), qm_elem_list)
            fatoms_qm[fatom_index[map_idx], :] = atom_qm_descriptor[fatom_index_mol[map_idx], :]

        for bond in mol.GetBonds():
            a1i, a2i = bond.GetBeginAtom().GetIntProp('molAtomMapNumber'), \
                       bond.GetEndAtom().GetIntProp('molAtomMapNumber')
            idx = fbond_index['{}-{}'.format(*sorted([a1i, a2i]))]
            a1 = fatom_index[a1i]
            a2 = fatom_index[a2i]

            a1i = fatom_index_mol[a1i]
            a2i = fatom_index_mol[a2i]

            if num_nbs[a1] == max_nb or num_nbs[a2] == max_nb:
                raise Exception(smiles)
            atom_nb[a1, num_nbs[a1]] = a2
            atom_nb[a2, num_nbs[a2]] = a1
            bond_nb[a1, num_nbs[a1]] = idx
            bond_nb[a2, num_nbs[a2]] = idx
            num_nbs[a1] += 1
            num_nbs[a2] += 1

            fbonds_geo[idx, :] = bond_features(bond)
            fbonds_qm[idx, :25 * qf_feature] = bond_index[a1i, a2i]
            fbonds_qm[idx, 25 * qf_feature:] = bond_distance[a1i, a2i]

    # rg
    n_atoms = mol_rg.GetNumAtoms()
    n_bonds = max(mol_rg.GetNumBonds(), 1)
    rg_fatoms_geo = np.zeros((n_atoms, rg_atom_fdim_geo))
    rg_fbonds_geo = np.zeros((n_bonds, bond_fdim_geo))

    rg_atom_nb = np.zeros((n_atoms, max_nb), dtype=np.int32)
    rg_bond_nb = np.zeros((n_atoms, max_nb), dtype=np.int32)
    rg_num_nbs = np.zeros((n_atoms,), dtype=np.int32)

    for atom in mol_rg.GetAtoms():
        idx = atom.GetIdx()
        if idx >= n_atoms:
            raise Exception(Chem.MolToSmiles(mol))
        rg_fatoms_geo[idx] = atom_features(atom, elem_list)

    for bond in mol_rg.GetBonds():
        a1 = bond.GetBeginAtom().GetIdx()
        a2 = bond.GetEndAtom().GetIdx()
        idx = bond.GetIdx()
        if rg_num_nbs[a1] == max_nb or rg_num_nbs[a2] == max_nb:
            raise Exception(Chem.MolToSmiles(mol))
        rg_atom_nb[a1,rg_num_nbs[a1]] = a2
        rg_atom_nb[a2,rg_num_nbs[a2]] = a1
        rg_atom_nb[a1,rg_num_nbs[a1]] = idx
        rg_atom_nb[a2,rg_num_nbs[a2]] = idx
        rg_num_nbs[a1] += 1
        rg_num_nbs[a2] += 1

        rg_fbonds_geo[idx, :] = bond_features(bond)

    return fatoms_geo, fatoms_qm, fbonds_qm, atom_nb, bond_nb, num_nbs, cores, connect_change, \
           rg_fatoms_geo, rg_fbonds_geo, rg_atom_nb, rg_bond_nb, rg_num_nbs


def smiles2graph_pr_qm(rt_smiles, p_smiles, rg_smiles, qm_descriptors, qf_feature=4):
    rxn_smiles = '>'.join([rt_smiles, rg_smiles, p_smiles])
    graph = _mol2graph_qm(rxn_smiles, qm_descriptors, qf_feature)
    return graph


def get_changed_bonds(rxn_smi):
    reactants = Chem.MolFromSmiles(rxn_smi.split('>')[0])
    products  = Chem.MolFromSmiles(rxn_smi.split('>')[2])

    conserved_maps = [a.GetProp('molAtomMapNumber') for a in products.GetAtoms() if a.HasProp('molAtomMapNumber')]
    bond_changes = set() # keep track of bond changes

    # Look at changed bonds
    bonds_prev = {}
    for bond in reactants.GetBonds():
        nums = sorted(
            [bond.GetBeginAtom().GetProp('molAtomMapNumber'),
             bond.GetEndAtom().GetProp('molAtomMapNumber')])
        if (nums[0] not in conserved_maps) and (nums[1] not in conserved_maps): continue
        bonds_prev['{}~{}'.format(nums[0], nums[1])] = bond.GetBondTypeAsDouble()
    bonds_new = {}
    for bond in products.GetBonds():
        nums = sorted(
            [bond.GetBeginAtom().GetProp('molAtomMapNumber'),
             bond.GetEndAtom().GetProp('molAtomMapNumber')])
        bonds_new['{}~{}'.format(nums[0], nums[1])] = bond.GetBondTypeAsDouble()

    for bond in bonds_prev:
        if bond not in bonds_new:
            bond_changes.add((int(bond.split('~')[0]), int(bond.split('~')[1]), bonds_prev[bond], 0.0)) # lost bond
        else:
            if bonds_prev[bond] != bonds_new[bond]:
                bond_changes.add((int(bond.split('~')[0]), int(bond.split('~')[1]), bonds_prev[bond], bonds_new[bond])) # changed bond
    for bond in bonds_new:
        if bond not in bonds_prev:
            bond_changes.add((int(bond.split('~')[0]), int(bond.split('~')[1]), 0.0, bonds_new[bond]))  # new bond

    ratoms = {a.GetIntProp('molAtomMapNumber'): a for a in reactants.GetAtoms()}
    patoms = {a.GetIntProp('molAtomMapNumber'): a for a in products.GetAtoms()}
    new_changes = []
    for b in bond_changes:
        if (b[2] == 1.5 and b[3] > 0) or (b[2] > 0 and b[3] == 1.5):
            continue
        if b[2] == 0 or b[3] == 0:
            new_changes.append(b)
            continue
        if ratoms[b[0]].GetNumExplicitHs() == patoms[b[0]].GetNumExplicitHs() and ratoms[b[1]].GetNumExplicitHs() == patoms[b[1]].GetNumExplicitHs():
                continue
        new_changes.append(b)
    return new_changes


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


def pack2D_cores(arr_list):
    cores = np.concatenate(arr_list, axis=0)
    a = np.zeros((len(cores), 3), dtype=np.int32)
    n = 0
    for i, arr in enumerate(arr_list):
        a[n:n+len(arr), 0] = i
        a[n:n+len(arr), 1:] = arr
        n += len(arr)
    return a


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
