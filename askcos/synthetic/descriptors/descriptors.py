"""
This module defines the PathwayRankingHandler for use in Torchserve.
"""

import os

import numpy as np
import torch
from rdkit import Chem

from askcos.synthetic.descriptors.featurization import mol2graph, get_atom_fdim, get_bond_fdim
from askcos import global_config as gc

model_pt_path = gc.DESCRIPTORS['model_path']

class ReactivityDescriptor:

    def __init__(self):
        self.device = None
        self.model = None
        self.initialize()

    def initialize(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        from askcos.synthetic.descriptors.model import MoleculeModel

        # Load model and args
        state = torch.load(model_pt_path, lambda storage, loc: storage)
        args, loaded_state_dict = state['args'], state['state_dict']
        atom_fdim = get_atom_fdim()
        bond_fdim = get_bond_fdim() + atom_fdim

        self.model = MoleculeModel(args, atom_fdim, bond_fdim)
        self.model.load_state_dict(loaded_state_dict)
        self.model.to(self.device)
        self.model.eval()

        self.initalized = True
        print('Model file {0} loaded successfully.'.format(model_pt_path))

    def preprocess(self, smiles):
        mol_graph = mol2graph(smiles)
        f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope, b2br, bond_types = mol_graph.get_components()
        f_atoms, f_bonds, a2b, b2a, b2revb, b2br, bond_types = \
            f_atoms.to(self.device), f_bonds.to(self.device), a2b.to(self.device), b2a.to(self.device), \
            b2revb.to(self.device), b2br.to(self.device), bond_types.to(self.device)

        return f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope, b2br, bond_types

    def inference(self, data):
        descs = self.model(data)

        return descs

    def postprocess(self, smiles, descs):

        descs = [x.data.cpu().numpy() for x in descs]

        partial_charge, partial_neu, partial_elec, NMR, bond_order, bond_distance = descs

        n_atoms, n_bonds = [], []
        for s in smiles:
            m = Chem.MolFromSmiles(s)

            m = Chem.AddHs(m)

            n_atoms.append(len(m.GetAtoms()))
            n_bonds.append(len(m.GetBonds()))

        partial_charge = [x.tolist() for x in np.split(partial_charge.flatten(), np.cumsum(np.array(n_atoms)))][:-1]
        partial_neu = [x.tolist() for x in np.split(partial_neu.flatten(), np.cumsum(np.array(n_atoms)))][:-1]
        partial_elec = [x.tolist() for x in np.split(partial_elec.flatten(), np.cumsum(np.array(n_atoms)))][:-1]
        NMR = [x.tolist() for x in np.split(NMR.flatten(), np.cumsum(np.array(n_atoms)))][:-1]

        bond_order = [x.tolist() for x in np.split(bond_order.flatten(), np.cumsum(np.array(n_bonds)))][:-1]
        bond_distance = [x.tolist() for x in np.split(bond_distance.flatten(), np.cumsum(np.array(n_bonds)))][:-1]

        results = [{'smiles': s, 'partial_charge': pc, 'fukui_neu': pn,
                    'fukui_elec': pe, 'NMR': nmr, 'bond_order': bo, 'bond_length': bd}
                   for s, pc, pn, pe, nmr, bo, bd in zip(smiles, partial_charge, partial_neu,
                                                         partial_elec, NMR, bond_order, bond_distance)]
        return results

    def evaluate(self, smiles):
        descriptors = self.inference(self.preprocess(smiles))
        result = self.postprocess(smiles, descriptors)

        return result


if __name__ == '__main__':

    handler = ReactivityDescriptor()

    data = ['CCCC', 'CCC', 'CCCCC']
    descriptors = handler.evaluate(data)

    print(descriptors)