import tensorflow as tf
from scipy.special import softmax
import pandas as pd
import numpy as np
from rdkit import Chem
from askcos.utilities.template_extractor import extract_from_reaction
from rdchiral.initialization import rdchiralReactants, rdchiralReaction
from rdchiral.main import rdchiralRun

from askcos import global_config as gc
from askcos.synthetic.selectivity.general_model.data_loading import gnn_data_generation, qm_gnn_data_generation
from askcos.synthetic.selectivity.general_model.loss import wln_loss
from askcos.synthetic.selectivity.general_model.models import WLNReactionClassifier
from askcos.synthetic.selectivity.general_model.qm_models import QMWLNPairwiseAtomClassifier, WLNPairwiseAtomClassifierNoReagent
from askcos.utilities import parsing


GNN_model_path = gc.GEN_SELECTIVITY['model_path']['GNN']
QM_GNN_model_path = gc.GEN_SELECTIVITY['model_path']['QM_GNN']
QM_GNN_no_reagent_model_path = gc.GEN_SELECTIVITY['model_path']['QM_GNN_no_reagent']
scaler_path = gc.GEN_SELECTIVITY['scalers']

initializer = '[CH4:1]>ClC(Cl)Cl>[CH4:1]'
initializer_qm_descriptors = {
    '[CH4:1]': {'partial_charge': [0.2840435 , 0.42871496, 0.42871496, 0.42871496, 0.42871493],
    'fukui_neu': [0.40352857, 0.27556148, 0.27556148, 0.27556148, 0.27556148],
    'fukui_elec': [0.69178367, 0.3259021 , 0.3259021 , 0.3259021 , 0.3259021 ],
    'NMR': [1.7733233, 0.705616 , 0.705616 , 0.7056161, 0.7056161],
    'bond_order_matrix': [[0, 0.94360828, 0.94360828, 0.94360828, 0.94360828],
                         [0.94360828, 0, 0, 0, 0],
                         [0.94360828, 0, 0, 0, 0],
                         [0.94360828, 0, 0, 0, 0],
                         [0.94360828, 0, 0, 0, 0]],
    'distance_matrix': [[0, 1.0847981, 1.0847981, 1.0847981, 1.0847981],
                       [1.0847981, 0, 0, 0, 0],
                       [1.0847981, 0, 0, 0, 0],
                       [1.0847981, 0, 0, 0, 0],
                       [1.0847981, 0, 0, 0, 0]]}}

GLOBAL_SCALE = ['partial_charge', 'fukui_neu', 'fukui_elec']
ATOM_SCALE = ['NMR']


def _bond_to_matrix(smiles, bond_vector):
    m = Chem.MolFromSmiles(smiles)

    m = Chem.AddHs(m)

    bond_matrix = np.zeros([len(m.GetAtoms()), len(m.GetAtoms())])
    for i, bp in enumerate(bond_vector):
        b = m.GetBondWithIdx(i)
        bond_matrix[b.GetBeginAtomIdx(), b.GetEndAtomIdx()] = bond_matrix[b.GetEndAtomIdx(), b.GetBeginAtomIdx()] = bp

    return bond_matrix


def _min_max_normalize(df, scalers):

    df = pd.DataFrame(df).applymap(np.array)

    for column in GLOBAL_SCALE:
        scaler = scalers[column]
        df[column] = df[column].apply(lambda x: scaler.transform(x.reshape(-1, 1)).reshape(-1))

    def _min_max_by_atom(atoms, data, scaler):
        data = [scaler[a].transform(np.array([[d]]))[0][0] for a, d in zip(atoms, data)]
        return np.array(data)

    if ATOM_SCALE:
        print('postprocessing atom-wise scaling')
        df['atoms'] = df.smiles.apply(lambda x: _get_atoms(x))
        for column in ATOM_SCALE:
            df[column] = df.apply(lambda x: _min_max_by_atom(x['atoms'], x[column], scalers[column]), axis=1)

    df['bond_order_matrix'] = df.apply(lambda x: _bond_to_matrix(x['smiles'], x['bond_order']), axis=1)
    df['distance_matrix'] = df.apply(lambda x: _bond_to_matrix(x['smiles'], x['bond_length']), axis=1)

    df = df[['smiles', 'partial_charge', 'fukui_neu', 'fukui_elec', 'NMR', 'bond_order_matrix', 'distance_matrix']].set_index('smiles')

    df = df.applymap(lambda x: x.tolist()).T.to_dict()

    return df


def _get_atoms(smiles):
    m = Chem.MolFromSmiles(smiles)

    m = Chem.AddHs(m)

    atoms = [x.GetSymbol() for x in m.GetAtoms()]

    return atoms


def apply_template(template, rxn_smiles):

    rt, _, pt = template.split('>')
    template = '({0})>>({1})'.format(rt, pt)
    r, rg, p = rxn_smiles.split('>')
    precursor = rdchiralReactants(r)
    forward_rxn = rdchiralReaction(str(template))

    outcomes = rdchiralRun(forward_rxn, precursor, return_mapped=True)
    outcomes = list([x[0] for x in outcomes[1].values()])
    reactants = Chem.MolToSmiles(precursor.reactants)

    return '>'.join([reactants, rg, '.'.join(outcomes)])


class GeneralSelectivityPredictor:

    def __init__(self, atom_mapper=None, descriptor_predictor=None):

        if atom_mapper is None:
            from askcos.synthetic.atom_mapper.wln_mapper import WLN_AtomMapper
            atom_mapper = WLN_AtomMapper().evaluate

        if descriptor_predictor is None:
            from askcos.synthetic.descriptors.descriptors import ReactivityDescriptor
            descriptor_predictor = ReactivityDescriptor().evaluate

        self.qm_scaler = pd.read_pickle(scaler_path)
        self.model = None

        self.atom_mapper = atom_mapper
        self.descriptor_predictor = descriptor_predictor

    def _initialize_model(self, initializer_x):
        opt = tf.keras.optimizers.Adam(lr=0.001, clipnorm=5.0)
        self.model.compile(
            optimizer=opt,
            loss=wln_loss,
        )
        self.model.predict_on_batch(initializer_x)

    def predict(self, rxnsmiles, atom_mapper=None, descriptor_predictor=None,
                mapped=False, all_outcomes=False, verbose=True, no_map_reagents=False):

        self.build()

        if atom_mapper is not None:
            self.atom_mapper = atom_mapper

        if descriptor_predictor is not None:
            self.descriptor_predictor = descriptor_predictor

        if not mapped:
            rsmi, rgsmi, psmi = rxnsmiles.split('>')

            if no_map_reagents:
                [rsmi_am, _, psmi_am] = self.atom_mapper(rsmi + '>>' + psmi).split('>')
            else:
                [rsmi_am, rgsmi, psmi_am] = self.atom_mapper(rxnsmiles).split('>')

            if rsmi_am and psmi_am:
                rxnsmiles = '>'.join([rsmi_am, rgsmi, psmi_am])
            else:
                raise RuntimeError('Failed to map the given reaction smiles with the selected mapping method, please select other mapping methods.')

        if not all_outcomes:
            rsmi, _, psmi = rxnsmiles.split('>')
            reaction = {'reactants': rsmi, 'products': psmi, '_id': 0}
            try:
                template = extract_from_reaction(reaction)
                rxnsmiles = apply_template(template, rxnsmiles)
            except Exception:
                raise RuntimeError('Failed to extract or apply reaction template for the given reaction. Please examine your reaction in the atom mapping app.')

        if len(rxnsmiles.split('>')[2].split('.')) <= 1:
            raise ValueError('The given reaction is not a selective reaction.')

        selectivity = self.reference(rxnsmiles)

        _, _, products = rxnsmiles.split('>')
        products = [parsing.canonicalize_mapped_smiles(s) for s in products.split('.')]

        if verbose:
            selectivity, products = zip(*sorted(zip(selectivity, products), reverse=True))
            results = [{'smiles': prod, 'prob': prob, 'rank': i+1} for i, (prod, prob) in enumerate(zip(products, selectivity))]
        else:
            results = selectivity

        return results


class GnnGeneralSelectivityPredictor(GeneralSelectivityPredictor):

    def __init__(self, atom_mapper=None, descriptor_predictor=None):
        super(GnnGeneralSelectivityPredictor, self).__init__(atom_mapper=atom_mapper, descriptor_predictor=descriptor_predictor)

    def build(self):
        model_path = GNN_model_path

        self.model = WLNReactionClassifier()
        initializer_x = gnn_data_generation(initializer.split('>')[0], initializer.split('>')[2])
        self._initialize_model(initializer_x)
        self.model.load_weights(model_path)

    def reference(self, rxnsmiles):
        reactants, _, products = rxnsmiles.split('>')
        test_gen = gnn_data_generation(reactants, products)

        out = self.model.predict_on_batch(test_gen).reshape([-1])
        out = tuple([float(x) for x in softmax(out)])

        return out


class QmGnnGeneralSelectivityPredictor(GeneralSelectivityPredictor):

    def __init__(self, atom_mapper=None, descriptor_predictor=None):
        super(QmGnnGeneralSelectivityPredictor, self).__init__(atom_mapper=atom_mapper, descriptor_predictor=descriptor_predictor)

    def build(self):
        model_path = QM_GNN_model_path
        initializer_x = qm_gnn_data_generation(initializer.split('>')[0], initializer.split('>')[2],
                                               initializer.split('>')[1], initializer_qm_descriptors)
        self.model = QMWLNPairwiseAtomClassifier()
        self._initialize_model(initializer_x)

        print('Loading QM-GNN-Reagent selectivity model')
        self.model.summary()
        self.model.load_weights(model_path)

    def reference(self, rxnsmiles):
        reactants, reagent, products = rxnsmiles.split('>')
        rsmis = reactants.split('.')
        descriptors = [self.descriptor_predictor(rsmi) for rsmi in rsmis]
        qm_df = _min_max_normalize(descriptors, self.qm_scaler)
        test_gen = qm_gnn_data_generation(reactants, products, reagent, qm_df)

        out = self.model.predict_on_batch(test_gen).reshape([-1])
        out = tuple([float(x) for x in softmax(out)])

        return out


class QmGnnGeneralSelectivityPredictorNoReagent(QmGnnGeneralSelectivityPredictor):

    def __init__(self, atom_mapper=None, descriptor_predictor=None):
        super(QmGnnGeneralSelectivityPredictorNoReagent, self).__init__(atom_mapper=atom_mapper, descriptor_predictor=descriptor_predictor)

    def build(self):
        model_path = QM_GNN_no_reagent_model_path
        initializer_x = qm_gnn_data_generation(initializer.split('>')[0], initializer.split('>')[2],
                                               initializer.split('>')[1], initializer_qm_descriptors)
        self.model = WLNPairwiseAtomClassifierNoReagent()
        self._initialize_model(initializer_x)

        print('Loading QM-GNN-No-Reagent selectivity model')
        self.model.summary()
        self.model.load_weights(model_path)


# for testing purposes
if __name__ == "__main__":
    predictor = QmGnnGeneralSelectivityPredictor()
    rawrxn = 'CC(COc1n[nH]cc1)C.CC(C)(OC(c1c(Cl)nc(Cl)cc1)=O)C>CN(C=O)C.O>CC(OC(c1ccc(n2ccc(OCC(C)C)n2)nc1Cl)=O)(C)C'
    res = predictor.predict(rawrxn)
    print(res)# (0.9809687733650208, 0.019030507653951645)
