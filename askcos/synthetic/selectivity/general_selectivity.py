import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
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
from askcos.synthetic.selectivity.general_model.qm_models import QMWLNPairwiseAtomClassifier
from askcos.synthetic.atom_mapper.wln_mapper import WLN_AtomMapper
from askcos.synthetic.descriptors.descriptors import ReactivityDescriptor


GNN_model_path = gc.GEN_SELECTIVITY['model_path']['GNN']
QM_GNN_model_path = gc.GEN_SELECTIVITY['model_path']['QM_GNN']
scaler_path = gc.GEN_SELECTIVITY['scalers']
initializer = gc.GEN_SELECTIVITY['initializer']
initializer_qm_descriptors = gc.GEN_SELECTIVITY['initializer_qm']

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

    def __init__(self, depth=4, hidden_size=200,
                 atom_mapper=WLN_AtomMapper().evaluate, descriptor_predictor=ReactivityDescriptor().evaluate):
        
        self.depth = depth
        self.hidden_size = hidden_size

        self.qm_scaler = pd.read_pickle(scaler_path)
        self.GNN_model = None
        self.QM_GNN_model = None
        self.model = None

        self.atom_mapper = atom_mapper
        self.descriptor_predictor = descriptor_predictor

        #self.build()
        
    def build(self):

        initializer_x, _ = self.DataLoader([initializer.split('>')[0]], [initializer.split('>')[2]])[0]

        # build GNN model
        self.GNN_model = WLNReactionClassifier(self.hidden_size)
        opt = tf.keras.optimizers.Adam(lr=0.001, clipnorm=5.0)
        self.GNN_model.compile(
            optimizer=opt,
            loss=wln_loss,
        )
        self.GNN_model.predict_on_batch(initializer_x)
        self.GNN_model.load_weights(GNN_model_path)

        # build QM GNN model
        self.QM_GNN_model = QMWLNPairwiseAtomClassifier(self.hidden_size)
        self.QM_GNN_model.compile(optimizer=opt, loss=wln_loss)
        self.QM_GNN_model.load_weights(QM_GNN_model_path)


    def _initialize_model(self, initializer_x):
        opt = tf.keras.optimizers.Adam(lr=0.001, clipnorm=5.0)
        self.model.compile(
            optimizer=opt,
            loss=wln_loss,
        )
        self.model.predict_on_batch(initializer_x)


    def predict_gnn(self, smiles):
        """
        predict the selectivity from atommapped reactions as reactants>>product1.product2.product3....
        :param smiles:
        :param template:
        :return:
        """

        reactants, _, products = smiles.split('>')

        # find atom map number

        #FIXME the dynamic model seems like to be incompatible with the celery, that the model created in build
        # actaully cannot be reused in the predict, currently an akward method is used to create and load model each
        # time the predict function is called. The grapn network need to be modified into static graph later.
        self.model = WLNReactionClassifier(self.hidden_size)
        initializer_x = gnn_data_generation(initializer.split('>')[0], initializer.split('>')[2])
        self._initialize_model(initializer_x)
        self.model.load_weights(GNN_model_path)
        test_gen = gnn_data_generation(reactants, products)

        out = self.model.predict_on_batch(test_gen).reshape([-1])
        out = tuple([float(x) for x in softmax(out)])

        return out

    def predict_qm_gnn(self, smiles, qm_df):
        """

        :param smiles:
        :return:
        """

        reactants, reagent, products = smiles.split('>')

        self.model = QMWLNPairwiseAtomClassifier(self.hidden_size)
        initializer_x = qm_gnn_data_generation(initializer.split('>')[0], initializer.split('>')[2],
                                               initializer.split('>')[1], initializer_qm_descriptors)
        self._initialize_model(initializer_x)
        self.model.summary()
        self.model.load_weights(QM_GNN_model_path)

        qm_df = _min_max_normalize(qm_df, self.qm_scaler)
        test_gen = qm_gnn_data_generation(reactants, products, reagent, qm_df)

        out = self.model.predict_on_batch(test_gen).reshape([-1])
        out = tuple([float(x) for x in softmax(out)])

        return out

    def predict(self, rxnsmiles, mapped=False, mode='qm-gnn', outcomes_included=False):

        if not mapped:
            rsmi, rgsmi, psmi = rxnsmiles.split('>')
            [rsmi_am, psmi_am] = self.atom_mapper(rsmi + '>>' + psmi).split('>>')

            if rsmi_am and psmi_am:
                rxnsmiles = '>'.join([rsmi_am, rgsmi, psmi_am])
            else:
                raise RuntimeError('Cannot find map atom number for the given unmapped reaction.')

        if not outcomes_included:
            rsmi, _, psmi = rxnsmiles.split('>')
            reaction = {'reactants': rsmi, 'products': psmi, '_id': 0}
            try:
                template = extract_from_reaction(reaction)
                template = template['reactants'] + '>>' + template['products']
                rxnsmiles = apply_template(template, rxnsmiles)
            except Exception as e:
                raise RuntimeError('Failed to find outcomes for the given reaction: {}'.format(e))

        if mode == 'qm-gnn':
            rsmis = rxnsmiles.split('>')[0].split('.')
            descriptors = self.descriptor_predictor(rsmis)
            selectivity = self.predict_qm_gnn(rxnsmiles, descriptors)
        elif mode == 'gnn':
            selectivity = self.predict_gnn(rxnsmiles)
        else:
            raise ValueError("Selectivity mode is invalid")

        return selectivity




# for testing purposes
if __name__ == "__main__":
    predictor = GeneralSelectivityPredictor()
    rawrxn = 'CC(COc1n[nH]cc1)C.CC(C)(OC(c1c(Cl)nc(Cl)cc1)=O)C>ClCCl.CN(C=O)C.O.CCOC(C)=O.[NaH]>CC(OC(c1ccc(n2ccc(OCC(C)C)n2)nc1Cl)=O)(C)C'
    res = predictor.predict(rawrxn)
    print(res)
