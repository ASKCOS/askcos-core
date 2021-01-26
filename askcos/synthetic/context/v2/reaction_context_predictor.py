#!/usr/bin/env python3

import pickle
import json
import copy

import numpy as np

import tensorflow as tf

from . import db
from . import search
from . import graph_util
from . import results_preprocess
from . import smiles_util

def get_input_dims(atom, bond, is_batch=False):
    '''
        atom: (batch_size, atom_dim, atom_nfeature)
        bond: (batch_size, atom_dim, atom_dim, bond_nfeature)
        or
        atom: (atom_dim, atom_nfeature)
        bond: (atom_dim, atom_dim, bond_nfeature)
    '''
    if is_batch:
        atom_dim = atom.shape[1]
        atom_nfeature = atom.shape[-1]
        bond_nfeature = bond.shape[-1]
    else:
        atom_dim = atom.shape[0]
        atom_nfeature = atom.shape[-1]
        bond_nfeature = bond.shape[-1]
    return atom_dim, atom_nfeature, bond_nfeature


def load_model(model_dir):
    imported = tf.saved_model.load(export_dir=model_dir, tags="serve")
    imported_function = imported.signatures["serving_default"]
    return imported, imported_function


class ReactionContextRecommenderWLN:
    '''
        This predictor requires atom mapped reactions.
        Require number of atoms in all reactants <= 50.
        max_num_reagents   <= 4
        max_num_reactants  <= 5
    '''
    def __init__(
        self,
        encoder=db.config_reaction_condition['models']['graph-20191118']['encoder'],
        reagents=db.config_reaction_condition['models']['graph-20191118']['reagents'],
        reagents_model=db.config_reaction_condition['models']['graph-20191118']['reagents_model'],
        temperature_model=db.config_reaction_condition['models']['graph-20191118']['temperature_model'],
        reagents_amount_model=db.config_reaction_condition['models']['graph-20191118']['reagents_amount_model'],
        reactants_amount_model=db.config_reaction_condition['models']['graph-20191118']['reactants_amount_model'],
        condensed_graph=db.config_reaction_condition['models']['graph-20191118']['condensed_graph'],
    ):
        '''
            condensed_graph: True if atom-mapping is required
        '''
        self.condensed_graph    = condensed_graph
        self.max_num_reagents   = 4
        self.max_num_reactants  = 5
        
        # graph encoder
        self.feature_encoder_fn = encoder
        with open(self.feature_encoder_fn, 'rb') as f:
            self.feature_encoder = pickle.load(f)
        
        # reagent encoder/decoder
        self.reagent_encoder_fn = reagents
        with open(self.reagent_encoder_fn, 'r') as f:
            self.reagent_encoder = results_preprocess.generate_reagents_encoder2(json.load(f))
        self.reagent_decoder = {v:k for k,v in self.reagent_encoder.items()}
        
        # reagent predictor
        self.reagent_model_fn = reagents_model
        self.reagent_model_tf, self.reagent_model = load_model(self.reagent_model_fn)
        
        # temperature predictor
        self.temperature_model_fn = temperature_model
        self.temperature_model_tf, self.temperature_model = load_model(self.temperature_model_fn)
        
        # reagent quantity predictor
        self.reagent_quantity_model_fn = reagents_amount_model
        self.reagent_quantity_model_tf, self.reagent_quantity_model = load_model(self.reagent_quantity_model_fn)
        
        # reactant quantity predictor
        self.reactant_quantity_model_fn = reactants_amount_model
        self.reactant_quantity_model_tf, self.reactant_quantity_model = load_model(self.reactant_quantity_model_fn)
    
    def add_batch_dimension(self, x):
        '''
            insert dim=1
        '''
        if isinstance(x, list):
            res = []
            for i in x:
                res.append(tf.convert_to_tensor(np.expand_dims(i, axis=0), dtype=tf.float32))
            return res
        else:
            return tf.convert_to_tensor(np.expand_dims(x, axis=0), dtype=tf.float32)
    
    def encode_condensed_graph(self, smiles):
        # feature
        f = graph_util.rxn2features(smiles)
        atom, bond, conn = graph_util.encode_features_atommapped_dense_graph(f, self.feature_encoder, isrand=False)
        atom, bond, conn = self.add_batch_dimension([atom, bond, conn])
        return {
            'Input_atom':tf.convert_to_tensor(atom, dtype=tf.float32),
            'Input_bond':tf.convert_to_tensor(bond, dtype=tf.float32),
            'Input_conn':tf.convert_to_tensor(conn, dtype=tf.float32),
        }
    
    def encode_reagents(self, reagents):
        '''
            reagents: list of strings, each string is a reagent SMILES
            
            Return:
                Sum of one-hot encoding, 0 or 1
        '''
        # build preprocessed structure
        r = []
        for i in reagents:
            a = {}
            a['smiles'] = i
            r.append(a)
        reagents_onehot = results_preprocess.prepare_reagents2(self.reagent_encoder, r)
        reagents_multiclass = results_preprocess.convert_onehots_to_multiclass(reagents_onehot)
        return self.add_batch_dimension(reagents_multiclass)
    
    def decode_reagents(self, encoded_reagents):
        return [self.reagent_decoder[i] for i in np.where(np.abs(encoded_reagents-1.0) < 1e-6)[0]]
    
    def predict_reagents(self, smiles=None, beam_size=10, is_decode=True, encoded_graph=None, reagents=None):
        '''
            Return:
                if is_decode: res = [(reagents_onehot, score)]
                else: res = [([reagents_smiles], score)]
        '''
        if encoded_graph is None:
            encoded_graph = self.encode_condensed_graph(smiles)
        if reagents is not None:
            encoded_reagents = self.encode_reagents(reagents)
        else:
            encoded_reagents = None
        res = search.beam_search(self.reagent_model, copy.copy(encoded_graph), len(self.reagent_encoder), max_steps=self.max_num_reagents+1, beam_size=beam_size, eos_id=0, keepall=False, reagents=encoded_reagents) # keepall=False is beam search
        res_top = search.top_n_results(res, n=beam_size)
        if is_decode:
            res_top_decode = []
            for r in res_top:
                res_top_decode.append((self.decode_reagents(r[0]), float(r[1])))
            return res_top_decode
        else:
            return res_top
    
    def predict_temperature(self, smiles=None, reagents=None, encoded_graph=None, encoded_reagents=None):
        if encoded_graph is None:
            encoded_graph = self.encode_condensed_graph(smiles)
        if encoded_reagents is None:
            encoded_reagents = self.encode_reagents(reagents)
        data_input = copy.copy(encoded_graph)
        data_input['Input_reagent'] = encoded_reagents
        y_pred = self.temperature_model(**data_input)['output_regression'].numpy()
        return float(y_pred[0][0]*273.15)
    
    def predict_reagent_quantities(self, smiles=None, reagents=None, encoded_graph=None, encoded_reagents=None):
        if encoded_graph is None:
            encoded_graph = self.encode_condensed_graph(smiles)
        if encoded_reagents is None:
            encoded_reagents = self.encode_reagents(reagents)
        data_input = copy.copy(encoded_graph)
        data_input['Input_reagent'] = encoded_reagents
        y_pred = self.reagent_quantity_model(**data_input)['multiply_1'].numpy()
        
        amount = {}
        for i in np.where(encoded_reagents[0,:]>1e-6)[0]:
            amount[self.reagent_decoder[i]] = float(np.exp(y_pred[0, i]))
        return amount
    
    def predict_reactant_quantities(self, smiles=None, reagents=None, encoded_reagents=None):
        '''
            Reactants in smiles are splitted by '.'
            Return:
                {'reactants':mole}
            
            Expect:
                May throw expections due to encoding failures.
        '''
        # all reactants and products
        f = graph_util.rxn2features(smiles)
        atom1, bond1, conn1, atom2, bond2, conn2 = graph_util.encode_features_non_mapped(f, self.feature_encoder, isrand=False)
        atom1, bond1, conn1, atom2, bond2, conn2 = self.add_batch_dimension([atom1, bond1, conn1, atom2, bond2, conn2])
        
        # reagents
        if encoded_reagents is None:
            encoded_reagents = self.encode_reagents(reagents)
        
        data_input = {
            'Input_atom_reactants': atom1,
            'Input_bond_reactants': bond1,
            'Input_conn_reactants': conn1,
            'Input_atom_products': atom2,
            'Input_bond_products': bond2,
            'Input_conn_products': conn2,
        }
        data_input['Input_reagent'] = encoded_reagents
        
        # individual reactants
        reactants = smiles.split('>')[0].split('.')
        if len(reactants) > self.max_num_reactants:
            raise ValueError('Number of reactants ({}) is greater than the allowed maximum {}.'.format(len(reactants), self.max_num_reactants))
        atom_dim, atom_nfeature, bond_nfeature = get_input_dims(atom1, bond1, is_batch=True)
        mask = np.zeros(shape=(1, self.max_num_reactants), dtype=np.float32)
        for i, r in enumerate(reactants):
            f = graph_util.smiles2features(r)
            a, b, c = graph_util.encode_features_onemol(f, self.feature_encoder, isrand=False)
            data_input['Input_atom_'+str(i)] = self.add_batch_dimension(a)
            data_input['Input_bond_'+str(i)] = self.add_batch_dimension(b)
            data_input['Input_conn_'+str(i)] = self.add_batch_dimension(c)
            mask[0, i] = 1
        
        # append empty vector
        for i in range(len(reactants), self.max_num_reactants):
            data_input['Input_atom_'+str(i)] = tf.zeros(shape=(1, atom_dim, atom_nfeature), dtype=tf.float32)
            data_input['Input_bond_'+str(i)] = tf.zeros(shape=(1, atom_dim, atom_dim, bond_nfeature), dtype=tf.float32)
            data_input['Input_conn_'+str(i)] = tf.zeros(shape=(1, atom_dim, atom_dim), dtype=tf.float32)
        
        data_input['Input_reactant_mask'] = tf.convert_to_tensor(mask, dtype=tf.float32)
        y_pred = self.reactant_quantity_model(**data_input)['multiply_1'].numpy()
        
        amount = {}
        for i, r in enumerate(reactants):
            amount[r] = float(np.exp(y_pred[0, i]))
        return amount
    
    def predict(self, smiles, beam_size=10, reagents=None):
        '''
            smiles: SMILES string of the reaction including reactant and product
            beam_size: beam size of beam search for predicting reagents
            
            Return:
                (beam_size) sets of conditions, [{'reagents':{'reagent1_smiles':mole1, 'reagent2_smiles':mole2}, 'reactants':{'reactant1_smiles':mole, 'reactant2_smiles':mole}, 'temperature':T, 'reagents_score':score}, ...]
            
            Except:
                May throw exceptions due to encoding failures
            
            Note:
                reagents_score is the product of individual reagent score, so it is very small.
        '''
        res = []
        encoded_graph = self.encode_condensed_graph(smiles)
        reagents = self.predict_reagents(smiles=None, beam_size=beam_size, is_decode=False, encoded_graph=encoded_graph, reagents=reagents)
        for encoded_reagents, score in reagents:
            res_one = {'reagents_score':float(score)}
            # reagents_smiles = self.decode_reagents(encoded_reagents)
            encoded_reagents = self.add_batch_dimension(encoded_reagents)
            res_one['temperature'] = self.predict_temperature(smiles=None, reagents=None, encoded_graph=encoded_graph, encoded_reagents=encoded_reagents) # Kelvin
            res_one['reagents'] = self.predict_reagent_quantities(smiles=None, reagents=None, encoded_graph=encoded_graph, encoded_reagents=encoded_reagents)
            res_one['reactants'] = self.predict_reactant_quantities(smiles=smiles, reagents=None, encoded_reagents=encoded_reagents)
            # append results
            res.append(res_one)
        return res


class ReactionContextRecommenderFP:
    '''
        This predictor requires atom mapped reactions.
        Require number of atoms in all reactants <= 50.
        max_num_reagents   <= 4
        max_num_reactants  <= 5
    '''
    def __init__(
        self,
        reagents=db.config_reaction_condition['models']['fp-20191118']['reagents'],
        reagents_model=db.config_reaction_condition['models']['fp-20191118']['reagents_model'],
        temperature_model=db.config_reaction_condition['models']['fp-20191118']['temperature_model'],
        reagents_amount_model=db.config_reaction_condition['models']['fp-20191118']['reagents_amount_model'],
        reactants_amount_model=db.config_reaction_condition['models']['fp-20191118']['reactants_amount_model'],
        fp_rad = db.config_reaction_condition['models']['fp-20191118']['fp_rad'],
        fp_len = db.config_reaction_condition['models']['fp-20191118']['fp_len'],
    ):
        self.fp_radius = fp_rad
        self.fp_length = fp_len
        self.max_num_reagents   = 4
        self.max_num_reactants  = 5
        
        # reagent encoder/decoder
        self.reagent_encoder_fn = reagents
        with open(self.reagent_encoder_fn, 'r') as f:
            self.reagent_encoder = results_preprocess.generate_reagents_encoder2(json.load(f))
        self.reagent_decoder = {v:k for k,v in self.reagent_encoder.items()}
        
        # reagent predictor
        self.reagent_model_fn = reagents_model
        self.reagent_model_tf, self.reagent_model = load_model(self.reagent_model_fn)
        
        # temperature predictor
        self.temperature_model_fn = temperature_model
        self.temperature_model_tf, self.temperature_model = load_model(self.temperature_model_fn)
        
        # reagent quantity predictor
        self.reagent_quantity_model_fn = reagents_amount_model
        self.reagent_quantity_model_tf, self.reagent_quantity_model = load_model(self.reagent_quantity_model_fn)
        
        # reactant quantity predictor
        self.reactant_quantity_model_fn = reactants_amount_model
        self.reactant_quantity_model_tf, self.reactant_quantity_model = load_model(self.reactant_quantity_model_fn)
    
    def add_batch_dimension(self, x):
        '''
            insert dim=1
        '''
        if isinstance(x, list):
            res = []
            for i in x:
                res.append(tf.convert_to_tensor(np.expand_dims(i, axis=0), dtype=tf.float32))
            return res
        else:
            return tf.convert_to_tensor(np.expand_dims(x, axis=0), dtype=tf.float32)
    
    def encode_condensed_graph(self, smiles, fp_length=None, fp_radius=None):
        if fp_length is None:
            fp_length = self.fp_length
        if fp_radius is None:
            fp_radius = self.fp_radius
        # feature
        smiles_splitted = smiles.split(' ')[0].split('>')
        r_fp = smiles_util.get_morgan_fp(smiles_splitted[0], fp_radius, fp_length)
        p_fp = smiles_util.get_morgan_fp(smiles_splitted[2], fp_radius, fp_length)
        input_fp = np.concatenate([r_fp, p_fp], axis=-1)
        return {'Input_fp': self.add_batch_dimension(input_fp)}
    
    def encode_reagents(self, reagents):
        '''
            reagents: list of strings, each string is a reagent SMILES
            
            Return:
                Sum of one-hot encoding, 0 or 1
        '''
        # build preprocessed structure
        r = []
        for i in reagents:
            a = {}
            a['smiles'] = i
            r.append(a)
        reagents_onehot = results_preprocess.prepare_reagents2(self.reagent_encoder, r)
        reagents_multiclass = results_preprocess.convert_onehots_to_multiclass(reagents_onehot)
        return self.add_batch_dimension(reagents_multiclass)
    
    def decode_reagents(self, encoded_reagents):
        return [self.reagent_decoder[i] for i in np.where(np.abs(encoded_reagents-1.0) < 1e-6)[0]]
    
    def predict_reagents(self, smiles=None, beam_size=10, is_decode=True, encoded_graph=None, reagents=None):
        '''
            Return:
                if is_decode: res = [(reagents_onehot, score)]
                else: res = [([reagents_smiles], score)]
        '''
        if encoded_graph is None:
            encoded_graph = self.encode_condensed_graph(smiles)
        if reagents is not None:
            encoded_reagents = self.encode_reagents(reagents)
        else:
            encoded_reagents = None
        res = search.beam_search(self.reagent_model, copy.copy(encoded_graph), len(self.reagent_encoder), max_steps=self.max_num_reagents+1, beam_size=beam_size, eos_id=0, keepall=False, reagents=encoded_reagents) # keepall=False is beam search
        res_top = search.top_n_results(res, n=beam_size)
        if is_decode:
            res_top_decode = []
            for r in res_top:
                res_top_decode.append((self.decode_reagents(r[0]), float(r[1])))
            return res_top_decode
        else:
            return res_top
    
    def predict_temperature(self, smiles=None, reagents=None, encoded_graph=None, encoded_reagents=None):
        if encoded_graph is None:
            encoded_graph = self.encode_condensed_graph(smiles, 16384)
        if encoded_reagents is None:
            encoded_reagents = self.encode_reagents(reagents)
        data_input = copy.copy(encoded_graph)
        data_input['Input_reagent'] = encoded_reagents
        y_pred = self.temperature_model(**data_input)['output_regression'].numpy()
        return float(y_pred[0][0]*273.15)
    
    def predict_reagent_quantities(self, smiles=None, reagents=None, encoded_graph=None, encoded_reagents=None):
        if encoded_graph is None:
            encoded_graph = self.encode_condensed_graph(smiles)
        if encoded_reagents is None:
            encoded_reagents = self.encode_reagents(reagents)
        data_input = copy.copy(encoded_graph)
        data_input['Input_reagent'] = encoded_reagents
        y_pred = self.reagent_quantity_model(**data_input)['multiply_1'].numpy()
        
        amount = {}
        for i in np.where(encoded_reagents[0,:]>1e-6)[0]:
            amount[self.reagent_decoder[i]] = float(np.exp(y_pred[0, i]))
        return amount
    
    def predict_reactant_quantities(self, smiles=None, reagents=None, encoded_reagents=None):
        '''
            Reactants in smiles are splitted by '.'
            Return:
                {'reactants':mole}
            
            Expect:
                May throw expections due to encoding failures.
        '''
        # all reactants and products
        encoded_graph = self.encode_condensed_graph(smiles)
        
        # reagents
        if encoded_reagents is None:
            encoded_reagents = self.encode_reagents(reagents)
        
        data_input = copy.copy(encoded_graph)
        data_input['Input_reagent'] = encoded_reagents
        
        # individual reactants
        reactants = smiles.split('>')[0].split('.')
        if len(reactants) > self.max_num_reactants:
            raise ValueError('Number of reactants ({}) is greater than the allowed maximum {}.'.format(len(reactants), self.max_num_reactants))
        mask = np.zeros(shape=(1, self.max_num_reactants), dtype=np.float32)
        for i, r in enumerate(reactants):
            r_fp = smiles_util.get_morgan_fp(r, self.fp_radius, self.fp_length)
            data_input['Input_fp_reactant_'+str(i)] = self.add_batch_dimension(r_fp)
            mask[0, i] = 1
        
        # append empty vector
        for i in range(len(reactants), self.max_num_reactants):
            data_input['Input_fp_reactant_'+str(i)] = tf.zeros(shape=(1, self.fp_length), dtype=tf.float32)
        
        data_input['Input_reactant_mask'] = tf.convert_to_tensor(mask, dtype=tf.float32)
        y_pred = self.reactant_quantity_model(**data_input)['multiply_1'].numpy()
        
        amount = {}
        for i, r in enumerate(reactants):
            amount[r] = float(np.exp(y_pred[0, i]))
        return amount
    
    def predict(self, smiles, beam_size=10, reagents=None):
        '''
            smiles: SMILES string of the reaction including reactant and product
            beam_size: beam size of beam search for predicting reagents
            
            Return:
                (beam_size) sets of conditions, [{'reagents':{'reagent1_smiles':mole1, 'reagent2_smiles':mole2}, 'reactants':{'reactant1_smiles':mole, 'reactant2_smiles':mole}, 'temperature':T, 'reagents_score':score}, ...]
            
            Except:
                May throw exceptions due to encoding failures
            
            Note:
                reagents_score is the product of individual reagent score, so it is very small.
        '''
        res = []
        encoded_graph = self.encode_condensed_graph(smiles)
        reagents = self.predict_reagents(smiles=None, beam_size=beam_size, is_decode=False, encoded_graph=encoded_graph, reagents=reagents)
        for encoded_reagents, score in reagents:
            res_one = {'reagents_score':float(score)}
            # reagents_smiles = self.decode_reagents(encoded_reagents)
            encoded_reagents = self.add_batch_dimension(encoded_reagents)
            res_one['temperature'] = self.predict_temperature(smiles=None, reagents=None, encoded_graph=encoded_graph, encoded_reagents=encoded_reagents) # Kelvin
            res_one['reagents'] = self.predict_reagent_quantities(smiles=None, reagents=None, encoded_graph=encoded_graph, encoded_reagents=encoded_reagents)
            res_one['reactants'] = self.predict_reactant_quantities(smiles=smiles, reagents=None, encoded_reagents=encoded_reagents)
            # append results
            res.append(res_one)
        return res

def __test_wln():
    predictor_wln = ReactionContextRecommenderWLN()
    # {'id': 84022, 'db_id': None, 'date': '20190801', 'rxn_smiles': '[N:1]#[C:2][CH:3]([C:4](=O)[c:5]1[cH:6][cH:7][cH:8][cH:9][cH:10]1)[c:11]1[cH:12][cH:13][cH:14][cH:15][cH:16]1.[NH2:17][NH2:18]>CC(=O)O.CCO>[NH2:1][c:2]1[nH:18][n:17][c:4]([c:3]1-[c:11]1[cH:16][cH:15][cH:14][cH:13][cH:12]1)-[c:5]1[cH:6][cH:7][cH:8][cH:9][cH:10]1', 'temperature': None, 'reactants': [{'smiles': 'N#CC(C(=O)c1ccccc1)c1ccccc1', 'conc': None, 'unit': None, 'value': None, 'is_set': True, 'mole': None, 'mass': None, 'volume': None}, {'smiles': 'NN', 'conc': None, 'unit': None, 'value': None, 'is_set': True, 'mole': None, 'mass': None, 'volume': None}], 'products': [{'smiles': 'Nc1[nH]nc(c1c1ccccc1)c1ccccc1', 'yield': None, 'unit': None, 'value': None, 'is_set': True, 'mole': None, 'mass': None, 'volume': None}], 'reagents': [{'smiles': 'CC(=O)O', 'conc': None, 'unit': None, 'value': None, 'is_set': True, 'mole': None, 'mass': None, 'volume': None}, {'smiles': 'CCO', 'conc': None, 'unit': None, 'value': None, 'is_set': True, 'mole': None, 'mass': None, 'volume': None}], 'filepath': 'applications/2019/I20190801_cdx_reactions_wbib.json', 'filelinenum': 1434}
    s = '[N:1]#[C:2][CH:3]([C:4](=O)[c:5]1[cH:6][cH:7][cH:8][cH:9][cH:10]1)[c:11]1[cH:12][cH:13][cH:14][cH:15][cH:16]1.[NH2:17][NH2:18]>>[NH2:1][c:2]1[nH:18][n:17][c:4]([c:3]1-[c:11]1[cH:16][cH:15][cH:14][cH:13][cH:12]1)-[c:5]1[cH:6][cH:7][cH:8][cH:9][cH:10]1'
    reagents_true = ['CC(=O)O','CCO']
    reagents = predictor_wln.predict_reagents(smiles=s)
    print('ReactionContextRecommenderWLN')
    print('reagents: ', json.dumps(reagents,indent=4))
    t = predictor_wln.predict_temperature(smiles=s, reagents=reagents_true)
    print('temperature (K): ', t)
    print()
    
    results = predictor_wln.predict(smiles=s)
    print('ReactionContextRecommenderWLN')
    print('all predictions')
    print(json.dumps(results,indent=4))
    print()
    
    results = predictor_wln.predict(smiles=s, reagents=['CC(=O)O'])
    print('ReactionContextRecommenderWLN')
    print('preset reagents')
    print(json.dumps(results,indent=4))
    print()
    
    print('test')
    s = 'CC(=O)O.Fc1ccccc1Nc1cccc2ccccc12>>Cc1c2cccc(F)c2nc2c1ccc1ccccc12'
    results = predictor_wln.predict(smiles=s)
    print('ReactionContextRecommenderWLN')
    print('all predictions')
    print(json.dumps(results,indent=4))
    s = 'CC(=O)O.Fc1ccccc1Nc1ccccc1>>Cc1c2cccc(F)c2nc2c1cccc2'
    results = predictor_wln.predict(smiles=s)
    print('ReactionContextRecommenderWLN')
    print('smaller rings')
    print(json.dumps(results,indent=4))
    s = 'CC(=O)O.c1ccccc1>>CC(=O)c1ccccc1'
    results = predictor_wln.predict(smiles=s)
    print('ReactionContextRecommenderWLN')
    print('benzene rings')
    print(json.dumps(results,indent=4))
    s = 'CC(=O)Cl.c1ccccc1>>CC(=O)c1ccccc1'
    results = predictor_wln.predict(smiles=s)
    print('ReactionContextRecommenderWLN')
    print('benzene Cl rings')
    print(json.dumps(results,indent=4))

def __test_fp():
    print('ReactionContextRecommenderFP')
    predictor_fp = ReactionContextRecommenderFP()
    s = '[N:1]#[C:2][CH:3]([C:4](=O)[c:5]1[cH:6][cH:7][cH:8][cH:9][cH:10]1)[c:11]1[cH:12][cH:13][cH:14][cH:15][cH:16]1.[NH2:17][NH2:18]>>[NH2:1][c:2]1[nH:18][n:17][c:4]([c:3]1-[c:11]1[cH:16][cH:15][cH:14][cH:13][cH:12]1)-[c:5]1[cH:6][cH:7][cH:8][cH:9][cH:10]1'
    reagents_true = ['CC(=O)O','CCO']
    reagents = predictor_fp.predict_reagents(smiles=s)
    print('reagents: ', json.dumps(reagents,indent=4))
    t = predictor_fp.predict_temperature(smiles=s, reagents=reagents_true)
    print('temperature (K): ', t)
    print()
    
    results = predictor_fp.predict(smiles=s)
    print('ReactionContextRecommenderFP')
    print('all predictions')
    print(json.dumps(results,indent=4))
    print()
    
    results = predictor_fp.predict(smiles=s, reagents=['OCC'])
    print('ReactionContextRecommenderWLN')
    print('preset reagents')
    print(json.dumps(results,indent=4))
    print()

if __name__ == '__main__':
    __test_wln()
    __test_fp()
