#import tensorflow as tf
from tensorflow.keras.utils import Sequence
import numpy as np
from random import shuffle

from makeit.synthetic.selectivity.mol_graph import smiles2graph_pr, pack1D, pack2D, pack2D_withidx, \
    get_mask, binary_features_batch
from makeit import global_config as gc

atom_fdim = gc.GEN_SELECTIVITY['atom_fdim']

def _data_generation(smiles_tmp, products_tmp):

    smiles_extend = []
    prs_extend = []
    labels_extend = []
    for r,ps in zip(smiles_tmp, products_tmp):
        size = len(ps.split('.'))
        prs_extend.extend([smiles2graph_pr(p, r, idxfunc=lambda x: x.GetIdx(), core_buffer=2, atom_fdim=atom_fdim)
                           for p in ps.split('.')])
        smiles_extend.extend([r]*size)

        labels_extend.extend([1]+[0]*(size-1))

    res_extend, prods_extend = zip(*prs_extend)
    # graph_inputs for reactants
    fatom_list, fbond_list, gatom_list, gbond_list, nb_list, core_mask = zip(*res_extend)
    res_graph_inputs = (pack2D(fatom_list), pack2D(fbond_list), pack2D_withidx(gatom_list),
                        pack2D_withidx(gbond_list), pack1D(nb_list), get_mask(fatom_list),
                        pack1D(core_mask), binary_features_batch(smiles_extend))

    # graph_inputs for products
    fatom_list, fbond_list, gatom_list, gbond_list, nb_list, core_mask = zip(*prods_extend)
    prods_graph_inputs = (pack2D(fatom_list), pack2D(fbond_list), pack2D_withidx(gatom_list),
                          pack2D_withidx(gbond_list), pack1D(nb_list), get_mask(fatom_list),
                          pack1D(core_mask))

    return (res_graph_inputs + prods_graph_inputs), np.array(labels_extend).astype('int32')


class GraphDataLoader(Sequence):
    def __init__(self, smiles, products, batch_size, shuffle=True):
        self.smiles = smiles
        self.products = products
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.smiles)/self.batch_size))

    def __getitem__(self, index):
        smiles_tmp = self.smiles[index*self.batch_size:(index+1)*self.batch_size]
        products_tmp = self.products[index*self.batch_size:(index+1)*self.batch_size]
        x, y = _data_generation(smiles_tmp, products_tmp)
        return x, y

    def on_epoch_end(self):
        if self.shuffle:
            zipped = list(zip(self.smiles, self.products))
            shuffle(zipped)
            self.smiles, self.products = zip(*zipped)
