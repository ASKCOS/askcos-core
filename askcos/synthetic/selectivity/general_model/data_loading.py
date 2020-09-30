import numpy as np

from askcos.synthetic.selectivity.mol_graph import smiles2graph_pr, pack1D, pack2D, pack2D_withidx, \
    get_mask, binary_features_batch, smiles2graph_pr_qm, pack2D_cores


def gnn_data_generation(smiles, products):

    size = len(products.split('.'))
    prs_extend = [smiles2graph_pr(p, smiles, idxfunc=lambda x: x.GetIdx(), core_buffer=2)
                  for p in products.split('.')]
    smiles_extend = [smiles] * size

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

    return res_graph_inputs + prods_graph_inputs



def qm_gnn_data_generation(smiles, products, reagents, qm_descriptors):

    prs_extend = [smiles2graph_pr_qm(smiles, p, reagents, qm_descriptors) for p in products.split('.')]

    fatom_list, fatom_qm_list, fbond_list, gatom_list, gbond_list, nb_list, cores, connect, \
    rg_fatom_list, rg_fbond_list, rg_gatom_list, rg_gbond_list, rg_nb_list = zip(*prs_extend)

    res_graph_inputs = (pack2D(fatom_list), pack2D(fbond_list), pack2D_withidx(gatom_list),
                        pack2D_withidx(gbond_list), pack1D(nb_list), get_mask(fatom_list),
                        np.stack([pack2D_cores(cores)] * len(prs_extend)),  # trick tensorflow
                        pack2D(fatom_qm_list),
                        np.stack([np.concatenate(connect, axis=0)] * len(prs_extend)),  # trick tensorflow
                        pack2D(rg_fatom_list), pack2D(rg_fbond_list), pack2D_withidx(rg_gatom_list),
                        pack2D_withidx(rg_gbond_list), pack1D(rg_nb_list), get_mask(rg_fatom_list),
                        )

    return res_graph_inputs
