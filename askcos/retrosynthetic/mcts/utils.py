"""
Utilities for processing tree builder results.
"""

import networkx as nx


def nx_to_legacy_json(path):
    """
    Convert json output from networkx to match output of old tree builder.

    Input should be a deserialized python object, not a raw json string.
    """
    key_map = {
        'smiles': 'smiles',
        'id': 'id',
        'as_reactant': 'as_reactant',
        'as_product': 'as_product',
        'ff_score': 'plausibility',
        'purchase_price': 'ppg',
        'template_score': 'template_score',
        'tforms': 'tforms',
        'num_examples': 'num_examples',
        'necessary_reagent': 'necessary_reagent',
    }

    output = {}
    for key, value in path.items():
        if key in key_map:
            output[key_map[key]] = value
        elif key == 'type':
            if value == 'chemical':
                output['is_chemical'] = True
            elif value == 'reaction':
                output['is_reaction'] = True
        elif key == 'children':
            output['children'] = [nx_to_legacy_json(c) for c in value]

    if 'children' not in output:
        output['children'] = []

    return output


def chem_to_nx_graph(chemicals):
    """
    Convert list of Chemical nodes to a networkx graph.
    """

    def _add_chem_node(_chem):
        attributes = vars(_chem).copy()
        attributes['type'] = 'chemical'
        del attributes['smiles']
        del attributes['prob']
        del attributes['template_idx_results']
        graph.add_node(_chem.smiles, **attributes)

    def _add_rxn_node(_rxn):
        attributes = vars(_rxn).copy()
        attributes['type'] = 'reaction'
        del attributes['smiles']
        del attributes['reactant_smiles']
        rxn_smiles = '.'.join(_rxn.reactant_smiles) + '>>' + _rxn.smiles
        graph.add_node(rxn_smiles, **attributes)
        return rxn_smiles

    graph = nx.DiGraph()
    # Create all chemical nodes first
    for chem in chemicals:
        _add_chem_node(chem)

    # Now go back and add reactions
    for chem in chemicals:
        for cta in chem.template_idx_results.values():
            for rxn in cta.reactions.values():
                rxn_smiles = _add_rxn_node(rxn)
                graph.add_edge(chem.smiles, rxn_smiles)
                for precursor in rxn.reactant_smiles:
                    graph.add_edge(rxn_smiles, precursor)

    return graph
