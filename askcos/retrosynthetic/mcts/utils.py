"""
Utilities for processing tree builder results.
"""

import uuid

import networkx as nx
import numpy as np

NODE_LINK_ATTRS = {'source': 'from', 'target': 'to', 'name': 'id', 'key': 'key', 'link': 'edges'}


def generate_unique_node():
    """
    Generate a unique node label using the UUID specification.

    Use UUIDv4 to generate random UUIDs instead of UUIDv1 which is used by
    ``networkx.utils.generate_unique_node``.
    """
    return str(uuid.uuid4())


def nx_paths_to_json(paths, root_uuid, json_format='treedata'):
    """
    Convert list of paths from networkx graphs to json.
    """
    if json_format == 'treedata':
        return [clean_json(nx.tree_data(path, root_uuid)) for path in paths]
    elif json_format == 'nodelink':
        return [clean_json(nx.node_link_data(path, attrs=NODE_LINK_ATTRS)) for path in paths]
    else:
        raise ValueError('Unsupported value for json_format: {0}'.format(json_format))


def clean_json(path):
    """
    Clean up json representation of a pathway. Accepts paths from either
    tree builder version.

    Note about chemical/reaction node identification:
        * For treedata format, chemical nodes have an ``is_chemical`` attribute,
          while reaction nodes have an ``is_reaction`` attribute
        * For nodelink format, all nodes have a ``type`` attribute, whose value
          is either ``chemical`` or ``reaction``
    """
    # Only fields in this dictionary are kept
    key_map = {
        'smiles': 'smiles',
        'id': 'id',
        'as_reactant': 'as_reactant',
        'as_product': 'as_product',
        'plausibility': 'plausibility',
        'purchase_price': 'ppg',
        'template_score': 'template_score',
        'terminal': 'terminal',
        'tforms': 'tforms',
        'num_examples': 'num_examples',
        'necessary_reagent': 'necessary_reagent',
        'precursor_smiles': 'precursor_smiles',
        'rms_molwt': 'rms_molwt',
        'num_rings': 'num_rings',
        'scscore': 'scscore',
        'rank': 'rank',
    }

    if 'nodes' in path:
        # Node link format
        nodes = []
        for node in path['nodes']:
            new_node = {key_map[key]: value for key, value in node.items() if key in key_map}
            new_node['type'] = node['type']
            nodes.append(new_node)
        path['nodes'] = nodes
        output = path
    else:
        # Tree data format
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
                output['children'] = [clean_json(c) for c in value]

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


def nx_graph_to_paths(tree, root, max_depth=None, max_trees=None,
                      sorting_metric='plausibility', validate_paths=True):
    """
    Return list of paths to buyables starting from the target node.

    Args:
        path_format (str): pathway output format, supports 'graph' or 'json'
        sorting_metric (str): how pathways are sorted, supports 'plausibility',
            'number_of_starting_materials', 'number_of_reactions'
        validate_paths (bool): require all leaves to meet terminal criteria
        legacy_json (bool): convert to json format used by old tree builder

    Returns:
        list of paths in specified format
    """
    # Use NIL UUID for root so we can easily identify it
    root_uuid = '00000000-0000-0000-0000-000000000000'

    paths = get_paths(
        tree,
        root=root,
        root_uuid=root_uuid,
        max_depth=max_depth,
        max_trees=max_trees,
        validate_paths=validate_paths,
    )  # returns generator

    paths = sort_paths(paths, sorting_metric)  # converts to a list

    return paths, root_uuid


def get_paths(tree, root, root_uuid, max_depth=None, max_trees=None, validate_paths=True):
    """
    Generate all paths from the root node as `nx.DiGraph` objects.

    All node attributes are copied to the output paths.

    Args:
        validate_paths (bool): require all leaves to meet terminal criteria

    Returns:
        generator of paths
    """
    import itertools

    def get_chem_paths(_node, _uuid, chem_path):
        """
        Return generator of paths with current node as the root.
        """
        if tree.out_degree(_node) == 0 or max_depth is not None and len(chem_path) >= max_depth:
            sub_path = nx.DiGraph()
            sub_path.add_node(_uuid, smiles=_node, **tree.nodes[_node])
            yield sub_path
        else:
            for rxn in tree.successors(_node):
                rxn_uuid = generate_unique_node()
                for sub_path in get_rxn_paths(rxn, rxn_uuid, chem_path + [_node]):
                    sub_path.add_node(_uuid, smiles=_node, **tree.nodes[_node])
                    sub_path.add_edge(_uuid, rxn_uuid)
                    yield sub_path

    def get_rxn_paths(_node, _uuid, chem_path):
        """
        Return generator of paths with current node as root.
        """
        precursors = list(tree.successors(_node))
        if set(precursors) & set(chem_path):
            # Adding this reaction would create a cycle
            return
        c_uuid = {c: generate_unique_node() for c in precursors}
        for path_combo in itertools.product(*(get_chem_paths(c, c_uuid[c], chem_path) for c in precursors)):
            sub_path = nx.union_all(path_combo)
            sub_path.add_node(_uuid, smiles=_node, **tree.nodes[_node])
            for c in tree.successors(_node):
                sub_path.add_edge(_uuid, c_uuid[c])
            yield sub_path

    def validate_path(_path):
        """Return true if all leaves are terminal."""
        leaves = (v for v, d in _path.out_degree() if d == 0)
        return all(_path.nodes[v]['terminal'] for v in leaves)

    num_paths = 0
    for path in get_chem_paths(root, root_uuid, []):
        if max_trees is not None and num_paths >= max_trees:
            break
        if validate_paths and validate_path(path) or not validate_paths:
            num_paths += 1
            yield path


def sort_paths(paths, metric):
    """
    Sort paths by some metric.
    """

    def number_of_starting_materials(tree):
        return len([v for v, d in tree.out_degree() if d == 0])

    def number_of_reactions(tree):
        return len([v for v in nx.dag_longest_path(tree) if tree.nodes[v]['type'] == 'reaction'])

    def overall_plausibility(tree):
        return np.prod([d['plausibility'] for v, d in tree.nodes(data=True) if d['type'] == 'reaction'])

    if metric == 'plausibility':
        paths = sorted(paths, key=lambda x: overall_plausibility(x), reverse=True)
    elif metric == 'number_of_starting_materials':
        paths = sorted(paths, key=lambda x: number_of_starting_materials(x))
    elif metric == 'number_of_reactions':
        paths = sorted(paths, key=lambda x: number_of_reactions(x))
    else:
        raise ValueError('Need something to sort by! Invalid option provided: {}'.format(metric))

    return paths
