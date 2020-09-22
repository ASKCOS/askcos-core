import numpy as np
from rdkit import Chem
from rdkit.Chem import DataStructs, AllChem


def calculate_evaluation_orders(adjacency_list, tree_size):
    """
    Calculates the node_order and edge_order from a tree adjacency_list and the tree_size.
    The TreeLSTM model requires node_order and edge_order to be passed into the model along
    with the node features and adjacency_list.  We pre-calculate these orders as a speed
    optimization.
    """
    # print(type(adjacency_list))
    adjacency_list = np.array(adjacency_list)
    # print(adjacency_list.shape)
    node_ids = np.arange(tree_size, dtype=int)

    node_order = np.zeros(tree_size, dtype=int)
    unevaluated_nodes = np.ones(tree_size, dtype=bool)

    parent_nodes = adjacency_list[:, 0]
    child_nodes = adjacency_list[:, 1]

    n = 0
    while unevaluated_nodes.any():
        # Find which child nodes have not been evaluated
        unevaluated_mask = unevaluated_nodes[child_nodes]

        # Find the parent nodes of unevaluated children
        unready_parents = parent_nodes[unevaluated_mask]

        # Mark nodes that have not yet been evaluated
        # and which are not in the list of parents with unevaluated child nodes
        nodes_to_evaluate = unevaluated_nodes & ~np.isin(node_ids, unready_parents)

        node_order[nodes_to_evaluate] = n
        unevaluated_nodes[nodes_to_evaluate] = False

        n += 1

    edge_order = node_order[parent_nodes]

    return node_order, edge_order


def label_node_index(node, n=0):
    node['index'] = n
    for child in node['child']:
        n += 1
        n = label_node_index(child, n)
    return n


def gather_node_features(node, key, level=0):
    features = [node[key]]
    # print('--' * level + str(node['index']) + '-' + node['idx'])
    for child in node['child']:
        features.extend(gather_node_features(child, key, level=level + 1))
    return features


def gather_adjacency_list(node):
    adjacency_list = []
    for child in node['child']:
        adjacency_list.append([node['index'], child['index']])
        adjacency_list.extend(gather_adjacency_list(child))
    return adjacency_list


def reaction_to_fp(rsmi, psmi, fpsize=2048):
    rsmi = rsmi.encode('utf-8')
    try:
        mol = Chem.MolFromSmiles(rsmi)
    except Exception as e:
        print("Cannot build reactant mol due to {}".format(e))
        return
    try:
        fp_bit = AllChem.GetMorganFingerprintAsBitVect(mol,
                                                       radius=2,
                                                       nBits=fpsize,
                                                       useFeatures=False,
                                                       useChirality=True)
        fp = np.empty(fpsize, dtype='int8')
        DataStructs.ConvertToNumpyArray(fp_bit, fp)
    except Exception as e:
        print("Cannot build reactant fp due to {}".format(e))
        print(rsmi)
        return

    rfp = fp

    psmi = psmi.encode('utf-8')
    try:
        mol = Chem.MolFromSmiles(psmi)
    except Exception as e:
        print("Cannot build product mol due to {}".format(e))
        return

    try:
        fp_bit = AllChem.GetMorganFingerprintAsBitVect(mol,
                                                       radius=2,
                                                       nBits=fpsize,
                                                       useFeatures=False,
                                                       useChirality=True)
        fp = np.empty(fpsize, dtype='int8')
        DataStructs.ConvertToNumpyArray(fp_bit, fp)

    except Exception as e:
        print("Cannot build product fp due to {}".format(e))
        return

    pfp = fp

    rxnfp = pfp - rfp
    return np.asarray(pfp), np.asarray(rxnfp)


def tree_to_fp(tree, fpsize=2048, n=0):
    # treefp = None
    if tree['child']:
        psmi = tree['smiles']
        rsmi = '.'.join([c['smiles'] for c in tree['child']])

        pfp, rxnfp = reaction_to_fp(rsmi, psmi, fpsize=fpsize)

        treefp = {'pfp': pfp,
                  'rxnfp': rxnfp,
                  'index': n,
                  'child': []}
        tree['index'] = n
        for c in tree['child']:
            if c['child']:
                n += 1
                output, n = tree_to_fp(c, fpsize=fpsize, n=n)
                treefp['child'].append(output)

        return treefp, n


def tree_to_input(tree, fpsize=2048):
    treefp, _ = tree_to_fp(tree, fpsize=fpsize)

    pfp = np.vstack(gather_node_features(treefp, 'pfp'))
    rxnfp = np.vstack(gather_node_features(treefp, 'rxnfp'))

    adjacency_list = gather_adjacency_list(treefp)

    node_order, edge_order = calculate_evaluation_orders(adjacency_list, len(pfp))

    return {
        'pfp': pfp,
        'rxnfp': rxnfp,
        'node_order': node_order,
        'adjacency_list': np.array(adjacency_list),
        'edge_order': edge_order,
        'num_nodes': len(pfp),
        'num_trees': 1
    }


def convert_askcos_trees(askcostrees):
    """
    Convert a list of askcos generate retrosynthetic trees to format needed
    for pathway ranker.

    Args:
        askcostrees (list): list of askcos trees

    Returns:
        list of pathway ranker trees
    """
    output = []
    for askcostree in askcostrees:
        tree = convert_tree(askcostree)
        tree['depth'] = find_depth(tree)
        output.append(tree)
    return output


def convert_tree(askcostree):
    """
    Extract relevant information from an askcos tree.

    Output contains chemical nodes only.
    """
    tree = {
        'smiles': askcostree['smiles'],
        'child': [],
    }

    if askcostree['children']:
        tree['reaction_plausibility'] = askcostree['children'][0]['plausibility']
        tree['reaction_examples'] = askcostree['children'][0]['num_examples']
        tree['template_score'] = askcostree['children'][0]['template_score']
        tree['reaction_reagent'] = askcostree['children'][0]['necessary_reagent']

        if askcostree['children'][0]['children']:
            for child in askcostree['children'][0]['children']:
                tree['child'].append(convert_tree(child))

    return tree


def find_depth(tree, level=0):
    depth = level
    if tree['child']:
        for c in tree['child']:
            new_depth = find_depth(c, level=level+1)
            if depth < new_depth:
                depth = new_depth
    return depth


def merge_into_batch(batch, to_tensor=False, device=None):
    """
    Merge fingerprints of individual trees into a single set of arrays.

    Args:
        batch (list): list of dictionaries containing tree fingerprints
        to_tensor (bool, optional): if True, convert to torch.tensor
        device (torch.device, optional): desired device for tensors

    Returns:
        dict:
            pfp: np.ndarray or torch.tensor
            rxnfp: np.ndarray or torch.tensor
            node_order: np.ndarray or torch.tensor
            adjacency_list: np.ndarray or torch.tensor
            edge_order: np.ndarray or torch.tensor
            num_nodes: list
            num_trees: list
            batch_size: int
    """
    if to_tensor:
        import torch

        def process_array(array, dtype):
            dtype_dict = {'float32': torch.float32, 'int64': torch.int64}
            return torch.tensor(array, device=device, dtype=dtype_dict[dtype])
    else:
        def process_array(array, dtype):
            dtype_dict = {'float32': np.float32, 'int64': np.int64}
            return array.astype(dtype=dtype_dict[dtype])

    pfp = np.vstack([record['pfp'] for record in batch])
    rxnfp = np.vstack([record['rxnfp'] for record in batch])

    node_order = np.hstack([record['node_order'] for record in batch])
    edge_order = np.hstack([record['edge_order'] for record in batch])

    # adjacency_list needs to add offset when concatenate the trees
    adjacency_list = []
    offset = 0
    for record in batch:
        n, a_l = record['pfp'].shape[0], record['adjacency_list']
        adjacency_list.append(a_l + offset)
        offset += n
    adjacency_list = np.vstack(adjacency_list)

    num_nodes = []
    if type(batch[0]['num_nodes']) is list:
        for record in batch:
            num_nodes += record['num_nodes']
    else:
        for record in batch:
            num_nodes.append(record['num_nodes'])

    num_trees = [record['num_trees'] for record in batch]
    batch_size = len(batch)

    return {
        'pfp': process_array(pfp, dtype='float32'),
        'rxnfp': process_array(rxnfp, dtype='float32'),
        'node_order': process_array(node_order, dtype='int64'),
        'adjacency_list': process_array(adjacency_list, dtype='int64'),
        'edge_order': process_array(edge_order, dtype='int64'),
        'num_nodes': num_nodes,
        'num_trees': num_trees,
        'batch_size': batch_size,
    }
