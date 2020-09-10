import hdbscan
import numpy as np
import sklearn.cluster as cluster

from askcos.retrosynthetic.pathway_ranker.utils import convert_askcos_trees, tree_to_input, merge_into_batch


class PathwayRanker:

    def __init__(self):
        # Fingerprint and LSTM sizes are fixed by the current model
        self.fp_size = 2048
        self.lstm_size = 512
        self.device = None
        self.model = None

    def load(self, model_path=None):
        """
        Initialize PyTorch model.
        """
        # Imports intentionally placed here so that inherited classes can avoid importing pytorch
        import torch
        from askcos.retrosynthetic.pathway_ranker.model import PathwayRankingModel

        if model_path is None:
            import askcos.global_config as gc
            model_path = gc.PATHWAY_RANKER['model_path']

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        state = torch.load(model_path, map_location=self.device)
        state_dict = state['state_dict']

        self.model = PathwayRankingModel(self.fp_size, self.lstm_size, encoder=True)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    def preprocess(self, trees):
        """
        Convert trees into fingerprints for model input.

        One-step trees are removed from the input because they cannot be ranked.

        The original indices of the remaining trees are also returned to help
        map the results to the original list of trees.

        Args:
            trees (list): list of askcos trees to process

        Returns:
            original_indices (list): list of the original indices of the selected trees
            batch (dict): dictionary of generated fingerprints and metadata
        """
        import torch

        output = convert_askcos_trees(trees)

        # Separate out one-step trees because they can't be ranked
        original_indices, remaining_trees = zip(*((i, tree) for i, tree in enumerate(output) if tree['depth'] > 1))

        # Convert trees to set of input fingerprints and metadata
        batch = merge_into_batch([tree_to_input(tree) for tree in remaining_trees])

        # Convert fingerprint arrays into tensors
        batch['pfp'] = torch.tensor(batch['pfp'], device=self.device, dtype=torch.float32)
        batch['rxnfp'] = torch.tensor(batch['rxnfp'], device=self.device, dtype=torch.float32)
        batch['node_order'] = torch.tensor(batch['node_order'], device=self.device, dtype=torch.int64)
        batch['adjacency_list'] = torch.tensor(batch['adjacency_list'], device=self.device, dtype=torch.int64)
        batch['edge_order'] = torch.tensor(batch['edge_order'], device=self.device, dtype=torch.int64)

        return list(original_indices), batch

    def inference(self, data):
        """
        Run the pathway ranking model on the prepared input data.
        """
        pfp = data['pfp']
        rxnfp = data['rxnfp']
        adjacency_list = data['adjacency_list']
        node_order = data['node_order']
        edge_order = data['edge_order']
        num_nodes = data['num_nodes']

        # Forward pass
        scores, encoded_trees = self.model(pfp, rxnfp, adjacency_list, node_order, edge_order, num_nodes)

        return {'scores': scores, 'encoded_trees': encoded_trees}

    def postprocess(self, data):
        return {
            'scores': data['scores'].view(-1,).tolist(),
            'encoded_trees': data['encoded_trees'].tolist(),
        }

    def scorer(self, trees, clustering=False, cluster_method='hdbscan', min_samples=5, min_cluster_size=5):
        """
        Score a list of retrosynthetic trees relative to each other and also
        cluster them if requested.

        One-step trees cannot be scored and are removed for scoring.
        The result arrays correspond exactly to the order of the input trees,
        with placeholder values for any one-step trees:
            * For scores and clusters, the placeholder value is -1.
            * For encoded_trees, the placeholder value is [] (empty list).

        Returns:
            output (dict):
                scores: list of scores assigned by the pathway ranker
                encoded_trees: list of embedded representations of the trees
                clusters: list of cluster IDs (if requested)
        """
        def _fill(array, indices, length, default=-1):
            """
            Insert values from array at specified indices to create a new list
            with the specified length and empty indices filled in with default.
            """
            mapping = dict(zip(indices, array))
            return [mapping.get(i, default) for i in range(length)]

        original_indices, batch = self.preprocess(trees)
        data = self.inference(batch)
        result = self.postprocess(data)

        scores, encoded_trees = result['scores'], result['encoded_trees']

        num_trees = len(trees)
        output = {
            'scores': _fill(scores, original_indices, num_trees),
            'encoded_trees': _fill(encoded_trees, original_indices, num_trees, default=[]),
        }

        if clustering:
            clusters = self._cluster_encoded_trees(
                [np.array(encoding) for encoding in encoded_trees],
                scores=scores,
                cluster_method=cluster_method,
                min_samples=min_samples,
                min_cluster_size=min_cluster_size
            )
            output['clusters'] = _fill(clusters, original_indices, num_trees)

        return output

    @staticmethod
    def _cluster_encoded_trees(encoded_trees, scores=None, cluster_method='hdbscan', min_samples=5, min_cluster_size=5):
        """
        Cluster the input trees using the specified options.

        If ``scores`` is provided, the cluster IDs will be renumbered in order
        based on the best score in each cluster.

        Args:
            encoded_trees (list): tree embeddings as numpy arrays
            scores (list, optional): tree scores from pathway ranker
            cluster_method (str, optional): hdbscan or kmeans
            min_samples (int, optional): min samples for hdbscan
            min_cluster_size (int, optional): min cluster size for hdbscan
        """
        if not encoded_trees:
            return []

        if cluster_method == 'hdbscan':
            clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                                        min_samples=min_samples,
                                        gen_min_span_tree=False)
            clusterer.fit(encoded_trees)
            res = clusterer.labels_
            # non-clustered inputs have id -1, make them appear as individual clusters
            max_cluster = np.amax(res)
            for i in range(len(res)):
                if res[i] == -1:
                    max_cluster += 1
                    res[i] = max_cluster
        elif cluster_method == 'kmeans':  # seems to be very slow
            for cluster_size in range(len(encoded_trees)):
                kmeans = cluster.KMeans(n_clusters=cluster_size + 1).fit(encoded_trees)
                if kmeans.inertia_ < 1:
                    break
            res = kmeans.labels_
        else:
            raise ValueError('Unsupported value for cluster_method: {0}'.format(cluster_method))

        res = [int(i) for i in res]

        if scores is not None:
            # Renumber the clusters based on best score
            if len(scores) != len(res):
                raise ValueError('Length of trees ({}) and scores ({}) are different.'.format(len(res), len(scores)))

            best_cluster_score = {}
            for cluster_id, score in zip(res, scores):
                best_cluster_score[cluster_id] = max(
                    best_cluster_score.get(cluster_id, -float('inf')),
                    score
                )

            new_order = sorted(best_cluster_score.keys(), key=lambda x: best_cluster_score[x], reverse=True)
            order_mapping = {cluster_id: i for i, cluster_id in enumerate(new_order)}
            res = [order_mapping[n] for n in res]

        return res
