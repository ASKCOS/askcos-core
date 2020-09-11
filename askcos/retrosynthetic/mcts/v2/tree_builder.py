import itertools
import json
import time
from collections import OrderedDict, defaultdict

import networkx as nx
import numpy as np
from rdkit import Chem
from rdchiral.initialization import rdchiralReaction, rdchiralReactants

from askcos.retrosynthetic.mcts.utils import nx_graph_to_paths, nx_paths_to_json
from askcos.utilities.descriptors import rms_molecular_weight, number_of_rings
from askcos.utilities.io.logger import MyLogger

treebuilder_loc = 'mcts_tree_builder_v2'


class MCTS:
    """Monte Carlo Tree Search"""

    def __init__(self, pricer=None, chemhistorian=None, scscorer=None,
                 retro_transformer=None, use_db=False,
                 template_set='reaxys', template_prioritizer='reaxys',
                 precursor_prioritizer='relevanceheuristic',
                 fast_filter='default', **kwargs):

        self.tree = nx.DiGraph()  # directed graph

        self.target = None  # the target compound
        self.target_uuid = None  # unique identifier for the target in paths
        self.paths = None  # pathway results as nx graphs

        self.chemicals = []  # list of chemical smiles
        self.reactions = []  # list of reaction smiles

        self.iterations = 0
        self.time_to_solve = 0

        # Models and databases
        self.pricer = pricer or self.load_pricer(use_db)
        self.chemhistorian = chemhistorian or self.load_chemhistorian(use_db)
        self.scscorer = scscorer or self.load_scscorer(pricer=self.pricer)

        # If template prioritizer or fast filter are provided, don't load them
        if template_prioritizer is not None and not isinstance(template_prioritizer, str):
            self.template_prioritizer = template_prioritizer
            template_prioritizer = None

        if fast_filter is not None and not isinstance(fast_filter, str):
            self.fast_filter = fast_filter
            fast_filter = None

        self.retro_transformer = retro_transformer or self.load_retro_transformer(
            template_set=template_set,
            template_prioritizer=template_prioritizer,
            precursor_prioritizer=precursor_prioritizer,
            fast_filter=fast_filter,
        )
        if isinstance(template_prioritizer, str):
            self.template_prioritizer = self.retro_transformer.template_prioritizer
        if isinstance(template_prioritizer, str):
            self.fast_filter = self.retro_transformer.fast_filter
        self.template_set = template_set

        # Retro transformer options
        self.template_max_count = None
        self.template_max_cum_prob = None
        self.fast_filter_threshold = None

        # Tree generation options
        self.expansion_time = None
        self.max_iterations = None
        self.max_chemicals = None
        self.max_reactions = None
        self.max_branching = None
        self.max_depth = None
        self.exploration_weight = None
        self.return_first = None
        self.max_trees = None
        self.banned_chemicals = None
        self.banned_reactions = None

        # Terminal node criteria
        self.max_ppg = None
        self.max_scscore = None
        self.max_elements = None
        self.min_history = None
        self.termination_logic = None
        self.buyables_source = None

        # Parse any keyword arguments and set default options
        self.set_options(**kwargs)

    @property
    def done(self):
        """
        Determine if we're done expanding the tree.
        """
        return (
            self.is_chemical_done(self.target)
            or (self.max_iterations is not None and self.iterations >= self.max_iterations)
            or (self.max_chemicals is not None and len(self.chemicals) >= self.max_chemicals)
            or (self.max_reactions is not None and len(self.reactions) >= self.max_reactions)
        )

    def set_options(self, **kwargs):
        """
        Parse keyword arguments and save options to corresponding attributes.
        Backwards compatible with argument names from original tree builder.

        If no keyword arguments are provided, resets to default options.
        """
        # Retro transformer options
        self.template_max_count = kwargs.get('template_max_count', kwargs.get('template_count', 100))
        self.template_max_cum_prob = kwargs.get('template_max_cum_prob', kwargs.get('max_cum_template_prob', 0.995))
        self.fast_filter_threshold = kwargs.get('fast_filter_threshold', 0.75)

        # Tree generation options
        self.expansion_time = kwargs.get('expansion_time', 30)
        self.max_iterations = kwargs.get('max_iterations', None)
        self.max_chemicals = kwargs.get('max_chemicals', None)
        self.max_reactions = kwargs.get('max_reactions', None)
        self.max_branching = kwargs.get('max_branching', 25)
        self.max_depth = kwargs.get('max_depth', 10)
        self.exploration_weight = kwargs.get('exploration_weight', 1.0)
        self.return_first = kwargs.get('return_first', False)
        self.max_trees = kwargs.get('max_trees', None)
        self.banned_chemicals = kwargs.get('banned_chemicals', kwargs.get('forbidden_molecules', []))
        self.banned_reactions = kwargs.get('banned_reactions', kwargs.get('known_bad_reactions', []))

        # Terminal node criteria
        self.max_ppg = kwargs.get('max_ppg', None)
        self.max_scscore = kwargs.get('max_scscore', None)
        self.max_elements = kwargs.get('max_elements', None)
        self.min_history = kwargs.get('min_history', None)
        self.termination_logic = kwargs.get('termination_logic', {'and': ['buyable']})
        self.buyables_source = kwargs.get('buyables_source', None)

    def to_branching(self):
        """
        Get branching representation of the tree.
        """
        branching = nx.dag_to_branching(self.tree)
        # Copy node attributes from original graph
        for node, data in branching.nodes(data=True):
            smiles = data.pop('source')
            data['smiles'] = smiles
            data.update(self.tree.nodes[smiles])
        return branching

    def get_graph(self):
        """
        Return cleaned version of original graph.
        """
        self.update_reaction_data()
        graph = self.tree.copy(as_view=False)

        # Remove unnecessary attributes
        attr_to_keep = {
            'smiles',
            'type',
            'purchase_price',
            'as_reactant',
            'as_product',
            'terminal',
            'plausibility',
            'template_score',
            'tforms',
            'num_examples',
            'necessary_reagent',
            'precursor_smiles',
            'rms_molwt',
            'num_rings',
            'scscore',
            'rank',
        }
        for node, node_data in graph.nodes.items():
            attr_to_remove = [attr for attr in node_data if attr not in attr_to_keep]
            for attr in attr_to_remove:
                del node_data[attr]

        return graph

    def get_union_of_paths(self):
        """
        Returns the union of self.paths as a single tree.
        """
        if self.paths:
            return nx.compose_all(self.paths)

    @staticmethod
    def load_chemhistorian(use_db):
        """
        Loads chemhistorian.
        """
        from askcos.utilities.historian.chemicals import ChemHistorian
        chemhistorian = ChemHistorian(use_db=use_db)
        chemhistorian.load()
        return chemhistorian

    @staticmethod
    def load_pricer(use_db):
        """
        Loads pricer.
        """
        from askcos.utilities.buyable.pricer import Pricer
        pricer = Pricer(use_db=use_db)
        pricer.load()
        return pricer

    @staticmethod
    def load_scscorer(pricer=None):
        """
        Loads pricer.
        """
        from askcos.prioritization.precursors.scscore import SCScorePrecursorPrioritizer
        scscorer = SCScorePrecursorPrioritizer(pricer=pricer)
        scscorer.load_model(model_tag='1024bool')
        return scscorer

    @staticmethod
    def load_retro_transformer(template_set='reaxys', template_prioritizer='reaxys',
                               precursor_prioritizer='relevanceheuristic', fast_filter='default'):
        """
        Loads retro transformer model.
        """
        from askcos.retrosynthetic.transformer import RetroTransformer
        retro_transformer = RetroTransformer(
            template_set=template_set,
            template_prioritizer=template_prioritizer,
            precursor_prioritizer=precursor_prioritizer,
            fast_filter=fast_filter,
            scscorer=None,
        )
        retro_transformer.load()
        return retro_transformer

    def get_buyable_paths(self, target, **kwargs):
        """
        Build retrosynthesis tree and return paths to buyable precursors.
        """
        self.build_tree(target, **kwargs)
        paths = self.enumerate_paths(**kwargs)
        status = self.tree_status()
        graph = nx.node_link_data(self.get_graph())
        return paths, status, graph

    def build_tree(self, target, **kwargs):
        """
        Build retrosynthesis tree by iterative expansion of precursor nodes.
        """
        self.set_options(**kwargs)

        MyLogger.print_and_log('Initializing tree...', treebuilder_loc)
        self._initialize(target)

        MyLogger.print_and_log('Starting tree expansion...', treebuilder_loc)
        start_time = time.time()
        elapsed_time = time.time() - start_time

        while elapsed_time < self.expansion_time and not self.done:
            self._rollout()

            elapsed_time = time.time() - start_time

            self.iterations += 1
            if self.iterations % 100 == 0:
                MyLogger.print_and_log('Iteration {0} ({1:.2f}s): |C| = {2} |R| = {3}'.format(self.iterations, elapsed_time, len(self.chemicals), len(self.reactions)), treebuilder_loc)

            if not self.time_to_solve and self.tree.nodes[self.target]['solved']:
                self.time_to_solve = elapsed_time
                MyLogger.print_and_log('Found first pathway after {:.2f} seconds.'.format(elapsed_time), treebuilder_loc)
                if self.return_first:
                    MyLogger.print_and_log('Stopping expansion to return first pathway.', treebuilder_loc)
                    break

        MyLogger.print_and_log('Tree expansion complete.', treebuilder_loc)
        self.print_stats()

    def print_stats(self):
        """
        Print tree statistics.
        """
        info = '\n'
        info += 'Number of iterations: {0}\n'.format(self.iterations)
        num_nodes = self.tree.number_of_nodes()
        info += 'Number of nodes: {0:d}\n'.format(num_nodes)
        info += '    Chemical nodes: {0:d}\n'.format(len(self.chemicals))
        info += '    Reaction nodes: {0:d}\n'.format(len(self.reactions))
        info += 'Number of edges: {0:d}\n'.format(self.tree.number_of_edges())
        if num_nodes > 0:
            info += 'Average in degree: {0:.4f}\n'.format(sum(d for _, d in self.tree.in_degree()) / num_nodes)
            info += 'Average out degree: {0:.4f}'.format(sum(d for _, d in self.tree.out_degree()) / num_nodes)
        MyLogger.print_and_log(info, treebuilder_loc)

    def tree_status(self):
        """
        Summarize statistics for tree exploration.

        Returns:
            (int, int, int):
                num_chemicals (int): Number of chemical nodes in the tree.
                num_reactions (int): Number of reaction nodes in the tree.
                num_templates (int): Total number of applied templates.
        """
        num_chemicals = len(self.chemicals)
        num_reactions = len(self.reactions)
        num_templates = sum(len(self.tree.nodes[chem]['explored']) for chem in self.chemicals)
        return num_chemicals, num_reactions, num_templates

    def clear(self):
        """
        Clear tree and reset chemicals and reactions.
        """
        self.tree.clear()
        self.chemicals = []
        self.reactions = []

    def dump_tree(self):
        """
        Serialize entire tree to json.
        """
        return json.dumps(nx.node_link_data(self.tree))

    def load_tree(self, data):
        """
        Deserialize and parse tree from json.
        """
        self.tree = nx.node_link_graph(json.loads(data))

    def _initialize(self, target):
        """
        Initialize the tree by with the target chemical.
        """
        self.target = target
        self.create_chemical_node(self.target)
        self.tree.nodes[self.target]['terminal'] = False
        self.tree.nodes[self.target]['done'] = False
        self.tree.nodes[self.target]['solved'] = False

    def _rollout(self):
        """
        Perform one iteration of tree expansion
        """
        chem_path, rxn_path, template = self._select()
        self._expand(chem_path, template)
        self._update(chem_path, rxn_path)

    def _expand(self, chem_path, template):
        """
        Expand the tree by applying chosen template to a chemical node.
        """
        leaf = chem_path[-1]
        explored = self.tree.nodes[leaf]['explored']
        if template not in explored:
            explored.append(template)
            precursors = self._get_precursors(leaf, template)
            self._process_precursors(leaf, template, precursors)

    def _update(self, chem_path, rxn_path):
        """
        Update status and reward for nodes in this path.

        Reaction nodes are guaranteed to only have a single parent. Thus, the
        status of its parent chemical will always be updated appropriately in
        ``_update`` and will not change until the next time the chemical is
        in the selected path. Thus, the done state of the chemical can be saved.

        However, chemical nodes can have multiple parents (i.e. can be reached
        via multiple reactions), so a given update cycle may only pass through
        one of multiple parent reactions. Thus, the done state of a reaction
        must be determined dynamically and cannot be saved.
        """
        assert chem_path[0] == self.target, 'Chemical path should start at the root node.'

        # Iterate over the full path in reverse
        # On each iteration, rxn will be the parent reaction of chem
        # For the root (target) node, rxn will be None
        for i, chem, rxn in itertools.zip_longest(range(len(chem_path)-1, -1, -1), reversed(chem_path), reversed(rxn_path)):
            chem_data = self.tree.nodes[chem]
            chem_data['visit_count'] += 1
            chem_data['min_depth'] = min(chem_data['min_depth'], i) if chem_data['min_depth'] is not None else i
            self.is_chemical_done(chem, update=True)
            if rxn is not None:
                rxn_data = self.tree.nodes[rxn]
                rxn_data['visit_count'] += 1
                self._update_value(rxn)

    def is_chemical_done(self, smiles, update=False):
        """
        Determine if the specified chemical node should be expanded further.

        If ``update=True``, will reassess the done state of the node, update
        the ``done`` attribute, and return the new result.

        Otherwise, return the ``done`` node attribute.

        Chemical nodes are done when one of the following is true:
        - The node is terminal
        - The node has exceeded max_depth
        - The node as exceeded max_branching
        - The node does not have any templates to expand
        """
        if update:
            data = self.tree.nodes[smiles]
            done = False
            if data['terminal']:
                done = True
            elif len(data['templates']) == 0:
                done = True
            elif data['min_depth'] is not None and data['min_depth'] >= self.max_depth:
                done = True
            elif self.tree.out_degree(smiles) >= self.max_branching or len(data['explored']) == len(data['templates']):
                done = all(self.is_reaction_done(r) for r in self.tree.successors(smiles))
            data['done'] = done
            return done
        else:
            return self.tree.nodes[smiles]['done']

    def is_reaction_done(self, smiles):
        """
        Determine if the specified reaction node should be expanded further.

        Reaction nodes are done when all of its children chemicals are done.
        """
        return self.tree.out_degree(smiles) > 0 and all(self.is_chemical_done(c) for c in self.tree.successors(smiles))

    def _select(self):
        """
        Select next leaf node to be expanded.

        This starts at the root node (target chemical), and at each level,
        use UCB to score each of the options which can be taken. It will take
        the optimal option, which may be a new template application, or an
        already explored reaction. For the latter, it will descend to the next
        level and repeat the process until a new template application is chosen.
        """
        chem_path = [self.target]
        rxn_path = []
        invalid_options = set()
        template = None
        while template is None:
            leaf = chem_path[-1]
            options = self.ucb(leaf, chem_path, invalid_options, self.exploration_weight)

            if not options:
                # There are no valid options from this chemical node, we need to backtrack
                invalid_options.add(leaf)
                del chem_path[-1]
                del rxn_path[-1]
                continue

            # Get the best option
            score, task = options[0]

            if isinstance(task, str):
                # This is an already explored reaction, so we need to descend the tree
                # If there are multiple reactants, pick the one with the lower visit count
                # Do not consider chemicals that are already done or chemicals that are on the path
                precursor = min(
                    (c for c in self.tree.successors(task)
                        if not self.is_chemical_done(c)
                        and c not in invalid_options),
                    key=lambda x: self.tree.nodes[x]['visit_count'],
                    default=None,
                )
                if precursor is None:
                    # There are no valid options from this reaction node, we need to backtrack
                    invalid_options.add(task)
                    continue
                else:
                    chem_path.append(precursor)
                    rxn_path.append(task)
            else:
                # This is a new template to apply
                template = task

        return chem_path, rxn_path, template

    def ucb(self, node, path, invalid_options, exploration_weight):
        """
        Calculate UCB score for all exploration options from the specified node.

        This algorithm considers both explored and unexplored template
        applications as potential routes for further exploration.

        Returns a list of (score, option) tuples sorted by score.
        """
        options = []

        templates = self.tree.nodes[node]['templates']
        explored = self.tree.nodes[node]['explored']
        product_visits = self.tree.nodes[node]['visit_count']

        # Get scores for explored templates (reaction node exists)
        for rxn in self.tree.successors(node):
            rxn_data = self.tree.nodes[rxn]

            if self.is_reaction_done(rxn) or len(set(self.tree.successors(rxn)) & set(path)) > 0 or rxn in invalid_options:
                continue

            est_value = rxn_data['est_value']
            node_visits = rxn_data['visit_count']
            template_probability = sum([templates[t] for t in rxn_data['templates']])

            # Q represents how good a move is
            q_sa = template_probability * est_value / node_visits
            # U represents how many times this move has been explored
            u_sa = np.sqrt(np.log(product_visits) / node_visits)

            score = q_sa + exploration_weight * u_sa

            # The options here are to follow a reaction down one level
            options.append((score, rxn))

        # Get score for most relevant unexplored template
        if self.tree.out_degree(node) < self.max_branching or node == self.target:
            for template_index in templates:
                if template_index not in explored:
                    q_sa = templates[template_index]
                    u_sa = np.sqrt(np.log(product_visits))
                    score = q_sa + exploration_weight * u_sa

                    # The options here are to apply a new template to this chemical
                    options.append((score, template_index))
                    break

        # Sort options from highest to lowest score
        options.sort(key=lambda x: x[0], reverse=True)

        return options

    def _get_precursors(self, chemical, template_idx):
        """
        Get all precursors from applying a template to a chemical.
        """
        mol = Chem.MolFromSmiles(chemical)
        smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
        mol = rdchiralReactants(smiles)

        template = self.retro_transformer.get_one_template_by_idx(template_idx, template_set=self.template_set)
        try:
            template['rxn'] = rdchiralReaction(template['reaction_smarts'])
        except ValueError:
            return []

        outcomes = self.retro_transformer.apply_one_template(mol, template)

        precursors = [o['smiles_split'] for o in outcomes]

        return precursors

    def _process_precursors(self, target, template, precursors):
        """
        Process a list of precursors:
        1. Filter precursors by fast filter score
        2. Create and register Chemical objects for each new precursor
        3. Generate template relevance probabilities
        4. Create and register Reaction objects
        """
        for reactant_list in precursors:
            reactant_smiles = '.'.join(reactant_list)
            reaction_smiles = reactant_smiles + '>>' + target

            # Check if this precursor meets the fast filter score threshold
            ff_score = self.fast_filter(reactant_smiles, target)
            if ff_score < self.fast_filter_threshold:
                continue

            # Check if the reaction is banned
            if reaction_smiles in self.banned_reactions:
                continue

            # Check if any precursors are banned
            if any(reactant in self.banned_chemicals for reactant in reactant_list):
                continue

            template_score = self.tree.nodes[target]['templates'][template]

            if reaction_smiles in self.reactions:
                # This reaction already exists
                rxn_data = self.tree.nodes[reaction_smiles]
                rxn_data['templates'].append(template)
                rxn_data['template_score'] = max(rxn_data['template_score'], template_score)
            else:
                # This is new, so create a Reaction node
                self.create_reaction_node(reaction_smiles, template, template_score, ff_score)

            # Add edges to connect target -> reaction -> precursors
            self.tree.add_edge(target, reaction_smiles)
            for reactant in reactant_list:
                if reactant not in self.chemicals:
                    # This is new, so create a Chemical node
                    self.create_chemical_node(reactant)

                self.tree.add_edge(reaction_smiles, reactant)

            self._update_value(reaction_smiles)

    def create_chemical_node(self, smiles):
        """
        Create a new chemical node from the provide SMILES and populate node
        properties with chemical data.

        Includes template relevance probabilities and purchase price.
        """
        probs, indices = self.template_prioritizer.predict(
            smiles,
            max_num_templates=self.template_max_count,
            max_cum_prob=self.template_max_cum_prob,
        )
        templates = OrderedDict(zip(indices.tolist(), probs.tolist()))

        purchase_price = self.pricer.lookup_smiles(smiles, source=self.buyables_source, alreadyCanonical=True)

        hist = self.chemhistorian.lookup_smiles(smiles, alreadyCanonical=True, template_set=self.template_set)

        terminal = self.is_terminal(smiles, purchase_price, hist)
        est_value = 1. if terminal else 0.

        self.chemicals.append(smiles)
        self.tree.add_node(
            smiles,
            as_reactant=hist['as_reactant'],
            as_product=hist['as_product'],
            est_value=est_value,      # total value of node
            explored=[],              # list of explored templates
            min_depth=None,           # minimum depth at which this chemical appears in the tree
            purchase_price=purchase_price,
            solved=terminal,          # whether a path to terminal leaves has been found from this node
            templates=templates,      # dict of template indices to relevance probabilities
            terminal=terminal,        # whether this chemical meets terminal criterial
            type='chemical',
            visit_count=1,
        )

        self.is_chemical_done(smiles, update=True)

    def create_reaction_node(self, smiles, template, template_score, ff_score):
        """
        Create a new reaction node from the provided smiles and data.
        """
        self.reactions.append(smiles)
        self.tree.add_node(
            smiles,
            est_value=0.,       # score for how feasible a route is, based on whether its precursors are terminal
            plausibility=ff_score,
            solved=False,             # whether a path to terminal leaves has been found from this node
            template_score=template_score,
            templates=[template],
            type='reaction',
            visit_count=1,
        )

    def _update_value(self, smiles):
        """
        Update the value of the specified reaction node and its parent.
        """
        rxn_data = self.tree.nodes[smiles]

        if rxn_data['type'] == 'reaction':
            # Calculate value as the sum of the values of all precursors
            est_value = sum(self.tree.nodes[c]['est_value'] for c in self.tree.successors(smiles))

            # Update estimated value of reaction
            rxn_data['est_value'] += est_value

            # Update estimated value of parent chemical
            chem_data = self.tree.nodes[next(self.tree.predecessors(smiles))]
            chem_data['est_value'] += est_value

            # Check if this node is solved
            solved = rxn_data['solved'] or all(self.tree.nodes[c]['solved'] for c in self.tree.successors(smiles))
            chem_data['solved'] = rxn_data['solved'] = solved

    def is_terminal(self, smiles, ppg=None, hist=None):
        """
        Determine if the specified chemical is a terminal node in the tree based
        on pre-specified criteria.

        Criteria to be considered are specified via ``self.termination_logic``,
        and the thresholds for each criteria are specified separately.

        If no criteria are specified, will always return ``False``.

        Args:
            smiles (str): smiles string of the chemical
            ppg (float): cost of the chemical
            hist (dict): historian data for the chemical
        """
        def buyable():
            return bool(ppg)

        def max_ppg():
            if self.max_ppg is not None:
                # ppg of 0 means not buyable
                return ppg is not None and 0 < ppg <= self.max_ppg
            return True

        def max_scscore():
            if self.max_scscore is not None:
                scscore = self.scscorer.get_score_from_smiles(smiles, noprice=True)
                return scscore <= self.max_scscore
            return True

        def max_elements():
            if self.max_elements is not None:
                # Get structural properties
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    elem_dict = defaultdict(lambda: 0)
                    for a in mol.GetAtoms():
                        elem_dict[a.GetSymbol()] += 1
                    elem_dict['H'] = sum(a.GetTotalNumHs() for a in mol.GetAtoms())

                    return all(elem_dict[k] <= v for k, v in self.max_elements.items())
            return True

        def min_history():
            if self.min_history is not None:
                return hist is not None and (hist['as_reactant'] >= self.min_history['as_reactant'] or
                                             hist['as_product'] >= self.min_history['as_product'])
            return True

        local_dict = locals()
        or_criteria = self.termination_logic.get('or')
        and_criteria = self.termination_logic.get('and')

        return (bool(or_criteria) and any(local_dict[criteria]() for criteria in or_criteria) or
                bool(and_criteria) and all(local_dict[criteria]() for criteria in and_criteria))

    def enumerate_paths(self, path_format='json', json_format='treedata', sorting_metric='plausibility',
                        validate_paths=True, **kwargs):
        """
        Return list of paths to buyables starting from the target node.

        Args:
            path_format (str): pathway output format, supports 'graph' or 'json'
            json_format (str): networkx json format, supports 'treedata' or 'nodelink'
            sorting_metric (str): how pathways are sorted, supports 'plausibility',
                'number_of_starting_materials', 'number_of_reactions'
            validate_paths (bool): require all leaves to meet terminal criteria

        Returns:
            list of paths in specified format
        """
        # Resolve template data before doing any node duplication
        self.update_reaction_data()

        self.paths, self.target_uuid = nx_graph_to_paths(
            self.tree,
            self.target,
            max_depth=self.max_depth,
            max_trees=self.max_trees,
            sorting_metric=sorting_metric,
            validate_paths=validate_paths,
        )

        MyLogger.print_and_log('Found {0} paths to buyable chemicals.'.format(len(self.paths)), treebuilder_loc)

        if path_format == 'graph':
            paths = self.paths
        elif path_format == 'json':
            paths = nx_paths_to_json(self.paths, self.target_uuid, json_format=json_format)
        else:
            raise ValueError('Unrecognized format type {0}'.format(path_format))

        return paths

    def update_reaction_data(self):
        """
        Update metadata for all reaction nodes.

        Adds the following fields:
            * rank
            * tforms
            * num_examples
            * necessary_reagent
            * precursor_smiles
            * num_rings
            * scscore
        """
        for chem in self.chemicals:
            reactions = sorted(self.tree.successors(chem),
                               key=lambda x: self.tree.nodes[x]['template_score'],
                               reverse=True)
            for i, rxn in enumerate(reactions):
                self.tree.nodes[rxn]['rank'] = i + 1

        for rxn in self.reactions:
            rxn_data = self.tree.nodes[rxn]
            template_ids = rxn_data['templates']
            info = self.retro_transformer.retrieve_template_metadata(template_ids)
            rxn_data.update(info)

            precursor_smiles = rxn.split('>>')[0]
            rxn_data['precursor_smiles'] = precursor_smiles
            rxn_data['rms_molwt'] = rms_molecular_weight(precursor_smiles)
            rxn_data['num_rings'] = number_of_rings(precursor_smiles)
            rxn_data['scscore'] = self.scscorer.get_score_from_smiles(precursor_smiles, noprice=True)
