import os
import pickle
import queue as VanillaQueue
import random
import time
from collections import defaultdict
from multiprocessing import Process, Manager, Queue

import numpy as np
import rdkit.Chem as Chem
from pymongo import MongoClient

import askcos.global_config as gc
from askcos.retrosynthetic.mcts.nodes import Chemical, Reaction, ChemicalTemplateApplication
from askcos.retrosynthetic.transformer import RetroTransformer
from askcos.utilities.buyable.pricer import Pricer
from askcos.utilities.formats import chem_dict, rxn_dict
from askcos.utilities.io.logger import MyLogger
from askcos.utilities.historian.chemicals import ChemHistorian
from askcos.prioritization.precursors.scscore import SCScorePrecursorPrioritizer
from askcos.prioritization.templates.relevance import RelevanceTemplatePrioritizer
from askcos.synthetic.evaluation.fast_filter import FastFilterScorer

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

treebuilder_loc = 'mcts_tree_builder'

VIRTUAL_LOSS = 1000000

WAITING = 0
DONE = 1


class MCTS:
    """
    This class implements a Monte Carlo Tree Search algorithm for retrosynthetic
    tree exploration. Individual retrosynthetic trees are then enumerated via a
    depth-first search.

    This implementation uses native Python multiprocessing.

    Note regarding model and data loading: This class uses pricing data,
    chemhistorian data, template prioritizer, and retro transformer. The retro
    transformer additionally needs the precursor prioritizer and fast filter.
    If instantiating this class with no arguments, the defaults will be loaded
    for all of these. Otherwise, Pricer, ChemHistorian, and RetroTransformer
    instances can be passed during initiation. The RetroTransformer should not
    include the template prioritizer or fast filter models as they need to be
    independently loaded by the children processes.

    Attributes:

    """

    def __init__(self, retroTransformer=None, pricer=None, scscorer=None,
                 max_branching=20, max_depth=3, expansion_time=60,
                 chemhistorian=None, nproc=8, num_active_pathways=None, template_set='reaxys',
                 template_prioritizer='reaxys', precursor_prioritizer='relevanceheuristic', fast_filter='default',
                 **kwargs):
        """
        Initialization of an object of the MCTS class.

        Sets default values for various settings and loads transformers as
        needed (i.e., based on whether Celery is being used or not).
        Most settings are overridden by the get_buyable_paths method anyway.

        Args:
            retroTransformer (None or RetroTransformer, optional):
                RetroTransformer object to be used for expansion when *not*
                using Celery. If None, will be initialized using the
                model_loader.load_Retro_Transformer function. (default: {None})
            pricer (None or Pricer, optional): Pricer object to be used for
                checking stop criteria (buyability). If None, will be
                initialized using default settings from the global
                configuration. (default: {None})
            scscorer (None or SCScorePrecursorPrioritizer, optional): SCScore
                instance for evaluating termination criteria. If None, will be
                loaded using default settings. (default: {None})
            max_branching (int, optional): Maximum number of precursor
                suggestions to add to the tree at each expansion.
                (default: {20})
            max_depth (int, optional): Maximum number of reactions to allow
                before stopping the recursive expansion down one branch.
                (default: {3})
            expansion_time (int, optional): Time (in seconds) to allow for
                expansion before searching the generated tree for buyable
                pathways. (default: {60})
            nproc (int, optional): Number of retrotransformer processes to fork
                for faster expansion. (default: {1})
            chem_historian (None or ChemHistorian, optional): ChemHistorian
                object used to see how often chemicals have occured in
                database. If None, will be loaded from the default file in the
                global configuration. (default: {None})
            num_active_pathways (None or int, optional): Number of active
                pathways. If None, will be set to ``nproc``. (default: {None})
            template_prioritizer (str, optional): Specifies which template
                prioritizer to use. (default: {'reaxys'})
            template_set (str, optional): Specifies which template set to use.
                (default: {'reaxys'})
            precursor_prioritizer (str, optional): Specify precursor prioritizer
                to be used. (default: {'relevanceheuristic'})
            fast_filter (str, optional): Specify fast filter to be used for
                scoring reactions. (default: {'default'})
        """

        if 'chiral' in kwargs and not kwargs['chiral']:
            raise ValueError('MCTS only works for chiral expansion!')

        self.max_depth = max_depth
        self.max_branching = max_branching
        self.expansion_time = expansion_time
        self.nproc = nproc
        self.sort_trees_by = 'plausibility'

        self.num_active_pathways = num_active_pathways or self.nproc

        # Initialize other class attributes, these are set by `get_buyable_paths`
        self.smiles = None
        self.max_trees = None
        self.max_cum_template_prob = None
        self.template_count = None
        self.filter_threshold = None
        self.apply_fast_filter = None
        self.max_ppg = None
        self.max_scscore = None
        self.max_elements = None
        self.min_history = None
        self.termination_logic = None
        self.buyables_source = None

        # Load data and models
        self.pricer = pricer or self.load_pricer(kwargs.get('use_db', False))
        self.scscorer = scscorer or self.load_scscorer(pricer=self.pricer)
        self.chemhistorian = chemhistorian or self.load_chemhistorian(kwargs.get('use_db', False))
        self.retroTransformer = retroTransformer or self.load_retro_transformer(
            template_set=template_set,
            precursor_prioritizer=precursor_prioritizer,
        )
        # The template prioritizer and fast filter are TF models which must be loaded in each child process
        self.template_prioritizer = template_prioritizer
        self.fast_filter = fast_filter

        # Initialize vars, reset dicts, etc.
        self.reset(soft_reset=False)

        from askcos.utilities.with_dummy import with_dummy
        self.allow_join_result = with_dummy

    @staticmethod
    def load_pricer(use_db):
        """
        Loads pricer.
        """
        pricer = Pricer(use_db=use_db)
        pricer.load()
        return pricer

    @staticmethod
    def load_scscorer(pricer=None):
        """
        Loads pricer.
        """
        scscorer = SCScorePrecursorPrioritizer(pricer=pricer)
        scscorer.load_model(model_tag='1024bool')
        return scscorer

    @staticmethod
    def load_chemhistorian(use_db):
        """
        Loads chemhistorian.
        """
        chemhistorian = ChemHistorian(use_db=use_db)
        chemhistorian.load()
        return chemhistorian

    @staticmethod
    def load_retro_transformer(template_set='reaxys', precursor_prioritizer='relevanceheuristic'):
        """
        Loads retro transformer model.
        """
        retro_transformer = RetroTransformer(
            template_set=template_set,
            template_prioritizer=None,
            precursor_prioritizer=precursor_prioritizer,
            fast_filter=None
        )
        retro_transformer.load()
        return retro_transformer

    @staticmethod
    def load_template_prioritizer(tp):
        """
        Loads template prioritizer model.
        """
        if tp in gc.RELEVANCE_TEMPLATE_PRIORITIZATION:
            template_prioritizer = RelevanceTemplatePrioritizer()
            template_prioritizer.load_model(gc.RELEVANCE_TEMPLATE_PRIORITIZATION[tp]['model_path'])
        else:
            raise ValueError('Unsupported template prioritizer "{0}"'.format(tp))
        return template_prioritizer

    @staticmethod
    def load_fast_filter(ff):
        if ff == 'default':
            fast_filter_object = FastFilterScorer()
            fast_filter_object.load()

            def fast_filter(x, y):
                return fast_filter_object.evaluate(x, y)[0][0]['score']
        else:
            raise ValueError('Unsupported fast filter "{0}"'.format(ff))
        return fast_filter

    def reset(self, soft_reset=False):
        """
        Prepare for a new expansion

        Args:
            soft_reset (bool, optional): Whether to do a soft reset.
                (default: {False})
        """
        self.reset_workers(soft_reset=soft_reset)
        self.running = False
        self.status = {}
        self.active_pathways = [{} for _id in range(self.num_active_pathways)]
        self.active_pathways_pending = [0 for _id in range(self.num_active_pathways)]
        self.pathway_count = 0
        self.mincost = 10000.0
        self.Chemicals = {} # new
        self.Reactions = {} # new
        self.time_for_first_path = -1

    def reset_workers(self, soft_reset=False):
        """
        Reset workers in preparation for a new expansion.
        """
        if not soft_reset:
            MyLogger.print_and_log('Doing a hard worker reset', treebuilder_loc)
            self.workers = []
            self.manager = Manager()
            self.done = self.manager.Value('i', 0)
            self.idle = self.manager.list()
            self.initialized = self.manager.list()
            for i in range(self.nproc):
                self.idle.append(True)
                self.initialized.append(False)
            self.expansion_queue = Queue()
            self.results_queue = Queue()
        else:
            MyLogger.print_and_log('Doing a soft worker reset', treebuilder_loc)
            for i in range(self.nproc):
                self.idle[i] = True
            try:
                while True:
                    self.expansion_queue.get(timeout=1)
            except VanillaQueue.Empty:
                pass

            try:
                while True:
                    self.results_queue.get(timeout=1)
            except VanillaQueue.Empty:
                pass

    def expand(self, _id, smiles, template_idx):
        """
        Adds pathway to be worked on with multiprocessing.

        Args:
            _id (int): ID of pending pathway.
            smiles (str): SMILES string of molecule to be expanded.
            template_idx (int): ID of template to apply to molecule.
        """
        self.expansion_queue.put((_id, smiles, template_idx))
        self.status[(smiles, template_idx)] = WAITING
        self.active_pathways_pending[_id] += 1

    def prepare(self):
        """Starts parallelization with multiprocessing."""
        if len(self.workers) == self.nproc:
            all_alive = True
            for p in self.workers:
                if not (p and p.is_alive()):
                    all_alive = False
            if all_alive:
                MyLogger.print_and_log('Found {} alive child processes, not generating new ones'.format(self.nproc),
                                       treebuilder_loc)
                return
        MyLogger.print_and_log('Tree builder spinning off {} child processes'.format(self.nproc), treebuilder_loc)
        for i in range(self.nproc):
            p = Process(target=self.work, args=(i,))
            # p.daemon = True
            self.workers.append(p)
            p.start()

    def get_ready_result(self):
        """
        Yields processed results from multiprocessing.

        Yields:
            list of 5-tuples of (int, string, int, list, float): Results
                from workers after applying a template to a molecule.
        """
        while not self.results_queue.empty():
            yield self.results_queue.get(timeout=0.5)

    def set_initial_target(self, _id, leaves):  # i = index of active pathway
        """
        Sets the first target.

        Expands given molecules with given templates.

        Args:
            _id (int): Unused; passed through.
            leaves (list of 2-tuples of (str, int)): Pairs of molecule
                SMILES and template IDs to be applied to them.
        """
        for leaf in leaves:
            if leaf in self.status:  # already being worked on
                continue
            chem_smi, template_idx = leaf
            self.expand(_id, chem_smi, template_idx)

    def stop(self, soft_stop=False):
        """
        Stops work with multiprocessing.

        Args:
            soft_stop (bool, optional): Whether to let active workers
                continue. (default: {false})
        """
        if not self.running:
            return
        # MyLogger.print_and_log('Terminating tree building process.', treebuilder_loc)
        if not soft_stop:
            self.done.value = 1
            for p in self.workers:
                if p and p.is_alive():
                    p.terminate()
        # MyLogger.print_and_log('All tree building processes done.', treebuilder_loc)
        self.running = False

    def wait_until_ready(self):
        """
        Wait until all workers are fully initialized and ready to being work.
        """
        while not all(self.initialized):
            MyLogger.print_and_log('Waiting for workers to initialize...', treebuilder_loc)
            time.sleep(5)

    def coordinate(self, soft_stop=False, known_bad_reactions=None, forbidden_molecules=None, return_first=False):
        """Coordinates workers.

        Args:
            soft_stop (bool, optional): Whether to use softly stop the workers.
                (default: {False})
            known_bad_reactions (list of str, optional): Reactions to
                eliminate from output. (default: {[]})
            forbidden_molecules (list of str, optional): SMILES strings of
                molecules to eliminate from output. (default: {[]})
            return_first (bool, optional): Whether to return after finding first
                pathway. (default: {False})
        """
        known_bad_reactions = known_bad_reactions or []
        forbidden_molecules = forbidden_molecules or []

        self.wait_until_ready()
        start_time = time.time()
        elapsed_time = time.time() - start_time
        next_log = 1
        MyLogger.print_and_log('Starting coordination loop', treebuilder_loc)
        while elapsed_time < self.expansion_time:

            if int(elapsed_time) // 5 == next_log:
                next_log += 1
                print("Worked for {}/{} s".format(int(elapsed_time * 10) / 10.0, self.expansion_time))
                print("... current min-price {}".format(self.Chemicals[self.smiles].price))
                print("... |C| = {} |R| = {}".format(len(self.Chemicals), len(self.status)))
                for _id in range(self.num_active_pathways):
                    print('Active pathway {}: {}'.format(_id, self.active_pathways[_id]))
                print('Active pathway pending? {}'.format(self.active_pathways_pending))

                # if self.celery:
                #     print('Pending results? {}'.format(len(self.pending_results)))
                # else:
                #     print('Expansion empty? {}'.format(self.expansion_queue.empty()))
                #     print('results_queue empty? {}'.format(self.results_queue.empty()))
                #     print('All idle? {}'.format(self.idle))

                # print(self.expansion_queue.qsize()) # TODO: make this Celery compatible
                # print(self.results_queue.qsize())

                # for _id in range(self.nproc):
                #   print(_id, self.expansion_queues[_id].qsize(), self.results_queues[_id].qsize())
                # time.sleep(2)

            for all_outcomes in self.get_ready_result():
                # Record that we've gotten a result for the _id of the active pathway
                _id = all_outcomes[0][0]
                # print('coord got outcomes for pathway ID {}'.format(_id))
                self.active_pathways_pending[_id] -= 1

                # Result of applying one template_idx to one chem_smi can be multiple eoutcomes
                for (_id, chem_smi, template_idx, reactants, filter_score) in all_outcomes:
                    # print('coord pulled {} result from result queue'.format(chem_smi))
                    self.status[(chem_smi, template_idx)] = DONE
                    chem = self.Chemicals[chem_smi]
                    cta = chem.template_idx_results[template_idx]  # TODO: make sure cta created
                    cta.waiting = False

                    # Any proposed reactants?
                    if len(reactants) == 0:
                        cta.valid = False # no precursors, reaction failed
                        # print('No reactants found for {} {}'.format(_id, chem_smi))
                        continue

                    # Get reactants SMILES
                    reactant_smiles = '.'.join([smi for (smi, _, _, _) in reactants])

                    # Banned reaction?
                    if '{}>>{}'.format(reactant_smiles, chem_smi) in known_bad_reactions:
                        cta.valid = False
                        continue

                    # Banned molecule?
                    if any(smi in forbidden_molecules for (smi, _, _, _) in reactants):
                        cta.valid = False
                        continue

                    matched_prev = False
                    for prev_tid, prev_cta in chem.template_idx_results.items():
                        if reactant_smiles in prev_cta.reactions:
                            prev_R = prev_cta.reactions[reactant_smiles]
                            matched_prev = True
                            # Now merge the two...
                            prev_R.tforms.append(template_idx)
                            prev_R.template_score = max(chem.prob[template_idx], prev_R.template_score)
                            cta.reactions[reactant_smiles] = prev_R
                            break
                    if matched_prev:
                        continue  # don't make a new reaction

                    # Define reaction using product SMILES, template_idx, and reactants SMILES
                    rxn = Reaction(chem_smi, template_idx)
                    rxn.plausibility = filter_score  # fast filter score
                    rxn.template_score = chem.prob[template_idx]  # template relevance
                    for (smi, top_probs, top_indices, value) in reactants:  # all precursors
                        rxn.reactant_smiles.append(smi)
                        if smi not in self.Chemicals:
                            self.Chemicals[smi] = Chemical(smi)
                            self.Chemicals[smi].set_template_relevance_probs(top_probs, top_indices, value)

                            ppg = self.pricer.lookup_smiles(smi, source=self.buyables_source, alreadyCanonical=True)
                            self.Chemicals[smi].purchase_price = ppg

                            hist = self.chemhistorian.lookup_smiles(smi, alreadyCanonical=True, template_set=self.template_set)
                            self.Chemicals[smi].as_reactant = hist['as_reactant']
                            self.Chemicals[smi].as_product = hist['as_product']

                            if self.is_a_terminal_node(smi, ppg, hist):
                                self.Chemicals[smi].set_price(1)  # all nodes treated the same for now
                                self.Chemicals[smi].terminal = True
                                self.Chemicals[smi].done = True
                                # print('TERMINAL: {}'.format(self.Chemicals[smi]))# DEBUG

                    rxn.estimate_price = sum([self.Chemicals[smi].estimate_price for smi in rxn.reactant_smiles])

                    # Add this reaction result to cta (key = reactant smiles)
                    cta.reactions[reactant_smiles] = rxn

            # See if this rollout is done (TODO: make this Celery compatible)
            for _id in range(self.num_active_pathways):
                if self.active_pathways_pending[_id] == 0: # this expansion step is done
                    # This expansion step is done = record!
                    self.update(self.smiles, self.active_pathways[_id])

                    # Set new target
                    leaves, pathway = self.select_leaf()
                    self.active_pathways[_id] = pathway
                    self.set_initial_target(_id, leaves)

            elapsed_time = time.time() - start_time

            if self.Chemicals[self.smiles].price != -1 and self.time_for_first_path == -1:
                self.time_for_first_path = elapsed_time
                MyLogger.print_and_log('Found the first pathway after {:.2f} seconds'.format(elapsed_time), treebuilder_loc)
                if return_first:
                    MyLogger.print_and_log('Stoping expansion to return first pathway as requested', treebuilder_loc)
                    break

            if all(pathway == {} for pathway in self.active_pathways) and len(self.pending_results) == 0:
                MyLogger.print_and_log('Cannot expand any further! Stuck?', treebuilder_loc)
                break

        self.stop(soft_stop=soft_stop)

        for _id in range(self.num_active_pathways):
            self.update(self.smiles, self.active_pathways[_id])

        self.active_pathways = [{} for _id in range(self.num_active_pathways)]

    def work(self, i):
        """
        Assigns work (if available) to given worker.

        Args:
            i (int): Index of worker to be assigned work.
        """
        # Need to load individual template prioritizer and fast filter models in each process
        template_prioritizer = self.load_template_prioritizer(self.template_prioritizer)
        fast_filter = self.load_fast_filter(self.fast_filter)

        self.initialized[i] = True

        while True:
            # If done, stop
            if self.done.value:
                # print 'Worker {} saw done signal, terminating'.format(i)
                break

            # Grab something off the queue
            if not self.expansion_queue.empty():
                try:
                    self.idle[i] = False
                    (_id, smiles, template_idx) = self.expansion_queue.get(timeout=0.1)  # short timeout

                    try:
                        # TODO: add settings
                        all_outcomes = self.retroTransformer.apply_one_template_by_idx(
                            _id, smiles, template_idx,
                            template_prioritizer=template_prioritizer,
                            fast_filter=fast_filter,
                            template_set=self.template_set
                        )
                    except Exception as e:
                        print(e)
                        all_outcomes = [(_id, smiles, template_idx, [], 0.0)]
                    self.results_queue.put(all_outcomes)
                except VanillaQueue.Empty:
                    self.idle[i] = True
                    pass   # looks like someone got there first...

            self.idle[i] = True

    def UCB(self, chem_smi, c_exploration=1., path=None):
        """
        Finds best reaction for a given chemical. Variation of upper confidence
        bound for trees.

        Can either select an unapplied template to apply, or select a specific
        reactant to expand further (?)
        TODO: check these changes...

        Args:
            chem_smi (str): SMILES string of target chemical.
            c_exploration (float, optional): weight for exploration. (default: {1.})
            path (list or dict, optional): Current reaction path. (default: {[]})

        Returns:
            2-tuple of (int, str): Index of template and SMILES strings of
                reactants corresponding to the highest scoring reaction
                resulting in the target product.
        """
        path = path or []
        rxn_scores = []

        chem = self.Chemicals[chem_smi]
        product_visits = chem.visit_count
        max_estimate_price = 0

        for template_idx in chem.template_idx_results:
            cta = chem.template_idx_results[template_idx]
            if cta.waiting or not cta.valid:
                continue

            for reactants_smi in cta.reactions:
                rxn = cta.reactions[reactants_smi]

                if len(set(rxn.reactant_smiles) & set(path)) > 0: # avoid cycles
                    continue
                if rxn.done:
                    continue
                max_estimate_price = max(max_estimate_price, rxn.estimate_price)
                Q_sa = - rxn.estimate_price
                try:
                    U_sa = c_exploration * chem.prob[template_idx] * np.sqrt(product_visits) / (1 + rxn.visit_count)
                except:
                    print(chem_smi, product_visits)
                score = Q_sa + U_sa
                rxn_scores.append((score, template_idx, reactants_smi))

        # unexpanded template - find most relevant template that hasn't been tried
        num_branches = len(rxn_scores)
        if num_branches < self.max_branching or chem_smi == self.smiles:
            for template_idx in chem.top_indeces:
                if template_idx not in chem.template_idx_results:
                    Q_sa = - (max_estimate_price + 0.1)
                    U_sa = c_exploration * chem.prob[template_idx] * np.sqrt(product_visits) / 1
                    score = Q_sa + U_sa
                    # record estimated score if we were to actually apply that template
                    rxn_scores.append((score, template_idx, None))
                    # TODO: figure out if this "None" makes sense for the reactants smiles
                    break

        if len(rxn_scores) > 0:
            sorted_rxn_scores = sorted(rxn_scores, key=lambda x: x[0], reverse=True)
            # get next best template to apply
            best_rxn_score, selected_template_idx, selected_reactants_smi = sorted_rxn_scores[0]
        else:
            selected_template_idx, selected_reactants_smi = None, None

        return selected_template_idx, selected_reactants_smi

    def select_leaf(self, c_exploration=1.):
        """
        Select leaf to be simulated next.

        Args:
            c_exploration (float, optional): weight for exploration passed to
                UCB (default: {1.0})

        Returns:
            2-tuple of (list of 2-tuples of (str, int), dict): SMILES strings of
                chemical and corresponding template index and pathways from a
                chemical to its reactants??
        """
        pathway = {}
        leaves = []
        queue = VanillaQueue.Queue()
        queue.put((self.smiles, 0, [self.smiles]))

        while not queue.empty():
            chem_smi, depth, path = queue.get()
            if depth >= self.max_depth or chem_smi in pathway:  # don't go too deep or recursively
                continue
            template_idx, reactants_smi = self.UCB(chem_smi, c_exploration=c_exploration, path=path)
            if template_idx is None:
                continue

            # Only grow pathway when we have picked a specific reactants_smi (?)
            if reactants_smi is not None:
                pathway[chem_smi] = (template_idx, reactants_smi)
            else:
                # TODO: figure out if reactants_smi==None case is an issue
                pathway[chem_smi] = template_idx  # still record template selection

            chem = self.Chemicals[chem_smi]
            chem.visit_count += VIRTUAL_LOSS

            if template_idx not in chem.template_idx_results:
                chem.template_idx_results[template_idx] = ChemicalTemplateApplication(chem_smi, template_idx)
                leaves.append((chem_smi, template_idx))
            else:
                # Can we assume that the reactants_smi exists in this cta? I guess so...
                cta = chem.template_idx_results[template_idx]

                if reactants_smi:  # if we choose a specific reaction, not just a template...
                    if reactants_smi in cta.reactions:

                        rxn = cta.reactions[reactants_smi]
                        rxn.visit_count += VIRTUAL_LOSS

                        for smi in rxn.reactant_smiles:
                            assert smi in self.Chemicals
                            if not self.Chemicals[smi].done:
                                queue.put((smi, depth+1, path+[smi]))
                        if rxn.done:
                            chem.visit_count += rxn.visit_count
                            rxn.visit_count += rxn.visit_count

        return leaves, pathway

    def update(self, chem_smi, pathway, depth=0):
        """
        Backpropagate results of simulation up the tree.

        Args:
            chem_smi (str): SMILES string of
            pathway (dict):
            depth (int, optional): (default: {0})
        """
        if depth == 0:
            for smi in pathway:
                if type(pathway[smi]) == tuple:
                    (template_idx, reactants_smi) = pathway[smi]
                else:
                    (template_idx, reactants_smi) = (pathway[smi], None)
                chem = self.Chemicals[smi]
                cta = chem.template_idx_results[template_idx]
                chem.visit_count -= (VIRTUAL_LOSS - 1)
                if reactants_smi:
                    rxn = cta.reactions[reactants_smi]
                    rxn.visit_count -= (VIRTUAL_LOSS - 1)

        if (chem_smi not in pathway) or (depth >= self.max_depth):
            return

        if type(pathway[chem_smi]) == tuple:
            (template_idx, reactants_smi) = pathway[chem_smi]
        else:
            (template_idx, reactants_smi) = (pathway[chem_smi], None)

        chem = self.Chemicals[chem_smi]
        cta = chem.template_idx_results[template_idx]
        if cta.waiting:  # haven't actually expanded
            return

        if reactants_smi:
            rxn = cta.reactions[reactants_smi]
            if rxn.valid and (not rxn.done):
                rxn.done = all([self.Chemicals[smi].done for smi in rxn.reactant_smiles])

                for smi in rxn.reactant_smiles:
                    self.update(smi, pathway, depth+1)

                estimate_price = sum([self.Chemicals[smi].estimate_price for smi in rxn.reactant_smiles])
                rxn.update_estimate_price(estimate_price)
                chem.update_estimate_price(estimate_price)

                price_list = [self.Chemicals[smi].price for smi in rxn.reactant_smiles]
                if all([price != -1 for price in price_list]):
                    price = sum(price_list)
                    rxn.price = price
                    if rxn.price < chem.price or chem.price == -1:
                        chem.price = rxn.price

        if sum(len(cta.reactions) for tid, cta in chem.template_idx_results.items()) >= self.max_branching:
            # print('{} hit max branching, checking if "done"'.format(chem_smi))
            chem.done = all([(rxn.done or (not rxn.valid)) for tid, cta in chem.template_idx_results.items() for rsmi, rxn in cta.reactions.items()])

        # if chem.price != -1 and chem.price < chem.estimate_price:
        #   chem.estimate_price = chem.price

    def full_update(self, chem_smi, depth=0, path=None):
        """??

        Args:
            chem_smi (str): SMILES string of target chemical.
            depth (int, optional): (default: {0})
            path (list?): (default: {[]})
        """
        path = path or []
        chem = self.Chemicals[chem_smi]
        chem.pathway_count = 0

        if chem.terminal:
            chem.pathway_count = 1
            return

        if depth > self.max_depth:
            return

        for template_idx in chem.template_idx_results:
            cta = chem.template_idx_results[template_idx]
            for reactants_smi in cta.reactions:
                rxn = cta.reactions[reactants_smi]
                rxn.pathway_count = 0
                if (not rxn.valid) or len(set(rxn.reactant_smiles) & set(path)) > 0:
                    continue
                for smi in rxn.reactant_smiles:
                    self.full_update(smi, depth+1, path+[chem_smi])
                price_list = [self.Chemicals[smi].price for smi in rxn.reactant_smiles]
                if all([price != -1 for price in price_list]):
                    price = sum(price_list)
                    rxn.price = price
                    if rxn.price < chem.price or chem.price == -1:
                        chem.price = rxn.price
                        chem.best_template = template_idx
                    rxn.pathway_count = np.prod([self.Chemicals[smi].pathway_count for smi in rxn.reactant_smiles])
                else:
                    rxn.pathway_count = 0

        chem.pathway_count = 0
        for tid,cta in chem.template_idx_results.items():
            for rct_smi,rxn in cta.reactions.items():
                chem.pathway_count += rxn.pathway_count

    def build_tree(self, soft_stop=False, known_bad_reactions=None, forbidden_molecules=None, return_first=False):
        """Builds retrosynthesis tree.

        Args:
            soft_stop (bool, optional): Whether to use softly stop the workers.
                (default: {False})
            known_bad_reactions (list of str, optional): Reactions to
                eliminate from output. (default: {[]})
            forbidden_molecules (list of str, optional): SMILES strings of
                molecules to eliminate from output. (default: {[]})
            return_first (bool, optional): Whether to return after finding first
                pathway. (default: {False})
        """
        known_bad_reactions = known_bad_reactions or []
        forbidden_molecules = forbidden_molecules or []
        self.running = True

        with self.allow_join_result():

            MyLogger.print_and_log('Preparing workers...', treebuilder_loc)
            self.prepare()

            # Define first chemical node (target)
            probs, indices = self.get_initial_prioritization()
            value = 1  # current value assigned to precursor (note: may replace with real value function)
            self.Chemicals[self.smiles] = Chemical(self.smiles)
            self.Chemicals[self.smiles].set_template_relevance_probs(probs, indices, value)
            MyLogger.print_and_log('Calculating initial probs for target', treebuilder_loc)
            hist = self.chemhistorian.lookup_smiles(self.smiles, alreadyCanonical=False)
            self.Chemicals[self.smiles].as_reactant = hist['as_reactant']
            self.Chemicals[self.smiles].as_product = hist['as_product']
            ppg = self.pricer.lookup_smiles(self.smiles, source=self.buyables_source, alreadyCanonical=False)
            self.Chemicals[self.smiles].purchase_price = ppg

            # First selection is all the same
            leaves, pathway = self.select_leaf()
            for _id in range(self.num_active_pathways):
                self.active_pathways[_id] = pathway
                self.set_initial_target(_id, leaves)
            MyLogger.print_and_log('Set initial leaves for active pathways', treebuilder_loc)

            # Coordinate workers.
            self.coordinate(
                soft_stop=soft_stop,
                known_bad_reactions=known_bad_reactions,
                forbidden_molecules=forbidden_molecules,
                return_first=return_first
            )

            # Do a final pass to get counts
            MyLogger.print_and_log('Doing final update of pathway counts / prices', treebuilder_loc)
            self.full_update(self.smiles)
            chem = self.Chemicals[self.smiles]

        print("Finished working.")
        print("=== found %d pathways (overcounting duplicate templates)" % chem.pathway_count)
        print("=== time for fist pathway: %.2fs" % self.time_for_first_path)
        print("=== min price: %.1f" % chem.price)
        print("---------------------------")

    def get_initial_prioritization(self):
        """
        Instantiate a template prioritization model and get predictions to
        initialize the tree search.
        """
        template_prioritizer = self.load_template_prioritizer(self.template_prioritizer)
        scores, indices = template_prioritizer.predict(self.smiles, self.template_count, self.max_cum_template_prob)
        return scores, indices

    # QUESTION: Why return the empty list?
    def tree_status(self):
        """Summarize size of tree after expansion.

        Returns:
            (int, int):
                num_chemicals (int): Number of chemical nodes in the tree.
                num_reactions (int): Number of reaction nodes in the tree.
        """
        num_chemicals = len(self.Chemicals)
        num_reactions = len(self.status)
        return num_chemicals, num_reactions, []

    def tid_list_to_info_dict(self, tids):
        """
        Returns dict of info from a given list of templates.

            Args:
                tids (list of int): Template IDs to get info about.
        """
        templates = [self.retroTransformer.templates[tid] for tid in tids]

        return {
            'tforms': [str(t.get('_id', -1)) for t in templates],
            'num_examples': int(sum([t.get('count', 1) for t in templates])),
            'necessary_reagent': templates[0].get('necessary_reagent', ''),
        }

    def return_trees(self):
        """Returns retrosynthetic pathways trees and their size."""

        def chem_info_dict(smi):
            """
            Returns dict of extra info about a given chemical.

            Args:
                smi (str): SMILES string of chemical.
            """
            return {
                'smiles': smi,
                'ppg': self.Chemicals[smi].purchase_price,
                'as_reactant': self.Chemicals[smi].as_reactant,
                'as_product': self.Chemicals[smi].as_product,
            }

        seen_rxnsmiles = {}
        self.current_index = 1

        def rxnsmiles_to_id(smi):
            if smi not in seen_rxnsmiles:
                seen_rxnsmiles[smi] = self.current_index
                self.current_index += 1
            return seen_rxnsmiles[smi]

        seen_chemsmiles = {}

        def chemsmiles_to_id(smi):
            if smi not in seen_chemsmiles:
                seen_chemsmiles[smi] = self.current_index
                self.current_index += 1
            return seen_chemsmiles[smi]

        def IDDFS():
            """
            Performs iterative depth-first search to find buyable pathways.

            Yields:
                dict: nested dictionaries defining synthesis trees
            """
            print(len(self.Reactions.keys()))
            for path in DLS_chem(self.smiles, depth=0, headNode=True):
                yield chem_dict(chemsmiles_to_id(self.smiles), children=path, **chem_info_dict(self.smiles))

        def DLS_chem(chem_smi, depth, headNode=False):
            """
            Expands at a fixed depth for the current node ``chem_id``.

            Args:
                chem_smi (str): SMILES string for given chemical.
                depth (int): Depth node is expanded at.
                headNode (bool, optional): Unused. (default: {False})
            """
            chem = self.Chemicals[chem_smi]
            if chem.terminal:
                yield []
            if depth > self.max_depth:
                return

            done_children_of_this_chemical = []

            # if depth > self.max_depth:
            #     return
            for tid, cta in chem.template_idx_results.items():
                ########??????????????????????????######################
                if cta.waiting:
                    continue
                for rct_smi, rxn in cta.reactions.items():
                    if (not rxn.valid) or rxn.price == -1:
                        continue
                    rxn_smiles = '.'.join(sorted(rxn.reactant_smiles)) + '>>' + chem_smi
                    if rxn_smiles not in done_children_of_this_chemical:  # necessary to avoid duplicates
                        for path in DLS_rxn(chem_smi, tid, rct_smi, depth):
                            yield [rxn_dict(rxnsmiles_to_id(rxn_smiles), rxn_smiles, children=path,
                                            plausibility=rxn.plausibility, template_score=rxn.template_score,
                                            **self.tid_list_to_info_dict(rxn.tforms))]
                        done_children_of_this_chemical.append(rxn_smiles)

        def DLS_rxn(chem_smi, template_idx, rct_smi, depth):
            """
            Yields children paths starting from a specific ``rxn_id``.

            Args:
                chem_smi (str): SMILES string for given chemical.
                template_idx (int): Index of given template.
                rct_smi (str): SMILES string for reactants.
                depth (int): depth of given reaction.
            """
            # TODO: add in auxiliary information about templates, etc.
            rxn = self.Chemicals[chem_smi].template_idx_results[template_idx].reactions[rct_smi]

            # rxn_list = []
            # for smi in rxn.reactant_smiles:
            #     rxn_list.append([chem_dict(smi, children=path, **{}) for path in DLS_chem(smi, depth+1)])

            # return [rxns[0] for rxns in itertools.product(rxn_list)]

            ###################
            # To get recursion working properly with generators, need to hard-code these cases? Unclear
            # whether itertools.product can actually work with generators. Seems like it can't work
            # well...

            # Only one reactant? easy!
            if len(rxn.reactant_smiles) == 1:
                chem_smi0 = rxn.reactant_smiles[0]
                for path in DLS_chem(chem_smi0, depth + 1):
                    yield [
                        chem_dict(chemsmiles_to_id(chem_smi0), children=path, **chem_info_dict(chem_smi0))
                    ]

            # Two reactants? want to capture all combinations of each node's
            # options
            elif len(rxn.reactant_smiles) == 2:
                chem_smi0 = rxn.reactant_smiles[0]
                chem_smi1 = rxn.reactant_smiles[1]
                for path0 in DLS_chem(chem_smi0, depth + 1):
                    for path1 in DLS_chem(chem_smi1, depth + 1):
                        yield [
                            chem_dict(chemsmiles_to_id(chem_smi0), children=path0, **chem_info_dict(chem_smi0)),
                            chem_dict(chemsmiles_to_id(chem_smi1), children=path1, **chem_info_dict(chem_smi1)),
                        ]

            # Three reactants? This is not elegant...
            elif len(rxn.reactant_smiles) == 3:
                chem_smi0 = rxn.reactant_smiles[0]
                chem_smi1 = rxn.reactant_smiles[1]
                chem_smi2 = rxn.reactant_smiles[2]
                for path0 in DLS_chem(chem_smi0, depth + 1):
                    for path1 in DLS_chem(chem_smi1, depth + 1):
                        for path2 in DLS_chem(chem_smi2, depth + 1):
                            yield [
                                chem_dict(chemsmiles_to_id(chem_smi0), children=path0, **chem_info_dict(chem_smi0)),
                                chem_dict(chemsmiles_to_id(chem_smi1), children=path1, **chem_info_dict(chem_smi1)),
                                chem_dict(chemsmiles_to_id(chem_smi2), children=path2, **chem_info_dict(chem_smi2)),
                            ]

            # I am ashamed
            elif len(rxn.reactant_smiles) == 4:
                chem_smi0 = rxn.reactant_smiles[0]
                chem_smi1 = rxn.reactant_smiles[1]
                chem_smi2 = rxn.reactant_smiles[2]
                chem_smi3 = rxn.reactant_smiles[3]
                for path0 in DLS_chem(chem_smi0, depth + 1):
                    for path1 in DLS_chem(chem_smi1, depth + 1):
                        for path2 in DLS_chem(chem_smi2, depth + 1):
                            for path3 in DLS_chem(chem_smi3, depth + 1):
                                yield [
                                    chem_dict(chemsmiles_to_id(chem_smi0), children=path0, **chem_info_dict(chem_smi0)),
                                    chem_dict(chemsmiles_to_id(chem_smi1), children=path1, **chem_info_dict(chem_smi1)),
                                    chem_dict(chemsmiles_to_id(chem_smi2), children=path2, **chem_info_dict(chem_smi2)),
                                    chem_dict(chemsmiles_to_id(chem_smi3), children=path3, **chem_info_dict(chem_smi3)),
                                ]

            else:
                print('Too many reactants! Only have cases 1-4 programmed')
                print('There probably are not any real 5 component reactions')
                print(rxn.reactant_smiles)

        MyLogger.print_and_log('Retrieving trees...', treebuilder_loc)
        trees = []
        for tree in IDDFS():
            trees.append(tree)
            if len(trees) >= self.max_trees:
                break

        # Sort by some metric
        def number_of_starting_materials(tree):
            # TODO: Is `tree` a list or dict?!?
            if tree != []:
                if tree['children']:
                    return sum(
                        number_of_starting_materials(tree_child) for tree_child in tree['children'][0]['children'])
            return 1.0

        def number_of_reactions(tree):
            if tree != []:
                if tree['children']:
                    return 1.0 + max(number_of_reactions(tree_child) for tree_child in tree['children'][0]['children'])
            return 0.0

        def overall_plausibility(tree):
            if tree != []:
                if tree['children']:
                    producing_reaction = tree['children'][0]
                    return producing_reaction['plausibility'] * np.prod(
                        [overall_plausibility(tree_child) for tree_child in producing_reaction['children']])
            return 1.0

        MyLogger.print_and_log('Sorting {} trees...'.format(len(trees)), treebuilder_loc)
        if self.sort_trees_by == 'plausibility':
            trees = sorted(trees, key=lambda x: overall_plausibility(x), reverse=True)
        elif self.sort_trees_by == 'number_of_starting_materials':
            trees = sorted(trees, key=lambda x: number_of_starting_materials(x))
        elif self.sort_trees_by == 'number_of_reactions':
            trees = sorted(trees, key=lambda x: number_of_reactions(x))
        else:
            raise ValueError('Need something to sort by! Invalid option provided {}'.format(self.sort_trees_by))

        return self.tree_status(), trees

    # TODO: use these settings...
    def get_buyable_paths(self,
                          smiles,
                          max_depth=10,
                          max_branching=25,
                          expansion_time=30,
                          nproc=12,
                          num_active_pathways=None,
                          max_trees=5000,
                          max_ppg=1e10,
                          known_bad_reactions=None,
                          forbidden_molecules=None,
                          template_count=100,
                          max_cum_template_prob=0.995,
                          max_scscore=None,
                          max_elements=None,
                          min_history=None,
                          termination_logic=None,
                          apply_fast_filter=True,
                          filter_threshold=0.75,
                          soft_reset=False,
                          return_first=False,
                          sort_trees_by='plausibility',
                          template_prioritizer='reaxys',
                          template_set='reaxys',
                          buyables_source=None,
                          **kwargs):
        """Returns trees with path ending in buyable chemicals.

        Args:
            smiles (str): SMILES string of target molecule.
            max_depth (int, optional): Maximum number of reactions to allow
                before stopping the recursive expansion down one branch.
                (default: {10})
            max_branching (int, optional): Maximum number of precursor
                suggestions to add to the tree at each expansion.
                (default: {25})
            expansion_time (int, optional): Time (in seconds) to allow for
                expansion before searching the generated tree for buyable
                pathways. (default: {30})
            nproc (int, optional): Number of retrotransformer processes to fork
                for faster expansion. (default: {12})
            num_active_pathways (int or None, optional): Number of active
                pathways. (default: {None})
            max_trees (int, optional): Maximum number of trees to return.
                (default: {5000})
            max_ppg (int, optional): Maximum price per gram of any chemical
                in a valid path. (default: {1e10})
            known_bad_reactions (list of str, optional): Reactions to eliminate
                from output. (default: {[]})
            forbidden_molecules (list of str, optional): SMILES strings of
                molecules to eliminate from output. (default: {[]})
            template_count (int, optional): Maximum number of relevant templates
                to consider. (default: {100})
            max_cum_template_prob (float, optional): Maximum cumulative
                probability of selected relevant templates. (default: {0.995})
            max_scscore (int, optional): Maximum synthetic complexity score for
                a chemical to be considered terminal. (default: {None})
            max_elements (dict, optional): Maximum number of certain elements
                for a chemical to be considered terminal. (default: {None})
            min_history (dict, optional): Minimum number of prior appearances
                as a reactant or product for a chemical to be considered
                terminal. (default: {None})
            termination_logic (dict, optional): Defines logic to be used when
                determining if a chemical is terminal. Should contain two keys,
                ['and', 'or'], and list of criteria as values. Supported
                criteria: ['buyable', 'max_ppg', 'max_scscore', 'max_elements',
                'max_history']. (default: {None})
            apply_fast_filter (bool, optional): Whether to apply the fast
                filter. (default: {True})
            filter_threshold (float, optional): Threshold to use for the fast
                filter. (default: {0.75})
            soft_reset (bool, optional): Whether to softly reset workers.
                (default: {False})
            return_first (bool, optional): Whether to return after finding first
                pathway. (default: {False})
            sort_trees_by (str, optional): Criteria used to sort trees.
                (default: {'plausibility'})
            template_prioritizer (str, optional): Specifies which template
                prioritizer to use. (default: {'reaxys'})
            template_set (str, optional): Specifies which template set to use.
                (default: {'reaxys'})
            buyables_source (str or list, optional): Specifies source of buyables
                data to use. Will use all available data if not provided.
            **kwargs: Additional optional arguments.

        Returns:
            ((int, int, dict), list of dict):
                tree_status ((int, int, dict)): Result of tree_status().

                trees (list of dict): List of dictionaries, where each dictionary
                    defines a synthetic route.
        """
        self.smiles = smiles
        self.max_depth = max_depth
        self.max_branching = max_branching
        self.expansion_time = expansion_time
        self.nproc = nproc
        self.num_active_pathways = num_active_pathways or self.nproc
        self.max_trees = max_trees
        self.max_cum_template_prob = max_cum_template_prob
        self.template_count = template_count
        self.filter_threshold = filter_threshold
        self.apply_fast_filter = apply_fast_filter

        self.max_ppg = max_ppg
        self.max_scscore = max_scscore
        self.max_elements = max_elements
        self.min_history = min_history
        self.termination_logic = termination_logic or {}
        self.buyables_source = buyables_source

        self.sort_trees_by = sort_trees_by
        self.template_prioritizer = template_prioritizer
        self.template_set = template_set

        known_bad_reactions = known_bad_reactions or []
        forbidden_molecules = forbidden_molecules or []

        MyLogger.print_and_log('Active pathway #: {}'.format(num_active_pathways), treebuilder_loc)

        if (self.chemhistorian is None and self.min_history is not None
                and ('min_history' in self.termination_logic['and'] or 'min_history' in self.termination_logic['or'])):
            self.chemhistorian = self.load_chemhistorian(kwargs.get('use_db', False))

        self.reset(soft_reset=soft_reset)

        MyLogger.print_and_log('Starting search for {}'.format(smiles), treebuilder_loc)
        self.build_tree(soft_stop=kwargs.pop('soft_stop', False),
                        known_bad_reactions=known_bad_reactions,
                        forbidden_molecules=forbidden_molecules,
                        return_first=return_first,
                        )

        return self.return_trees()

    def is_a_terminal_node(self, smiles, ppg, hist):
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

    def return_chemical_results(self):
        results = defaultdict(list)
        for chemical in self.Chemicals.values():
            if not chemical.template_idx_results:
                results[chemical.smiles]
            for cta in chemical.template_idx_results.values():
                for res in cta.reactions.values():
                    reaction = vars(res)
                    reaction['pathway_count'] = int(reaction['pathway_count'])
                    results[chemical.smiles].append(reaction)
        return dict(results)


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--simulation-time', default=30)
    parser.add_argument('-n', '--num-processes', default=4)
    args = parser.parse_args()

    random.seed(1)
    np.random.seed(1)
    MyLogger.initialize_logFile()
    simulation_time = int(args.simulation_time)

    # Load tree builder
    n_procs = int(args.num_processes)
    print("There are {} processes available ... ".format(n_procs))
    tree = MCTS(nproc=n_procs)

    ####################################################################################
    ############################# SCOPOLAMINE TEST #####################################
    ####################################################################################

    smiles = 'Cc1ncc([N+](=O)[O-])n1CC(C)O'
    import rdkit.Chem as Chem
    smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles), True)
    status, paths = tree.get_buyable_paths(smiles,
                                           nproc=n_procs,
                                           expansion_time=30,
                                           max_cum_template_prob=0.995,
                                           template_count=100,
                                           # min_history={'as_reactant':5, 'as_product':5,'logic':'none'},
                                           soft_reset=False,
                                           soft_stop=False)
    print(status)
    for path in paths[:5]:
        print(path)
    print('Total num paths: {}'.format(len(paths)))
