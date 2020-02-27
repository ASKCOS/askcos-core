import os
import queue as VanillaQueue
import time
from collections import defaultdict
from multiprocessing import Process, Manager, Queue

import numpy as np
import rdkit.Chem as Chem
from pymongo import MongoClient

import makeit.global_config as gc
from makeit.retrosynthetic.mcts.nodes import Chemical, Reaction, ChemicalTemplateApplication
from makeit.retrosynthetic.transformer import RetroTransformer
from makeit.utilities.buyable.pricer import Pricer
from makeit.utilities.formats import chem_dict, rxn_dict
from makeit.utilities.io import model_loader
from makeit.utilities.io.logger import MyLogger
from makeit.utilities.historian.chemicals import ChemHistorian

import askcos_site.askcos_celery.treebuilder.tb_c_worker as tb_c_worker

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
    instances can be passed during initiation. The template prioritizer
    should be included as part of the RetroTransformer.

    Attributes:

    """

    def __init__(self, retroTransformer=None, pricer=None, max_branching=20, max_depth=3, expansion_time=60,
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
        """

        if 'chiral' in kwargs and not kwargs['chiral']:
            raise ValueError('MCTS only works for chiral expansion!')

        self.max_depth = max_depth
        self.max_branching = max_branching
        self.expansion_time = expansion_time
        self.nproc = nproc
        self.max_cum_template_prob = 1
        self.sort_trees_by = 'plausibility'

        self.num_active_pathways = num_active_pathways or self.nproc

        # Load data and models
        self.pricer = pricer or self.load_pricer()
        self.chemhistorian = chemhistorian or self.load_chemhistorian()
        self.retroTransformer = retroTransformer or self.load_retro_transformer(
            template_set=template_set,
            template_prioritizer=template_prioritizer,
            precursor_prioritizer=precursor_prioritizer,
            fast_filter=fast_filter
        )

        # Initialize vars, reset dicts, etc.
        self.reset(soft_reset=False)

        self.status = {}

        from makeit.utilities.with_dummy import with_dummy
        self.allow_join_result = with_dummy

    @staticmethod
    def load_pricer():
        """
        Loads pricer.
        """
        pricer = Pricer(use_db=False)
        pricer.load()
        return pricer

    @staticmethod
    def load_chemhistorian():
        """
        Loads chemhistorian.
        """
        chemhistorian = ChemHistorian(use_db=False)
        chemhistorian.load()
        return chemhistorian

    @staticmethod
    def load_retro_transformer(template_set='reaxys', template_prioritizer='reaxys',
                               precursor_prioritizer='relevanceheuristic', fast_filter='default'):
        """
        Loads retro transformer model.
        """
        retro_transformer = RetroTransformer(
            template_set=template_set,
            template_prioritizer=template_prioritizer,
            precursor_prioritizer=precursor_prioritizer,
            fast_filter=fast_filter
        )
        retro_transformer.load()
        return retro_transformer

    def reset(self, soft_reset=False):
        """
        Prepare for a new expansion

        TODO: add "soft" feature which does not spin up new workers

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
        """Adds pathway to be worked on with multiprocessing.

        Args:
            _id (int): ID of pending pathway.
            smiles (str): SMILES string of molecule to be exanded.
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
        """Yields processed results from multiprocessing.

        Yields:
            list of 5-tuples of (int, string, int, list, float): Results
                from workers after applying a template to a molecule.
        """
        while not self.results_queue.empty():
            yield self.results_queue.get(timeout=0.5)

    def set_initial_target(self, _id, leaves):  # i = index of active pathway
        """Sets the first target.

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
        """Stops work with multiprocessing.

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

    def coordinate(self, soft_stop=False, known_bad_reactions=[], forbidden_molecules=[], return_first=False):
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
        # if not self.celery:
        #     while not all(self.initialized):
        #         MyLogger.print_and_log('Waiting for workers to initialize...', treebuilder_loc)
        #         time.sleep(2)
        start_time = time.time()
        elapsed_time = time.time() - start_time
        next = 1
        MyLogger.print_and_log('Starting cooridnation loop', treebuilder_loc)
        while (elapsed_time < self.expansion_time): # and self.waiting_for_results():

            if (int(elapsed_time)//5 == next):
                next += 1
                print ("Worked for {}/{} s".format(int(elapsed_time*10)/10.0, self.expansion_time))
                print ("... current min-price {}".format(self.Chemicals[self.smiles].price))
                print ("... |C| = {} |R| = {}".format(len(self.Chemicals), len(self.status)))
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
                    # R = self.Chemicals[chem_smi].reactions[template_idx]
                    C = self.Chemicals[chem_smi]
                    CTA = C.template_idx_results[template_idx] # TODO: make sure CTA created
                    CTA.waiting = False

                    # Any proposed reactants?
                    if len(reactants) == 0:
                        CTA.valid = False # no precursors, reaction failed
                        # print('No reactants found for {} {}'.format(_id, chem_smi))
                        continue

                    # Get reactants SMILES
                    reactant_smiles = '.'.join([smi for (smi, _, _, _) in reactants])

                    # Banned reaction?
                    if '{}>>{}'.format(reactant_smiles, chem_smi) in known_bad_reactions:
                        CTA.valid = False
                        continue

                    # Banned molecule?
                    if any(smi in forbidden_molecules for (smi, _, _, _) in reactants):
                        CTA.valid = False
                        continue

                    # TODO: check if banned reaction
                    matched_prev = False
                    for prev_tid, prev_cta in C.template_idx_results.items():
                        if reactant_smiles in prev_cta.reactions:
                            prev_R = prev_cta.reactions[reactant_smiles]
                            matched_prev = True
                            # Now merge the two...
                            prev_R.tforms.append(template_idx)
                            prev_R.template_score = max(C.prob[template_idx], prev_R.template_score)
                            CTA.reactions[reactant_smiles] = prev_R
                            break
                    if matched_prev:
                        continue # don't make a new reaction

                    # Define reaction using product SMILES, template_idx, and reactants SMILES
                    R = Reaction(chem_smi, template_idx)
                    R.plausibility = filter_score # fast filter score
                    R.template_score = C.prob[template_idx] # template relevance
                    #for smi, prob, value in reactants:
                    for (smi, top_probs, top_indeces, value) in reactants: # all precursors
                        R.reactant_smiles.append(smi)
                        if smi not in self.Chemicals:
                            self.Chemicals[smi] = Chemical(smi)
                            self.Chemicals[smi].set_template_relevance_probs(top_probs, top_indeces, value)

                            ppg = self.pricer.lookup_smiles(smi, alreadyCanonical=True)
                            self.Chemicals[smi].purchase_price = ppg
                            # if ppg is not None and ppg > 0:
                            #     self.Chemicals[smi].set_price(ppg)

                            hist = self.chemhistorian.lookup_smiles(smi, alreadyCanonical=True)
                            self.Chemicals[smi].as_reactant = hist['as_reactant']
                            self.Chemicals[smi].as_product = hist['as_product']

                            if self.is_a_terminal_node(smi, ppg, hist):
                                self.Chemicals[smi].set_price(1) # all nodes treated the same for now
                                self.Chemicals[smi].terminal = True
                                self.Chemicals[smi].done = True
                                # print('TERMINAL: {}'.format(self.Chemicals[smi]))# DEBUG

                    R.estimate_price = sum([self.Chemicals[smi].estimate_price for smi in R.reactant_smiles])

                    # Add this reaction result to CTA (key = reactant smiles)
                    CTA.reactions[reactant_smiles] = R

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

            if  all(pathway == {} for pathway in self.active_pathways) and len(self.pending_results) == 0:
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
                        all_outcomes = self.retroTransformer.apply_one_template_by_idx(_id, smiles, template_idx)
                    except Exception as e:
                        print(e)
                        all_outcomes = [(_id, smiles, template_idx, [], 0.0)]
                    self.results_queue.put(all_outcomes)
                except VanillaQueue.Empty:
                    self.idle[i] = True
                    pass   # looks like someone got there first...

            self.idle[i] = True

    def UCB(self, chem_smi, c_exploration=0.2, path=[]):
        """Finds best reaction for a given chemical.

        Can either select an unapplied template to apply, or select a specific
        reactant to expand further (?)
        TODO: check these changes...

        Args:
            chem_smi (str): SMILES string of target chemical.
            c_exploration (float, optional): ?? Used to calculate reaction
                score. (default: {0.2})
            path (list or dict, optional): Current reation path. (default: {[]})

        Returns:
            2-tuple of (int, str): Index of template and SMILES strings of
                reactants corresponding to the highest scoring reaction
                resulting in the target product.
        """
        rxn_scores = []

        C = self.Chemicals[chem_smi]
        product_visits = C.visit_count
        max_estimate_price = 0

        for template_idx in C.template_idx_results:
            CTA = C.template_idx_results[template_idx]
            if CTA.waiting or not CTA.valid:
                continue

            for reactants_smi in CTA.reactions:
                R = CTA.reactions[reactants_smi]

                if len(set(R.reactant_smiles) & set(path)) > 0: # avoid cycles
                    continue
                if R.done:
                    continue
                max_estimate_price = max(max_estimate_price, R.estimate_price)
                Q_sa = - R.estimate_price
                try:
                    U_sa = c_exploration * C.prob[template_idx] * np.sqrt(product_visits) / (1 + R.visit_count)
                except:
                    print(chem_smi, product_visits)
                score = Q_sa + U_sa
                rxn_scores.append((score, template_idx, reactants_smi))

        # unexpanded template - find most relevant template that hasn't been tried
        num_branches = len(rxn_scores)
        if num_branches < self.max_branching or chem_smi == self.smiles:
            for template_idx in C.top_indeces:
                if template_idx not in C.template_idx_results:
                    Q_sa = - (max_estimate_price + 0.1)
                    U_sa = c_exploration * C.prob[template_idx] * np.sqrt(product_visits) / 1
                    score = Q_sa + U_sa
                    rxn_scores.append((score, template_idx, None)) # record estimated score if we were to actually apply that template
                    # TODO: figure out if this "None" makes sense for the reactants smiles
                    break

        if len(rxn_scores) > 0:
            sorted_rxn_scores = sorted(rxn_scores, key=lambda x: x[0], reverse=True)
            best_rxn_score, selected_template_idx, selected_reactants_smi = sorted_rxn_scores[0] # get next best template to apply
        else:
            selected_template_idx, selected_reactants_smi = None, None

        return selected_template_idx, selected_reactants_smi

    def select_leaf(self, c_exploration=1.):
        """Builds reaction pathways??

        Args:
            c_exploration (float, optional): ?? (default: {1.0})

        Returns:
            2-tuple of (list of 2-tuples of (str, int), dict): SMILES strings of
                chemical and corresponding template index and pathways from a
                chemical to its reactants??
        """
        #start_time = time.time()
        pathway = {}
        leaves = []
        queue = VanillaQueue.Queue()
        queue.put((self.smiles, 0, [self.smiles]))

        while not queue.empty():
            chem_smi, depth, path = queue.get()
            if depth >= self.max_depth or chem_smi in pathway: # don't go too deep or recursively
                continue
            template_idx, reactants_smi = self.UCB(chem_smi, c_exploration=c_exploration, path=path)
            if template_idx is None:
                continue

            # Only grow pathway when we have picked a specific reactants_smi (?)
            if reactants_smi is not None:
                pathway[chem_smi] = (template_idx, reactants_smi) # TODO: figure out if reactants_smi==None case is an issue
            else:
                pathway[chem_smi] = template_idx # still record template selection

            C = self.Chemicals[chem_smi]
            C.visit_count += VIRTUAL_LOSS

            # print('Looking at chemical C: {}'.format(C))
            if template_idx not in C.template_idx_results:
                # print('Creating CTA for {} and {}'.format(chem_smi, template_idx))
                C.template_idx_results[template_idx] = ChemicalTemplateApplication(chem_smi, template_idx)
                CTA = C.template_idx_results[template_idx]

                # TODO: figure out VIRTUAL_LOSS for R.visit_count change?
                # C.reactions[template_idx] = Reaction(chem_smi, template_idx)
                # R = C.reactions[template_idx]
                # R.visit_count += VIRTUAL_LOSS
                leaves.append((chem_smi, template_idx))

            else:
                # Can we assume that the reactants_smi exists in this CTA? I guess so...
                CTA = C.template_idx_results[template_idx]

                if reactants_smi: # if we choose a specific reaction, not just a template...
                    if reactants_smi in CTA.reactions:

                        R = CTA.reactions[reactants_smi]
                        R.visit_count += VIRTUAL_LOSS

                        for smi in R.reactant_smiles:
                            assert smi in self.Chemicals
                            # if self.Chemicals[smi].purchase_price == -1:
                            if not self.Chemicals[smi].done:
                                queue.put((smi, depth+1, path+[smi]))
                        if R.done:
                            C.visit_count += R.visit_count
                            R.visit_count += R.visit_count

        return leaves, pathway

    def update(self, chem_smi, pathway, depth=0):
        """??

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
                C = self.Chemicals[smi]
                CTA = C.template_idx_results[template_idx]
                C.visit_count -= (VIRTUAL_LOSS - 1)
                if reactants_smi:
                    R = CTA.reactions[reactants_smi]
                    R.visit_count -= (VIRTUAL_LOSS - 1)

        if (chem_smi not in pathway) or (depth >= self.max_depth):
            return

        if type(pathway[chem_smi]) == tuple:
            (template_idx, reactants_smi) = pathway[chem_smi]
        else:
            (template_idx, reactants_smi) = (pathway[chem_smi], None)

        C = self.Chemicals[chem_smi]
        CTA = C.template_idx_results[template_idx]
        if CTA.waiting: # haven't actually expanded
            return

        if reactants_smi:
            R = CTA.reactions[reactants_smi]
            if R.valid and (not R.done):
                R.done = all([self.Chemicals[smi].done for smi in R.reactant_smiles])

                for smi in R.reactant_smiles:
                    self.update(smi, pathway, depth+1)

                estimate_price = sum([self.Chemicals[smi].estimate_price for smi in R.reactant_smiles])
                R.update_estimate_price(estimate_price)
                C.update_estimate_price(estimate_price)

                price_list = [self.Chemicals[smi].price for smi in R.reactant_smiles]
                if all([price != -1 for price in price_list]):
                    price = sum(price_list)
                    R.price = price
                    if R.price < C.price or C.price == -1:
                        C.price = R.price

        if sum(len(CTA.reactions) for tid,CTA in C.template_idx_results.items()) >= self.max_branching:
            # print('{} hit max branching, checking if "done"'.format(chem_smi))
            C.done = all([(R.done or (not R.valid)) for rsmi,R in CTA.reactions.items() for tid,CTA in C.template_idx_results.items()])

        # if C.price != -1 and C.price < C.estimate_price:
        #   C.estimate_price = C.price

    def full_update(self, chem_smi, depth=0, path=[]):
        """??

        Args:
            chem_smi (str): SMILES string of target chemical.
            depth (int, optional): (default: {0})
            path (list?): (default: {[]})
        """
        C = self.Chemicals[chem_smi]
        C.pathway_count = 0

        if C.terminal:
            C.pathway_count = 1
            return

        if depth > self.max_depth:
            return

        prefix = '    '* depth

        for template_idx in C.template_idx_results:
            CTA = C.template_idx_results[template_idx]
            for reactants_smi in CTA.reactions:
                R = CTA.reactions[reactants_smi]
                R.pathway_count = 0
                if (not R.valid) or len(set(R.reactant_smiles) & set(path)) > 0:
                    continue
                for smi in R.reactant_smiles:
                    self.full_update(smi, depth+1, path+[chem_smi])
                price_list = [self.Chemicals[smi].price for smi in R.reactant_smiles]
                if all([price != -1 for price in price_list]):
                    price = sum(price_list)
                    R.price = price
                    if R.price < C.price or C.price == -1:
                        C.price = R.price
                        C.best_template = template_idx
                    R.pathway_count = np.prod([self.Chemicals[smi].pathway_count for smi in R.reactant_smiles])
                    # if R.pathway_count != 0:
                    #   print(prefix + '  Reac %d: '%template_idx + str(R.reactant_smiles) + ' %d paths'%R.pathway_count)
                else:
                    R.pathway_count = 0

                # print(prefix + str(R.reactant_smiles) + ' - %d' % R.pathway_count)

        C.pathway_count = 0
        for tid,CTA in C.template_idx_results.items():
            for rct_smi,R in CTA.reactions.items():
                C.pathway_count += R.pathway_count

        # if C.pathway_count != 0:
        #   print(prefix + chem_smi + ' %d paths, price: %.1f' % (C.pathway_count, C.price))

    def build_tree(self, soft_stop=False, known_bad_reactions=[], forbidden_molecules=[], return_first=False):
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
        self.running = True

        with self.allow_join_result():

            MyLogger.print_and_log('Preparing workers...', treebuilder_loc)
            self.prepare()

            # Define first chemical node (target)
            probs, indeces = self.prioritizer()
            value = 1 # current value assigned to precursor (note: may replace with real value function)
            self.Chemicals[self.smiles] = Chemical(self.smiles)
            self.Chemicals[self.smiles].set_template_relevance_probs(probs, indeces, value)
            MyLogger.print_and_log('Calculating initial probs for target', treebuilder_loc)
            hist = self.chemhistorian.lookup_smiles(self.smiles, alreadyCanonical=False)
            self.Chemicals[self.smiles].as_reactant = hist['as_reactant']
            self.Chemicals[self.smiles].as_product = hist['as_product']
            ppg = self.pricer.lookup_smiles(self.smiles, alreadyCanonical=False)
            self.Chemicals[self.smiles].purchase_price = ppg

            # First selection is all the same
            leaves, pathway = self.select_leaf()
            for _id in range(self.num_active_pathways):
                self.active_pathways[_id] = pathway
                self.set_initial_target(_id, leaves)
            MyLogger.print_and_log('Set initial leaves for active pathways', treebuilder_loc)

            # Coordinate workers.
            self.coordinate(soft_stop=soft_stop, known_bad_reactions=known_bad_reactions,
                forbidden_molecules=forbidden_molecules, return_first=return_first)

            # Do a final pass to get counts
            MyLogger.print_and_log('Doing final update of pathway counts / prices', treebuilder_loc)
            self.full_update(self.smiles)
            C = self.Chemicals[self.smiles]

        print("Finished working.")
        print("=== found %d pathways (overcounting duplicate templates)" % C.pathway_count)
        print("=== time for fist pathway: %.2fs" % self.time_for_first_path)
        print("=== min price: %.1f" % C.price)
        print("---------------------------")
        return # self.Chemicals, C.pathway_count, self.time_for_first_path

    def prioritizer(self):
        return self.retroTransformer.template_prioritizer.predict(self.smiles, self.template_count, self.max_cum_template_prob)

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
        return (num_chemicals, num_reactions, [])

    def return_trees(self):
        """Returns retrosynthetic pathways trees and their size."""

        def cheminfodict(smi):
            """Returns dict of extra info about a given chemical.

            Args:
                smi (str): SMILES string of chemical.
            """
            return {
                'smiles': smi,
                'ppg': self.Chemicals[smi].purchase_price,
                'as_reactant': self.Chemicals[smi].as_reactant,
                'as_product': self.Chemicals[smi].as_product,
            }

        def tidlisttoinfodict(tids):
            """Returns dict of info from a given list of templates.

                Args:
                    tids (list of int): Template IDs to get info about.
            """
            if self.retroTransformer.load_all or not self.retroTransformer.use_db:
                return {
                    'tforms': [str(self.retroTransformer.templates[tid]['_id']) for tid in tids],
                    'num_examples': int(sum([self.retroTransformer.templates[tid]['count'] for tid in tids])),
                    'necessary_reagent': self.retroTransformer.templates[tids[0]]['necessary_reagent'],
                }
            else:
                db_client = MongoClient(gc.MONGO['path'], gc.MONGO[
                    'id'], connect=gc.MONGO['connect'])

                db_name = gc.RETRO_TEMPLATES['database']
                collection = gc.RETRO_TEMPLATES['collection']
                TEMPLATE_DB = db_client[db_name][collection]
                tforms = []
                num_examples = 0
                necessary_reagent = None
                for tid in tids:
                    template = TEMPLATE_DB.find_one({'index': tid})
                    tforms.append(str(template.get('_id', -1)))
                    num_examples += template.get('count', 1)
                    if necessary_reagent is None:
                        necessary_reagent = template.get('necessary_reagent', '')
                return {
                    'tforms': tforms,
                    'num_examples': int(num_examples),
                    'necessary_reagent': necessary_reagent,
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
            """Performs iterative depth-first search to find buyable pathways.

            Yields:
                dict: nested dictionaries defining synthesis trees
            """
            print(len(self.Reactions.keys()))
            for path in DLS_chem(self.smiles, depth=0, headNode=True):
                yield chem_dict(chemsmiles_to_id(self.smiles), children=path, **cheminfodict(self.smiles))

        def DLS_chem(chem_smi, depth, headNode=False):
            """Expands at a fixed depth for the current node ``chem_id``.

            Args:
                chem_smi (str): SMILES string for given chemical.
                depth (int): Depth node is expanded at.
                headNode (bool, optional): Unused. (default: {False})
            """
            C = self.Chemicals[chem_smi]
            if C.terminal:
                yield []
            if depth > self.max_depth:
                return

            done_children_of_this_chemical = []

            # if depth > self.max_depth:
            #     return
            for tid, CTA in C.template_idx_results.items():
                ########??????????????????????????######################
                if CTA.waiting:
                    continue
                for rct_smi, R in CTA.reactions.items():
                    if (not R.valid) or R.price == -1:
                        continue
                    rxn_smiles = '.'.join(sorted(R.reactant_smiles)) + '>>' + chem_smi
                    if rxn_smiles not in done_children_of_this_chemical:  # necessary to avoid duplicates
                        for path in DLS_rxn(chem_smi, tid, rct_smi, depth):
                            yield [rxn_dict(rxnsmiles_to_id(rxn_smiles), rxn_smiles, children=path,
                                            plausibility=R.plausibility, template_score=R.template_score,
                                            **tidlisttoinfodict(R.tforms))]
                        done_children_of_this_chemical.append(rxn_smiles)

        def DLS_rxn(chem_smi, template_idx, rct_smi, depth):
            """Yields children paths starting from a specific ``rxn_id``.

            Args:
                chem_smi (str): SMILES string for given chemical.
                template_idx (int): Index of given template.
                rct_smi (str): SMILES string for reactants.
                depth (int): depth of given reaction.
            """
            # TODO: add in auxiliary information about templates, etc.
            R = self.Chemicals[chem_smi].template_idx_results[template_idx].reactions[rct_smi]

            # rxn_list = []
            # for smi in R.reactant_smiles:
            #     rxn_list.append([chem_dict(smi, children=path, **{}) for path in DLS_chem(smi, depth+1)])

            # return [rxns[0] for rxns in itertools.product(rxn_list)]

            ###################
            # To get recursion working properly with generators, need to hard-code these cases? Unclear
            # whether itertools.product can actually work with generators. Seems like it can't work
            # well...

            # Only one reactant? easy!
            if len(R.reactant_smiles) == 1:
                chem_smi0 = R.reactant_smiles[0]
                for path in DLS_chem(chem_smi0, depth + 1):
                    yield [
                        chem_dict(chemsmiles_to_id(chem_smi0), children=path, **cheminfodict(chem_smi0))
                    ]

            # Two reactants? want to capture all combinations of each node's
            # options
            elif len(R.reactant_smiles) == 2:
                chem_smi0 = R.reactant_smiles[0]
                chem_smi1 = R.reactant_smiles[1]
                for path0 in DLS_chem(chem_smi0, depth + 1):
                    for path1 in DLS_chem(chem_smi1, depth + 1):
                        yield [
                            chem_dict(chemsmiles_to_id(chem_smi0), children=path0, **cheminfodict(chem_smi0)),
                            chem_dict(chemsmiles_to_id(chem_smi1), children=path1, **cheminfodict(chem_smi1)),
                        ]

            # Three reactants? This is not elegant...
            elif len(R.reactant_smiles) == 3:
                chem_smi0 = R.reactant_smiles[0]
                chem_smi1 = R.reactant_smiles[1]
                chem_smi2 = R.reactant_smiles[2]
                for path0 in DLS_chem(chem_smi0, depth + 1):
                    for path1 in DLS_chem(chem_smi1, depth + 1):
                        for path2 in DLS_chem(chem_smi2, depth + 1):
                            yield [
                                chem_dict(chemsmiles_to_id(chem_smi0), children=path0, **cheminfodict(chem_smi0)),
                                chem_dict(chemsmiles_to_id(chem_smi1), children=path1, **cheminfodict(chem_smi1)),
                                chem_dict(chemsmiles_to_id(chem_smi2), children=path2, **cheminfodict(chem_smi2)),
                            ]

            # I am ashamed
            elif len(R.reactant_smiles) == 4:
                chem_smi0 = R.reactant_smiles[0]
                chem_smi1 = R.reactant_smiles[1]
                chem_smi2 = R.reactant_smiles[2]
                chem_smi3 = R.reactant_smiles[3]
                for path0 in DLS_chem(chem_smi0, depth + 1):
                    for path1 in DLS_chem(chem_smi1, depth + 1):
                        for path2 in DLS_chem(chem_smi2, depth + 1):
                            for path3 in DLS_chem(chem_smi3, depth + 1):
                                yield [
                                    chem_dict(chemsmiles_to_id(chem_smi0), children=path0, **cheminfodict(chem_smi0)),
                                    chem_dict(chemsmiles_to_id(chem_smi1), children=path1, **cheminfodict(chem_smi1)),
                                    chem_dict(chemsmiles_to_id(chem_smi2), children=path2, **cheminfodict(chem_smi2)),
                                    chem_dict(chemsmiles_to_id(chem_smi3), children=path3, **cheminfodict(chem_smi3)),
                                ]

            else:
                print('Too many reactants! Only have cases 1-4 programmed')
                print('There probably are not any real 5 component reactions')
                print(R.reactant_smiles)

        MyLogger.print_and_log('Retrieving trees...', treebuilder_loc)
        trees = []
        for tree in IDDFS():
            trees.append(tree)
            if len(trees) >= self.max_trees:
                break

        # Sort by some metric
        def number_of_starting_materials(tree):
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
                          chiral=True,
                          max_trees=5000,
                          max_ppg=1e10,
                          known_bad_reactions=[],
                          forbidden_molecules=[],
                          template_count=100,
                          max_cum_template_prob=0.995,
                          max_natom_dict=defaultdict(lambda: 1e9, {'logic': None}),
                          min_chemical_history_dict={'as_reactant': 1e9, 'as_product': 1e9, 'logic': None},
                          apply_fast_filter=True,
                          filter_threshold=0.75,
                          soft_reset=False,
                          return_first=False,
                          sort_trees_by='plausibility',
                          template_prioritizer='reaxys',
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
            chiral (bool, optional): Whether or not to pay close attention to
                chirality. When False, even achiral templates can lead to
                accidental inversion of chirality in non-reacting parts of the
                molecule. It is highly recommended to keep this as True.
                (default: {True})
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
            max_natom_dict (defaultdict, optional): Specifies maximum amounts
                for certain atoms and the logic it should use to select a
                chemical (buyable, buyable or max atoms, buyable and max atoms).
                (default: {defaultdict(lambda: 1e9, {'logic': None})})
            min_chemical_history_dict (dict, optional): Minimum number of times
                a chemical must appear as a reactant or product to be selected
                when logic is "OR" and chemical is not buyable.
                (default: {{'as_reactant':1e9, 'as_product':1e9,'logic':None}})
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
            **kwargs: Additional optional arguments.

        Returns:
            ((int, int, dict), list of dict):
                tree_status ((int, int, dict)): Result of tree_status().

                trees (list of dict): List of dictionaries, where each dictionary
                    defines a synthetic route.
        """
        self.smiles = smiles
        self.max_depth = max_depth
        self.expansion_time = expansion_time
        self.nproc = nproc
        if num_active_pathways is None:
            num_active_pathways = nproc
        self.num_active_pathways = num_active_pathways
        self.max_trees = max_trees
        self.max_cum_template_prob = max_cum_template_prob
        self.template_count = template_count
        self.filter_threshold = filter_threshold
        self.apply_fast_filter = apply_fast_filter
        self.min_chemical_history_dict = min_chemical_history_dict
        self.max_natom_dict = max_natom_dict
        self.max_ppg = max_ppg
        self.sort_trees_by = sort_trees_by
        self.template_prioritizer = template_prioritizer
        MyLogger.print_and_log('Active pathway #: {}'.format(num_active_pathways), treebuilder_loc)

        if min_chemical_history_dict['logic'] not in [None, 'none'] and self.chemhistorian is None:
            self.load_chemhistorian()

        # Define stop criterion
        def is_buyable(ppg):
            return ppg and (ppg <= self.max_ppg)

        def is_small_enough(smiles):
            # Get structural properties
            natom_dict = defaultdict(lambda: 0)
            mol = Chem.MolFromSmiles(smiles)
            if not mol:
                return False
            for a in mol.GetAtoms():
                natom_dict[a.GetSymbol()] += 1
            natom_dict['H'] = sum(a.GetTotalNumHs() for a in mol.GetAtoms())
            max_natom_satisfied = all(natom_dict[k] <= v for (
                k, v) in max_natom_dict.items() if k != 'logic')
            return max_natom_satisfied

        def is_popular_enough(hist):
            return hist['as_reactant'] >= min_chemical_history_dict['as_reactant'] or \
                   hist['as_product'] >= min_chemical_history_dict['as_product']

        if min_chemical_history_dict['logic'] in [None, 'none']:
            if max_natom_dict['logic'] in [None, 'none']:
                def is_a_terminal_node(smiles, ppg, hist):
                    return is_buyable(ppg)
            elif max_natom_dict['logic'] == 'or':
                def is_a_terminal_node(smiles, ppg, hist):
                    return is_buyable(ppg) or is_small_enough(smiles)
            else:
                def is_a_terminal_node(smiles, ppg, hist):
                    return is_buyable(ppg) and is_small_enough(smiles)
        else:
            if max_natom_dict['logic'] in [None, 'none']:
                def is_a_terminal_node(smiles, ppg, hist):
                    return is_buyable(ppg) or is_popular_enough(hist)
            elif max_natom_dict['logic'] == 'or':
                def is_a_terminal_node(smiles, ppg, hist):
                    return is_buyable(ppg) or is_popular_enough(hist) or is_small_enough(smiles)
            else:
                def is_a_terminal_node(smiles, ppg, hist):
                    return is_popular_enough(hist) or (is_buyable(ppg) and is_small_enough(smiles))

        self.is_a_terminal_node = is_a_terminal_node

        self.reset(soft_reset=soft_reset)

        MyLogger.print_and_log('Starting search for {}'.format(smiles), treebuilder_loc)
        self.build_tree(soft_stop=kwargs.pop('soft_stop', False),
                        known_bad_reactions=known_bad_reactions,
                        forbidden_molecules=forbidden_molecules,
                        return_first=return_first,
                        )

        return self.return_trees()

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


class MCTSCelery(MCTS):
    """
    This is a subclass of MCTS which uses celery for multiprocessing.

    Note regarding model and data loading: This class uses pricing data,
    chemhistorian data, template prioritizer, and retro transformer. The retro
    transformer additionally needs the precursor prioritizer and fast filter.
    If instantiating this class with no arguments, only Pricer and ChemHistorian
    data will be loaded (using database) by default. The RetroTransformer
    is provided via ``tb_c_worker`` which is configured separately. The
    template prioritizer should be passed directly to ``get_buyable_paths``.

    Attributes:

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        from celery.result import allow_join_result
        self.allow_join_result = allow_join_result

    @staticmethod
    def load_pricer():
        """
        Loads pricer.
        """
        pricer = Pricer(use_db=True)
        pricer.load()
        return pricer

    @staticmethod
    def load_chemhistorian():
        """
        Loads chemhistorian.
        """
        chemhistorian = ChemHistorian(use_db=True)
        chemhistorian.load()
        return chemhistorian

    @staticmethod
    def load_retro_transformer(**kwargs):
        """
        Loads retro transformer model.
        """
        # Still need to load to have num refs, etc.
        MyLogger.print_and_log('Loading transforms for informational purposes only', treebuilder_loc)
        retro_transformer = RetroTransformer(
            template_prioritizer=None,
            precursor_prioritizer=None,
            fast_filter=None
        )
        retro_transformer.load()
        MyLogger.print_and_log(
            '...done loading {} informational transforms!'.format(len(retro_transformer.templates)), treebuilder_loc)
        return retro_transformer

    def reset_workers(self, soft_reset=False):
        # general parameters in celery format
        # TODO: anything goes here?
        self.pending_results = []

    def expand(self, _id, smiles, template_idx):  # TODO: make Celery workers
        """Adds pathway to be worked on with Celery.

        Args:
            _id (int): ID of pending pathway.
            smiles (str): SMILES string of molecule to be exanded.
            template_idx (int): ID of template to apply to molecule.
        """
        # Chiral transformation or heuristic prioritization requires
        # same database. _id is _id of active pathway
        self.pending_results.append(tb_c_worker.apply_one_template_by_idx.apply_async(
            args=(_id, smiles, template_idx),
            kwargs={'max_num_templates': self.template_count,
                    'max_cum_prob': self.max_cum_template_prob,
                    'fast_filter_threshold': self.filter_threshold,
                    'template_prioritizer': self.template_prioritizer},
            # queue=self.private_worker_queue, ## CWC TEST: don't reserve
        ))
        self.status[(smiles, template_idx)] = WAITING
        self.active_pathways_pending[_id] += 1

    def prepare(self):
        """Starts parallelization with Celery."""
        try:
            ## CWC TEST: don't reserve
            res = tb_c_worker.apply_one_template_by_idx.delay(
                1,
                'CCOC(=O)[C@H]1C[C@@H](C(=O)N2[C@@H](c3ccccc3)CC[C@@H]2c2ccccc2)[C@@H](c2ccccc2)N1',
                1,
                template_prioritizer='reaxys'
            )
            res.get(20)
        except Exception as e:
            res.revoke()
            raise IOError('Did not find any workers? Try again later ({})'.format(e))

    def get_ready_result(self):
        """Yields processed results from Celery.

        Yields:
            list of 5-tuples of (int, string, int, list, float): Results
                from workers after applying a template to a molecule.
        """
        # Update which processes are ready
        self.is_ready = [i for (i, res) in enumerate(self.pending_results) if res.ready()]
        for i in self.is_ready:
            yield self.pending_results[i].get(timeout=0.1)
            self.pending_results[i].forget()
        self.pending_results = [res for (i, res) in enumerate(self.pending_results) if i not in self.is_ready]

    def stop(self, soft_stop=False):
        """Stops work with Celery.

        Args:
            soft_stop (bool, optional): Unused. (default: {false})
        """
        self.running = False
        if self.pending_results != []:  # clear anything left over - might not be necessary
            for i in range(len(self.pending_results)):
                self.pending_results[i].revoke()

    def prioritizer(self):
        res = tb_c_worker.template_relevance.delay(self.smiles, self.template_count, self.max_cum_template_prob)
        return res.get(10)

    def work(self, i):
        """
        Explicitly override work method of MCTS since Celery does not use it.
        """
        raise NotImplementedError('MCTSCelery does not support the work method. Did you mean to use MCTS?')
