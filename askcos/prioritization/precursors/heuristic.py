from askcos.prioritization.prioritizer import Prioritizer
import rdkit.Chem as Chem
from rdkit.Chem import AllChem
import numpy as np
from askcos.utilities.buyable.pricer import Pricer
from askcos.utilities.io.logger import MyLogger
heuristic_precursor_prioritizer_loc = 'heuristic_precursor_prioritizer'


class HeuristicPrecursorPrioritizer(Prioritizer):
    """A precursor Prioritizer that uses a heuristic scoring function.

    Attributes:
        pricer (Pricer or None): Used to look up chemical prices.
    """
    def __init__(self):
        """Initializes HeuristicPrecursorPrioritizer."""
        self.pricer = None
        self._loaded = False

    def get_priority(self, retroPrecursor, **kwargs):
        """Gets priority of given precursor based on heuristic function.

        Args:
            retroPrecursor (RetroPrecursor): Precursor to calculate priority of.
            **kwargs: Unused.

        Returns:
            float: Priority score of precursor.
        """
        if not self._loaded:
            self.load_model()

        necessary_reagent_atoms = retroPrecursor.necessary_reagent.count('[') / 2.
        scores = []
        for smiles in retroPrecursor.smiles_list:
            # If buyable, basically free
            ppg = self.pricer.lookup_smiles(smiles, alreadyCanonical=True)
            if ppg:
                scores.append(- ppg / 5.0)
                continue

            # Else, use heuristic
            x = Chem.MolFromSmiles(smiles)
            total_atoms = x.GetNumHeavyAtoms()
            ring_bonds = sum([b.IsInRing() - b.GetIsAromatic()
                              for b in x.GetBonds()])
            chiral_centers = len(Chem.FindMolChiralCenters(x))

            scores.append(
                - 2.00 * np.power(total_atoms, 1.5)
                - 1.00 * np.power(ring_bonds, 1.5)
                - 2.00 * np.power(chiral_centers, 2.0)
            )

        return np.sum(scores) - 4.00 * np.power(necessary_reagent_atoms, 2.0)

    def load_model(self):
        """Loads the Pricer used in the heuristic priority scoring."""
        self.pricer = Pricer()
        self.pricer.load()
        self._loaded = True
