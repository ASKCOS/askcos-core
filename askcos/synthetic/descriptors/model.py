from argparse import Namespace

import torch.nn as nn

from askcos.synthetic.descriptors.mpn import MPN
from askcos.synthetic.descriptors.ffn import MultiReadout

class MoleculeModel(nn.Module):
    """A MoleculeModel is a model which contains a message passing network following by feed-forward layers."""

    def __init__(self, args, atom_fdim, bond_fdim):
        """
        Initializes the MoleculeModel.

        :param classification: Whether the model is a classification model.
        """
        super(MoleculeModel, self).__init__()
        self.create_encoder(args, atom_fdim, bond_fdim)
        self.create_ffn(args)

    def create_encoder(self, args: Namespace, atom_fdim, bond_fdim):
        """
        Creates the message passing encoder for the model.

        :param args: Arguments.
        """
        self.encoder = MPN(args, atom_fdim, bond_fdim)

    def create_ffn(self, args: Namespace):
        """
        Creates the feed-forward network for the model.

        :param args: Arguments.
        """

        # Create readout layer
        self.readout = MultiReadout(args, args.atom_targets, args.bond_targets,
                                    args.atom_constraints, args.bond_constraints)

    def forward(self, input):
        """
        Runs the MoleculeModel on input.

        :param input: Input.
        :return: The output of the MoleculeModel.
        """
        output_all = self.readout(self.encoder(input))

        return output_all
