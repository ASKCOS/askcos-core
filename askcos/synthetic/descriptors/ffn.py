from argparse import Namespace

import torch
import torch.nn as nn
import re

class AttrProxy(object):
    """Translates index lookups into attribute lookups"""
    def __init__(self, module, prefix):
        self.module = module
        self.prefix = prefix

    def __len__(self):
        return len([x for x in self.module.__dict__['_modules'].keys() if re.match(f'{self.prefix}\d+', x)])

    def __getitem__(self, item):
        if item >= len(self):
            raise IndexError
        return getattr(self.module, self.prefix + str(item))

class MultiReadout(nn.Module):
    """A fake list of FFNs for reading out as suggested in
    https://discuss.pytorch.org/t/list-of-nn-module-in-a-nn-module/219/3 """

    def __init__(self, args: Namespace, atom_targets, bond_targets=None,
                 atom_constraints=None, bond_constraints=None, attention=False):
        """

        :param args:
        :param args:
        :param constraints:
        """

        features_size = args.hidden_size
        hidden_size = args.ffn_hidden_size
        num_layers = args.ffn_num_layers
        output_size = args.output_size
        dropout = nn.Dropout(args.dropout)
        activation = nn.ReLU()

        super(MultiReadout, self).__init__()
        for i, a_target in enumerate(atom_targets):
            constraint = atom_constraints[i] if atom_constraints is not None and i < len(atom_constraints) else None
            if attention:
                self.add_module(f'readout_{i}', FFNAtten(features_size, hidden_size, num_layers,
                                                         output_size, dropout, activation, constraint, ffn_type='atom'))
            else:
                self.add_module(f'readout_{i}', FFN(features_size, hidden_size, num_layers,
                                                    output_size, dropout, activation, constraint, ffn_type='atom'))

        i += 1
        for j, b_target in enumerate(bond_targets):
            i += j
            constraint = bond_constraints[i] if bond_constraints and j < len(bond_constraints) else None
            self.add_module(f'readout_{i}', FFN(features_size, hidden_size, num_layers,
                                                output_size, dropout, activation, constraint, ffn_type='bond'))

        self.ffn_list = AttrProxy(self, 'readout_')

    def forward(self, *input):
        return [ffn(*input) for ffn in self.ffn_list]


class FFNAtten(nn.Module):
    def __init__(self, features_size, hidden_size, num_layers, output_size,
                 dropout, activation, constraint=None, ffn_type='atom'):
        """Initializes the FFN.

        args: Arguments.
        constraints: constraints applied to output
        """

        super(FFN, self).__init__()

        self.ffn = DenseLayers(features_size, hidden_size,
                               num_layers, hidden_size, dropout, activation)
        self.ffn_readout = DenseLayers(hidden_size, hidden_size, 1, output_size, dropout, activation)


        self.weights_readout = DenseLayers(first_linear_dim=hidden_size, output_size=output_size, num_layers=2,
                                           dropout=dropout, activation=activation)
        self.constraint = constraint

    def forward(self, input):
        """
        Runs the FFN on input

        :param input:
        :return:
        """
        a_hidden, a_scope, b_hidden, b_scope, b2br, bond_types = input

        hidden = a_hidden
        scope = a_scope

        output_hidden = self.ffn(hidden)
        output = self.ffn_readout(output_hidden)

        weights = self.weights_readout(output_hidden)
        constrained_output = []
        for i, (a_start, a_size) in enumerate(scope):
            if a_size == 0:
                continue
            else:
                cur_weights = weights.narrow(0, a_start, a_size)
                cur_output = output.narrow(0, a_start, a_size)

                cur_weights = torch.nn.Softmax()(cur_weights)

                cur_output_sum = cur_output.sum()

                cur_output = cur_output + cur_weights * (self.constraint - cur_output_sum)
                constrained_output.append(cur_output)
        output = torch.cat(constrained_output, dim=0)

        return output


class FFN(nn.Module):
    """A Feedforward netowrk reading out properties from fingerprint"""

    def __init__(self, features_size, hidden_size, num_layers, output_size,
                 dropout, activation, constraint=None, ffn_type='atom', attention=False):
        """Initializes the FFN.

        args: Arguments.
        constraints: constraints applied to output
        """

        super(FFN, self).__init__()
        if ffn_type == 'atom':
            self.ffn = DenseLayers(features_size, hidden_size,
                                   num_layers, output_size, dropout, activation)
        elif ffn_type == 'bond':
            self.ffn = DenseLayers(2*features_size, hidden_size,
                                   num_layers, output_size, dropout, activation)
        self.ffn_type = ffn_type
        self.attention = attention

        if constraint is not None:
            self.weights_readout = DenseLayers(features_size, hidden_size,
                                               num_layers, output_size, dropout, activation)
            if attention:
                self.weights_readout = DenseLayers(first_linear_dim=hidden_size, output_size=1, num_layers=1,
                                                   dropout=dropout, activation=activation)
            self.constraint = constraint
        else:
            self.constraint = None

    def forward(self, input):
        """
        Runs the FFN on input

        :param input:
        :return:
        """
        a_hidden, a_scope, b_hidden, b_scope, b2br, bond_types = input

        if self.ffn_type == 'atom':
            hidden = a_hidden
            scope = a_scope

            output = self.ffn(hidden)
            if self.attention:
                weights = self.weights_readout(output)


            if self.constraint is not None:
                weights = self.weights_readout(hidden)
                constrained_output = []
                for i, (a_start, a_size) in enumerate(scope):
                    if a_size == 0:
                        continue
                    else:
                        cur_weights = weights.narrow(0, a_start, a_size)
                        cur_output = output.narrow(0, a_start, a_size)

                        cur_weights_sum = cur_weights.sum()
                        cur_output_sum = cur_output.sum()

                        cur_output = cur_output + cur_weights * \
                                     (self.constraint - cur_output_sum) / cur_weights_sum
                        constrained_output.append(cur_output)
                output = torch.cat(constrained_output, dim=0)
            else:
                output = output[1:]
        elif self.ffn_type == 'bond':

            forward_bond = b_hidden[b2br[:, 0]]
            backward_bond = b_hidden[b2br[:, 1]]

            b_hidden = torch.cat([forward_bond, backward_bond], dim=1)

            output = self.ffn(b_hidden) + bond_types.reshape(-1, 1)

        return output


class DenseLayers(nn.Module):
    "Dense layers"

    def __init__(self,
                 first_linear_dim: int,
                 hidden_size: int,
                 num_layers: int,
                 output_size: int,
                 dropout: nn.Module,
                 activation) -> nn.Sequential:
        """
        :param first_linear_dim:
        :param hidden_size:
        :param num_layers:
        :param output_size:
        :param dropout:
        :param activation:
        """
        super(DenseLayers, self).__init__()
        if num_layers == 1:
            layers = [
                dropout,
                nn.Linear(first_linear_dim, output_size)
            ]
        else:
            layers = [
                dropout,
                nn.Linear(first_linear_dim, hidden_size)
            ]
            for _ in range(num_layers - 2):
                layers.extend([
                    activation,
                    dropout,
                    nn.Linear(hidden_size, hidden_size),
                ])
            layers.extend([
                activation,
                dropout,
                nn.Linear(hidden_size, output_size),
            ])

        self.dense_layers = nn.Sequential(*layers)

    def forward(self, input):
        return self.dense_layers(input)
