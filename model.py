import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, drop_p = 0.0, hidden_layers_config = [64,64], debug=False,
                 xavier_init=False):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        "*** YOUR CODE HERE ***"

        # ANAS

        print(f"state_size: {state_size}, action_size: {action_size}")
        # hidden_layers = [64, 64]  # ANAS MAKE constructor arg
        # drop_p = 0.0  # ANAS make constructor arg
        self.hidden_layers_config=hidden_layers_config
        self.hidden_layers = nn.ModuleList([nn.Linear(state_size, hidden_layers_config[0])])

        # Add a variable number of more hidden layers
        layer_sizes = zip(hidden_layers_config[:-1], hidden_layers_config[1:])
        if debug:
            print(f"layer_sizes: {layer_sizes}")
        for i, layer_details in enumerate(layer_sizes):
            if debug:
                print(f"hidden layer {i}, layer_details: {layer_details}")
            linear_layer = nn.Linear(layer_details[0], layer_details[1])
            if xavier_init:
                nn.init.xavier_uniform_(linear_layer.weight)
            self.hidden_layers.extend([linear_layer])

        # self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
        if debug:
            print(f"hidden_layers: {self.hidden_layers}")

        self.output = nn.Linear(hidden_layers_config[-1], action_size)
        self.drop_p = drop_p
        self.dropout = nn.Dropout(p=drop_p)

    def forward(self, state):
        """Build a network that maps state -> action values."""

        # ANAS
        # Forward through each layer in `hidden_layers`, with ReLU activation and dropout
        x = state

        for linear in self.hidden_layers:
            x = F.relu(linear(x))
            x = self.dropout(x)
        x = self.output(x)

        return x