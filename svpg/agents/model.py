import torch.nn as nn


def make_model(input_size, output_size, activation="ReLU", **kwargs):
    hidden_size = list(kwargs.values())

    if len(hidden_size) > 1:
        hidden_layers = [
            [nn.Linear(hidden_size[i], hidden_size[i+1]), nn.ReLU()]
            for i in range(0, len(hidden_size) - 1)
        ] if activation == "ReLU" else [
            [nn.Linear(hidden_size[i], hidden_size[i+1]), nn.SiLU()]
            for i in range(0, len(hidden_size) - 1)
        ]

        hidden_layers = [l for layers in hidden_layers for l in layers]
    else:
        hidden_layers = [nn.Identity()]

    
    return nn.Sequential(
        nn.Linear(input_size, hidden_size[0]),
        nn.ReLU(),
        *hidden_layers,
        nn.Linear(hidden_size[-1], output_size)
    ) if activation == "ReLU" else nn.Sequential(
        nn.Linear(input_size, hidden_size[0]),
        nn.SiLU(),
        *hidden_layers,
        nn.Linear(hidden_size[-1], output_size)
    )
