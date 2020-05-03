import torch.nn as nn

class MLP(nn.Module):
    """
    Vanilla MLP model. Can be used for classification or regression.
    """
    def __init__(self, input_dim, output_dim, hidden_dims=[], activation=nn.ReLU):
        super().__init__()
        sizes = [input_dim] + hidden_dims + [output_dim]
        layers = []
        for i in range(1, len(sizes)):
            layers.append(nn.Linear(sizes[i - 1], sizes[i]))
            self.xavier_init(layers[-1])  # Xavier initialization
            layers.append(activation())
        self.net = nn.Sequential(*layers[:-1])  # drop the last activation layer
        self.num_layers = len(layers)

    def forward(self, x):
        return self.net(x)

    def xavier_init(self, linear_layer):
        nn.init.xavier_uniform_(linear_layer.weight, gain=1.0)
        if linear_layer.bias is not None:
            nn.init.zeros_(linear_layer.bias)