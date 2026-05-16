class LayerNormalization(nn.Module):
    def __init__(self, parameters_shape, eps=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(parameters_shape)) #initialize learnable params
        self.beta  = nn.Parameter(torch.zeros(parameters_shape)) #initialize learnable params
        self.eps   = eps # avoid std=0, if all values are identical

    def forward(self, x):
        dims = [-(i + 1) for i in range(len(self.gamma.shape))]
        mean = x.mean(dim=dims, keepdim=True)
        std  = ((x - mean) ** 2).mean(dim=dims, keepdim=True).add(self.eps).sqrt()
        return self.gamma * (x - mean) / std + self.beta


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, hidden, drop_prob=0.1): #hidden = size of FFN  network
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden), # find the weight when dmodel -> hidden
            nn.ReLU(), #normalizing (0,1)
            nn.Dropout(drop_prob),
            nn.Linear(hidden, d_model), #change to ori dimension
        )

    def forward(self, x):
        return self.net(x) # pass x to FFN