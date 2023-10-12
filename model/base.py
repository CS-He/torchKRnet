import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0.0)
class DiagGaussian():
    """
    Multivariate Gaussian distribution with diagonal covariance matrix
    """

    def __init__(self, mu, cov):
        """Constructor

        Args:
          shape: Tuple with shape of data, if int shape has one dimension
          trainable: Flag whether to use trainable or fixed parameters
        """
        # super().__init__()
        self.shape = mu.shape
        self.d = np.prod(self.shape)
       
        self.loc=  mu
        self.scale = torch.sqrt(torch.diagonal(cov)).view(-1,)
        self.log_scale=torch.log(self.scale)#torch.zeros(1, *self.shape)

    def forward(self, num_samples=1):
        eps = torch.randn(
            (num_samples,) + self.shape, dtype=self.loc.dtype, device=self.loc.device
        )
        log_scale = self.log_scale

        z = self.loc + self.scale * eps
        log_p = -0.5 * self.d * np.log(2 * np.pi) - torch.sum(
            log_scale + 0.5 * torch.pow(eps, 2), -1
        )
        return z, log_p
    def sample(self, sample_shape=torch.Size()):
        z, log_p = self.forward(sample_shape[0])
        return z
    def log_prob(self, z):
        log_scale = self.log_scale
        # qurad_form = torch.pow((z - self.loc) / torch.exp(log_scale), 2)
        log_p = -0.5 * self.d * np.log(2 * np.pi) - torch.sum(
            log_scale + 0.5 * torch.pow((z - self.loc) / self.scale, 2),
           dim=-1)
        return log_p

class modified_MLP(nn.Module):
    def __init__(self,layer_dim_list) -> None:
        super().__init__()
        self.linear1 = nn.Linear(layer_dim_list[0], layer_dim_list[1])
        self.linear2 = nn.Linear(layer_dim_list[0], layer_dim_list[1])
        self.layers = nn.ModuleList([nn.Linear(layer_dim_list[i], layer_dim_list[i+1],bias=False) for i in range(len(layer_dim_list)-1)])
        self.output_bias = nn.Parameter(torch.zeros(layer_dim_list[-1]))
    def forward(self,x):
        U = torch.tanh(self.linear1(x))
        V = torch.tanh(self.linear2(x))
        for layer in self.layers[:-1]:
            output = torch.tanh(layer(x))
            x = torch.multiply(output, U) + torch.multiply(1-output, V)
        output = self.layers[-1](x) + self.output_bias
        return output    


# Logistic mapping layer: mapping between a bounded domain (0,1)
#                         and an infinite domain (-inf, +inf)
# The default direction is from (-inf, +inf) to (0,1)
class Logistic_mapping(nn.Module):
  def __init__(self, input_dim):
    super().__init__()
    self.s_init = 0.5
    self.input_dim = input_dim
    self.s = nn.Parameter(torch.zeros(1, input_dim))

  # the direction of this mapping is not related to the flow
  # direction between the data and the prior
  def forward(self, inputs):
    x = inputs
    x = (torch.tanh(x / (self.s_init + self.s)) + 1.0) / 2.0

    x = torch.clip(x, 1.0e-10, 1.0-1.0e-10)
    tp = torch.log(x) + torch.log(1-x) + torch.log(2.0/(self.s+self.s_init))
    dlogdet = torch.sum(tp, dim=1)
    return x, dlogdet
  def inverse(self, inputs):
    x = inputs
    x = torch.clip(x, 1.0e-10, 1.0-1.0e-10)
    tp1 = torch.log(x)
    tp2 = torch.log(1 - x)
    x = (self.s_init + self.s) / 2.0 * (tp1 - tp2)
    tp = torch.log((self.s+self.s_init)/2.0) - tp1 - tp2
    dlogdet = torch.sum(tp, dim=1)
    return x, dlogdet


class GaussianMixture(nn.Module):
    """
    Mixture of Gaussians with diagonal covariance matrix
    """
    def __init__(self, n_modes, dim, loc=None, scale=None, weights=None, trainable=True):
        """
        Constructor
        :param n_modes: Number of modes of the mixture model
        :param dim: Number of dimensions of each Gaussian
        :param loc: List of mean values
        :param scale: List of diagonals of the covariance matrices
        :param weights: List of mode probabilities
        :param trainable: Flag, if true parameters will be optimized during training
        """
        super().__init__()

        self.n_modes = n_modes
        self.dim = dim

        if loc is None:
            loc = np.random.randn(self.n_modes, self.dim)
        loc = np.array(loc)[None, ...]
        if scale is None:
            scale = np.ones((self.n_modes, self.dim))
        scale = np.array(scale)[None, ...]
        if weights is None:
            weights = np.ones(self.n_modes)
        weights = np.array(weights)[None, ...]
        weights /= weights.sum(1)

        if trainable:
            self.loc = nn.Parameter(torch.tensor(1. * loc))
            self.log_scale = nn.Parameter(torch.tensor(np.log(1. * scale)))
            self.weight_scores = nn.Parameter(torch.tensor(np.log(1. * weights)))
        else:
            self.register_buffer("loc", torch.tensor(1. * loc))
            self.register_buffer("log_scale", torch.tensor(np.log(1. * scale)))
            self.register_buffer("weight_scores", torch.tensor(np.log(1. * weights)))

    def forward(self, num_samples=1):
        # Sample mode indices
        mode_ind = torch.randint(high=self.n_modes, size=(num_samples,),device=self.loc.device)
        mode_1h = torch.zeros((num_samples, self.n_modes), dtype=torch.int64, device=self.loc.device)
        mode_1h.scatter_(1, mode_ind[:, None], 1)
        mode_1h = mode_1h[..., None]

        # Get weights
        weights = torch.softmax(self.weight_scores, 1)

        # Get samples
        eps = torch.randn(num_samples, self.dim, dtype=self.loc.dtype, device=self.loc.device)
        scale_sample = torch.sum(torch.exp(self.log_scale) * mode_1h, 1)
        loc_sample = torch.sum(self.loc * mode_1h, 1)
        z = eps * scale_sample + loc_sample

        # Compute log probability
        log_p = - 0.5 * self.dim * np.log(2 * np.pi) + torch.log(weights)\
                - 0.5 * torch.sum(torch.pow(eps, 2), 1, keepdim=True)\
                - torch.sum(self.log_scale, 2)
        log_p = torch.logsumexp(log_p, 1)

        return z, log_p
    def sample(self, num_shape,t=0):
        num_samples = num_shape[0]
        return self.forward(num_samples)[0]
    def log_prob(self, z, t=0):
        # Get weights
        weights = torch.softmax(self.weight_scores, 1)

        # Compute log probability
        eps = (z[:, None, :] - self.loc) / torch.exp(self.log_scale)
        log_p = - 0.5 * self.dim * np.log(2 * np.pi) + torch.log(weights) \
                - 0.5 * torch.sum(torch.pow(eps, 2), 2) \
                - torch.sum(self.log_scale, 2)
        log_p = torch.logsumexp(log_p, 1)

        return log_p

def input_mapping(x, B): 
  if B is None:
    return x
  else:
    x_proj = (2.*np.pi*x) @ B.T
    return np.concatenate([np.sin(x_proj), np.cos(x_proj)], axis=-1)

