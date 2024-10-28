# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import inspect
from typing import Callable, List, Optional, Set, Tuple, Union
import warnings

import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.wishart import Wishart
from torch.distributions.dirichlet import Dirichlet
from torch.distributions.gamma import Gamma
from torch import nn
import torch.nn.functional as F
from torch.distributions import constraints

from copy import deepcopy as cp

from packaging import version
from safetensors.torch import storage_ptr, storage_size
from torch import nn

from .utils import is_torch_xla_available, logging


ALL_LAYERNORM_LAYERS = [nn.LayerNorm]

logger = logging.get_logger(__name__)
eps = 1e-5

parsed_torch_version_base = version.parse(version.parse(torch.__version__).base_version)

is_torch_greater_or_equal_than_2_2 = parsed_torch_version_base >= version.parse("2.2")
is_torch_greater_or_equal_than_2_1 = parsed_torch_version_base >= version.parse("2.1")
is_torch_greater_or_equal_than_2_0 = parsed_torch_version_base >= version.parse("2.0")
is_torch_greater_or_equal_than_1_13 = parsed_torch_version_base >= version.parse("1.13")
is_torch_greater_or_equal_than_1_12 = parsed_torch_version_base >= version.parse("1.12")


def softmax_backward_data(parent, grad_output, output, dim, self):
    """
    A function that calls the internal `_softmax_backward_data` PyTorch method and that adjusts the arguments according
    to the torch version detected.
    """

    from torch import _softmax_backward_data

    return _softmax_backward_data(grad_output, output, parent.dim, self.dtype)


def prune_linear_layer(layer: nn.Linear, index: torch.LongTensor, dim: int = 0) -> nn.Linear:
    """
    Prune a linear layer to keep only entries in index.

    Used to remove heads.

    Args:
        layer (`torch.nn.Linear`): The layer to prune.
        index (`torch.LongTensor`): The indices to keep in the layer.
        dim (`int`, *optional*, defaults to 0): The dimension on which to keep the indices.

    Returns:
        `torch.nn.Linear`: The pruned layer as a new layer with `requires_grad=True`.
    """
    index = index.to(layer.weight.device)
    W = layer.weight.index_select(dim, index).clone().detach()
    if layer.bias is not None:
        if dim == 1:
            b = layer.bias.clone().detach()
        else:
            b = layer.bias[index].clone().detach()
    new_size = list(layer.weight.size())
    new_size[dim] = len(index)
    new_layer = nn.Linear(new_size[1], new_size[0], bias=layer.bias is not None).to(layer.weight.device)
    new_layer.weight.requires_grad = False
    new_layer.weight.copy_(W.contiguous())
    new_layer.weight.requires_grad = True
    if layer.bias is not None:
        new_layer.bias.requires_grad = False
        new_layer.bias.copy_(b.contiguous())
        new_layer.bias.requires_grad = True
    return new_layer

class CooperativeLinear(nn.Linear):
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        num_experts: int,
        fan_in_fan_out : bool = False, 
        use_entropy = True,
        sample_period = 1,
        dirichlet_prior = 1,
        var_loss_scale = 1e-2,
        use_averaging = True,
        averaging_factor = 0.9,
        kl_loss_weight = 1e-5,
        train_cooperative = True,
        device = 'cuda' if torch.cuda.is_available() else 'cpu',
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        
        self.num_experts = num_experts
        self.in_features = in_features
        self.out_features = out_features
        self.fan_in_fan_out = fan_in_fan_out
        self.use_entropy = use_entropy
        self.device = device
        self.sample_period = sample_period
        self.use_averaging = use_averaging
        self.kl_loss_weight = kl_loss_weight

        self.train_cooperative = train_cooperative

        self.wishart_df = out_features
        self.wishert_prior = torch.eye(out_features)
        self.dirichlet_prior = dirichlet_prior
        self.var_loss_scale = var_loss_scale
        
        #self.expert_weights_prior = nn.Parameter(-dirichlet_prior + 2*dirichlet_prior * torch.rand(num_experts))
        
        self.expert_weights_prior = nn.Parameter(torch.rand(num_experts))

        self.std_prior = nn.Parameter(torch.rand(out_features))

        #nn.init.uniform_(self.expert_weights_prior)
        try:
            nn.init.xavier_normal_(self.std_prior)
        except:
            pass
        self.averaging_factor = averaging_factor
        self.sample_counter = 0
        self.forward_counter = 0

    def gamma(self, v):
        return torch.lgamma(v).exp()
        
    def multivariate_reparameterization(self, mu, var2):
        # https://www.wikiwand.com/en/Multivariate_normal_distribution#Drawing_values_from_the_distribution
        sampler = MultivariateNormal(loc=torch.zeros(self.out_features).to(self.device), \
                                     covariance_matrix=torch.eye(self.out_features).to(self.device))
        all_vars = sampler.sample((self.num_experts, self.in_features)).to(self.device)
        L = torch.linalg.cholesky(var2).to(var2.dtype)
        varsum = torch.einsum('eio,op->eip', all_vars, L)
        return self.var_loss_scale * varsum
        
    def multivariate_kl(self, var):
        # https://statproofbook.github.io/P/mvn-kl.html
        # log-sum inequality - https://mat.hjg.com.ar/tic/img/lecture3.pdf
        return self.num_experts * 0.5 * (var.trace() - torch.log(var).trace() - self.out_features)
        
    def wishart_reparameterization(self, std):
        # http://sfb649.wiwi.hu-berlin.de/fedc_homepage/xplore/tutorials/mvahtmlnode40.html
        sampler = Wishart(df=self.wishart_df, scale_tril=torch.eye(self.out_features).to(self.device))

        sample = sampler._bartlett_sampling(torch.Size()).to(torch.float32)

        updated_var =  std @ sample.to(self.device) @ std.T
        updated_var = torch.diag(torch.clip(updated_var.diag(),min=eps)).to(updated_var.device).to(torch.float32)

        return updated_var
        
    def wishart_kl(self, std):
        var = (std @ std.T)
        var = torch.diag(var.diag()).to(var.device)
        return 0.5 * (-torch.log(var).trace()*self.wishart_df + var.trace()*self.wishart_df - self.wishart_df**2)

    def dirichlet_reparameterization(self, alpha2):
        # https://arxiv.org/pdf/1703.01488
        sampler = MultivariateNormal(loc=torch.zeros(self.num_experts).to(self.device), \
                                     covariance_matrix=torch.eye(self.num_experts).to(self.device))
        sample = sampler.sample().to(self.device)
        mu = torch.log(alpha2) - 1/self.num_experts * torch.log(alpha2).sum()
        sigma = torch.diag(1/alpha2 * (1 - 2/self.num_experts) + 1/(self.num_experts ** 2) * (1/alpha2).sum())
        return torch.linalg.cholesky(sigma) @ sample + mu
        
    def dirichlet_kl(self, alpha2):
        # https://statproofbook.github.io/P/dir-kl.html
        alpha1 = torch.tensor([self.dirichlet_prior]*self.num_experts).to(self.device)
        kld = torch.log(self.gamma(alpha2.sum())/self.gamma(alpha1.sum())) + (torch.log(self.gamma(alpha2)/self.gamma(alpha1))).sum() + \
              ((alpha2 - alpha1)*(torch.digamma(alpha2) - torch.digamma(alpha2.sum()))).sum()
        return kld

    def get_variational_loss(self):
        #print (self.training)
        if self.train_cooperative == True and self.training == True:
            #print ("KL Loss weight", self.kl_loss_weight)
            # kld of product of independent variables - http://www.math.tau.ac.il/~mansour/advanced-agt+ml/scribe5-lower-bound-MAB.pdf
            kl1 = self.dirichlet_kl(self.expert_weights_prior)
            kl2 = self.wishart_kl(torch.diag(self.std_prior))
            #kl2 = self.wishart_kl(self.std_prior)
            kl3 = self.multivariate_kl(self.gaussian_var_prior)
            
            if self.use_entropy == True:
                loss4 = self.calculate_entropy(self.expert_weights)
                return self.kl_loss_weight*(kl1 + kl2 + kl3), loss4
                #return loss4
            else:
                return self.kl_loss_weight * (kl1 + kl2 + kl3), torch.tensor(0)
                #return 0
        else:
            return 0, 0
        
    def calculate_entropy(self, expert_weights):
        #return (expert_weights * expert_weights.log()).sum()
        return (expert_weights ** 2).sum()

    def forward(self, x):       
        mu = cp(self.weight.data)
        if self.fan_in_fan_out == False:
            mu = mu.T

        if self.training == True:            
            if self.train_cooperative == True:  
                gaussian_var_prior = self.wishart_reparameterization(torch.diag(self.std_prior)) #*self.var_loss_scale
                self.gaussian_var_prior = gaussian_var_prior

                var = self.multivariate_reparameterization(mu, gaussian_var_prior)
                self.var = self.var_loss_scale * var
                updated_mu = self.var_loss_scale * var + mu

                updated_mu = torch.nan_to_num(updated_mu, nan=0.0)

                #expert_weights = self.dirichlet_reparameterization(nn.Sigmoid()(self.expert_weights_prior))

                expert_weights = self.dirichlet_reparameterization(self.expert_weights_prior)
                expert_weights = nn.Sigmoid()(expert_weights)
                expert_weights = expert_weights/expert_weights.sum()

                updated_mu = torch.einsum('i,ijk->jk', expert_weights, updated_mu)
                updated_mu = torch.nan_to_num(updated_mu, nan=0.0)
                self.expert_weights = expert_weights

                #print (x.device, self.updated_mu.device)                     
                if self.fan_in_fan_out == False:
                    #print (F.linear(x, mu.T, self.bias))
                    res = F.linear(x, updated_mu.T, self.bias) 
                else:
                    #print (F.linear(x, mu.T, self.bias))
                    res = F.linear(x, updated_mu, self.bias)
                
                #print (updated_mu.shape, self.weight.data.shape)

                if self.fan_in_fan_out == False:
                    self.weight.data.copy_(updated_mu.T)
                else:
                    self.weight.data.copy_(updated_mu)
            else:
                if self.fan_in_fan_out == False:
                    res = F.linear(x, mu.T, self.bias)
                else:
                    res = F.linear(x, mu, self.bias)

            self.forward_counter += 1
            
            return res
        else:
            if self.fan_in_fan_out == False:
                res = F.linear(x, mu.T, self.bias)
            else:
                res = F.linear(x, mu, self.bias)

            return res
        
    def eval(self):
        self.training = False

class CooperativeLinear_V1(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_experts: int,
        fan_in_fan_out : bool = False,
        log_variance_init = -10,
        single_variance = False,
        var_loss_scale = 1e-1,
        use_entropy = False,
        weight_normalization = 'Softmax',
        expert_weight_init = -1,
        inference_mixing_coeff = 1,
        sample_period = 1,
        **kwargs
    ):
        self.num_experts = num_experts
        self.in_features = in_features
        self.out_features = out_features
        self.fan_in_fan_out = fan_in_fan_out
        self.var_loss_scale = var_loss_scale
        self.weight_normalization = weight_normalization
        self.use_entropy = use_entropy

        nn.Linear.__init__(self, in_features, out_features, **kwargs)

        self.scale = 1 #nn.Parameter(torch.ones(1))
        self.cons_scale = 1
        self.single_variance = single_variance
        self.log_variance_init = log_variance_init
        if not single_variance:
            self.logvar = nn.Parameter((0.5 + torch.rand(num_experts,out_features)/2)*(log_variance_init))
        else:
            self.logvar = nn.Parameter((0.5 + torch.rand(out_features)/2)*(log_variance_init))
        self.expert_weights = nn.Parameter(-expert_weight_init + 2*expert_weight_init * torch.rand(num_experts))
        if weight_normalization == 'Softmax':
            self.weight_normalizer = nn.Softmax()
        else:
            self.weight_normalizer = nn.Sigmoid()
        self.last_var_param = None
        self.inference_mixing_coeff = inference_mixing_coeff
        self.var_param = None
        self.sample_counter = 0
        self.sample_period = sample_period

    def forward(self, x):
        if self.training:
            mu = cp(self.weight.data)

            if self.fan_in_fan_out == False:
                mu = mu.T
            if self.sample_counter == 0:
                self.sample_counter = (self.sample_counter + 1) % self.sample_period
                if self.single_variance:
                    var = torch.diag(self.logvar.exp() * self.scale* self.cons_scale).to(x.device)
                else:
                    var = torch.stack([torch.diag(i.exp()* self.scale) for i in self.logvar]).to(x.device)
                sampler = MultivariateNormal(torch.zeros(self.out_features).to(x.device), torch.eye(self.out_features).to(x.device))
                all_vars = sampler.sample((self.num_experts, self.in_features)).to(x.device)
                all_vars = all_vars @ var

                all_vars += mu
                expert_weights = self.weight_normalizer(self.expert_weights).to(x.device)
                expert_weights = expert_weights/expert_weights.sum()

                var_param = torch.einsum('i,ijk->jk', expert_weights, all_vars)
                self.var_param = torch.nan_to_num(var_param, nan=0.0)

            if self.fan_in_fan_out == False:
                res = F.linear(x, self.var_param.T, self.bias)
            else:
                res = F.linear(x, self.var_param, self.bias)
            if self.inference_mixing_coeff == 1:
                self.last_var_param = self.var_parama
            elif self.inference_mixing_coeff > 0 and self.inference_mixing_coeff < 1:
                if self.last_var_param is None:
                    self.last_var_param = self.var_param
                else:
                    self.last_var_param = self.var_param * self.inference_mixing_coeff + self.last_var_param * (1- self.inference_mixing_coeff)

            #if torch.isnan(var_param).max() == True:
            #    print ("variance", var)
            #    print ("param", var_param)
            #    print ("out", res)

            return res
        else:
            if self.last_var_param is not None:
                mu = self.last_var_param.to(x.device).T
            else:
                mu = self.weight.data

            if self.fan_in_fan_out == False:
                mu = mu.T

            if self.fan_in_fan_out == False:
                res = F.linear(x, mu.T, self.bias)
            else:
                res = F.linear(x, mu, self.bias)
            return res

    def eval(self):
        self.training = False

    def get_variational_loss(self):
        expert_weights = self.weight_normalizer(self.expert_weights)
        expert_weight_loss = expert_weights/expert_weights.sum()
        expert_logvar = self.logvar
        loss = ((2 * expert_logvar.sum() - expert_logvar.exp().square().sum())) * expert_weights.sum()
        if self.use_entropy:
            expert_weight_loss = (expert_weight_loss * expert_weight_loss.log() ).sum()
            loss += expert_weight_loss
        return self.var_loss_scale * loss


    # def get_variational_loss(self):
    #     expert_weights = self.weight_normalizer(self.expert_weights)
    #     expert_weight_loss = expert_weights/expert_weights.sum()
    #     loss = ((2 * self.logvar.sum() - self.logvar.exp().square().sum())) * expert_weights.sum()
    #     if self.use_entropy:
    #         expert_weight_loss = (expert_weight_loss * expert_weight_loss.log() ).sum()
    #         loss += expert_weight_loss
    #     return self.var_loss_scale * loss

class Conv1D(nn.Module):
    """
    1D-convolutional layer as defined by Radford et al. for OpenAI GPT (and also used in GPT-2).

    Basically works like a linear layer but the weights are transposed.

    Args:
        nf (`int`): The number of output features.
        nx (`int`): The number of input features.
    """

    def __init__(self, nf, nx):
        super().__init__()
        self.nf = nf
        self.weight = nn.Parameter(torch.empty(nx, nf))
        self.bias = nn.Parameter(torch.zeros(nf))
        nn.init.normal_(self.weight, std=0.02)

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        x = x.view(size_out)
        return x

class CooperativeConv1D(Conv1D):
    """
    1D-convolutional layer as defined by Radford et al. for OpenAI GPT (and also used in GPT-2).

    Basically works like a linear layer but the weights are transposed.

    Args:
        nf (`int`): The number of output features.
        nx (`int`): The number of input features.
    """

    def __init__(self, nf, nx,
                 num_experts = 4,
                 log_variance_init = -10,
                 single_variance = False,
                 var_loss_scale = 1e-7,
                 use_entropy = False,
                 weight_normalization = 'Sigmoid',
                 expert_weight_init = -1,
                inference_mixing_coeff = 1,):
        super().__init__(nx, nf)
        self.nf = nf
        self.weight = nn.Parameter(torch.empty(nx, nf))
        self.bias = nn.Parameter(torch.zeros(nf))
        self.var_loss_scale = var_loss_scale
        self.use_entropy = use_entropy
        self.weight_normalization = weight_normalization
        self.num_experts = num_experts
        self.single_variance = single_variance
        self.out_features = nf
        self.in_features = nx
        self.last_var_param = None
        self.inference_mixing_coeff = inference_mixing_coeff

        if not single_variance:
            self.logvar = nn.Parameter((0.5 + torch.rand(num_experts,nf)/2)*(log_variance_init))
        else:
            self.logvar = nn.Parameter((0.5 + torch.rand(nf)/2)*(log_variance_init))
        self.expert_weights = nn.Parameter(-expert_weight_init + 2*expert_weight_init * torch.rand(num_experts))
        if weight_normalization == 'Softmax':
            self.weight_normalizer = nn.Softmax()
        else:
            self.weight_normalizer = nn.Sigmoid()

        nn.init.normal_(self.weight, std=0.02)

    def forward(self, x):

        if self.training or (not self.training and (self.inference_mixing_coeff == -1)):
            mu = cp(self.weight.data)

            if self.fan_in_fan_out == False:
                mu = mu.T

            if self.single_variance:
                var = torch.diag(self.logvar.exp() * self.scale* self.cons_scale).to(x.device)
            else:
                var = torch.stack([torch.diag(i.exp()* self.scale) for i in self.logvar]).to(x.device)
            sampler = MultivariateNormal(torch.zeros(self.out_features).to(x.device), torch.eye(self.out_features).to(x.device))
            all_vars = sampler.sample((self.num_experts, self.in_features)).to(x.device)
            all_vars = all_vars @ var

            all_vars += mu
            expert_weights = self.weight_normalizer(self.expert_weights).to(x.device)
            expert_weights = expert_weights/expert_weights.sum()

            var_param = torch.einsum('i,ijk->jk', expert_weights, all_vars)
            var_param = torch.nan_to_num(var_param, nan=0.0)

            size_out = x.size()[:-1] + (self.nf,)
            res = torch.addmm(self.bias, x.view(-1, x.size(-1)), var_param)
            res = x.view(size_out)

            if self.inference_mixing_coeff == 1:
                self.last_var_param = var_param.to('cpu')
            elif self.inference_mixing_coeff > 0 and self.inference_mixing_coeff < 1:
                if self.last_var_param is None:
                    self.last_var_param = var_param.to('cpu')
                else:
                    self.last_var_param = var_param.to('cpu') * self.inference_mixing_coeff + self.last_var_param.to('cpu') * (1- self.inference_mixing_coeff)
            return res
        else:
            if self.last_var_param is not None:
                mu = self.last_var_param.to(x.device).T
            else:
                mu = self.weight.data

            size_out = x.size()[:-1] + (self.nf,)
            res = torch.addmm(self.bias, x.view(-1, x.size(-1)), mu)
            res = x.view(size_out)
            return res

    def get_variational_loss(self):
        expert_weights = self.weight_normalizer(self.expert_weights)
        expert_weight_loss = expert_weights/expert_weights.sum()
        loss = ((2 * self.logvar.sum() - self.logvar.exp().square().sum())) * expert_weights.sum()
        if self.use_entropy:
            expert_weight_loss = (expert_weight_loss * expert_weight_loss.log() ).sum()
            loss += expert_weight_loss
        return self.var_loss_scale * loss


def prune_conv1d_layer(layer: Conv1D, index: torch.LongTensor, dim: int = 1) -> Conv1D:
    """
    Prune a Conv1D layer to keep only entries in index. A Conv1D work as a Linear layer (see e.g. BERT) but the weights
    are transposed.

    Used to remove heads.

    Args:
        layer ([`~pytorch_utils.Conv1D`]): The layer to prune.
        index (`torch.LongTensor`): The indices to keep in the layer.
        dim (`int`, *optional*, defaults to 1): The dimension on which to keep the indices.

    Returns:
        [`~pytorch_utils.Conv1D`]: The pruned layer as a new layer with `requires_grad=True`.
    """
    index = index.to(layer.weight.device)
    W = layer.weight.index_select(dim, index).clone().detach()
    if dim == 0:
        b = layer.bias.clone().detach()
    else:
        b = layer.bias[index].clone().detach()
    new_size = list(layer.weight.size())
    new_size[dim] = len(index)
    new_layer = Conv1D(new_size[1], new_size[0]).to(layer.weight.device)
    new_layer.weight.requires_grad = False
    new_layer.weight.copy_(W.contiguous())
    new_layer.weight.requires_grad = True
    new_layer.bias.requires_grad = False
    new_layer.bias.copy_(b.contiguous())
    new_layer.bias.requires_grad = True
    return new_layer


def prune_layer(
    layer: Union[nn.Linear, Conv1D], index: torch.LongTensor, dim: Optional[int] = None
) -> Union[nn.Linear, Conv1D]:
    """
    Prune a Conv1D or linear layer to keep only entries in index.

    Used to remove heads.

    Args:
        layer (`Union[torch.nn.Linear, Conv1D]`): The layer to prune.
        index (`torch.LongTensor`): The indices to keep in the layer.
        dim (`int`, *optional*): The dimension on which to keep the indices.

    Returns:
        `torch.nn.Linear` or [`~pytorch_utils.Conv1D`]: The pruned layer as a new layer with `requires_grad=True`.
    """
    if isinstance(layer, nn.Linear):
        return prune_linear_layer(layer, index, dim=0 if dim is None else dim)
    elif isinstance(layer, Conv1D):
        return prune_conv1d_layer(layer, index, dim=1 if dim is None else dim)
    else:
        raise ValueError(f"Can't prune layer of class {layer.__class__}")


def apply_chunking_to_forward(
    forward_fn: Callable[..., torch.Tensor], chunk_size: int, chunk_dim: int, *input_tensors
) -> torch.Tensor:
    """
    This function chunks the `input_tensors` into smaller input tensor parts of size `chunk_size` over the dimension
    `chunk_dim`. It then applies a layer `forward_fn` to each chunk independently to save memory.

    If the `forward_fn` is independent across the `chunk_dim` this function will yield the same result as directly
    applying `forward_fn` to `input_tensors`.

    Args:
        forward_fn (`Callable[..., torch.Tensor]`):
            The forward function of the model.
        chunk_size (`int`):
            The chunk size of a chunked tensor: `num_chunks = len(input_tensors[0]) / chunk_size`.
        chunk_dim (`int`):
            The dimension over which the `input_tensors` should be chunked.
        input_tensors (`Tuple[torch.Tensor]`):
            The input tensors of `forward_fn` which will be chunked

    Returns:
        `torch.Tensor`: A tensor with the same shape as the `forward_fn` would have given if applied`.


    Examples:

    ```python
    # rename the usual forward() fn to forward_chunk()
    def forward_chunk(self, hidden_states):
        hidden_states = self.decoder(hidden_states)
        return hidden_states


    # implement a chunked forward function
    def forward(self, hidden_states):
        return apply_chunking_to_forward(self.forward_chunk, self.chunk_size_lm_head, self.seq_len_dim, hidden_states)
    ```"""

    assert len(input_tensors) > 0, f"{input_tensors} has to be a tuple/list of tensors"

    # inspect.signature exist since python 3.5 and is a python method -> no problem with backward compatibility
    num_args_in_forward_chunk_fn = len(inspect.signature(forward_fn).parameters)
    if num_args_in_forward_chunk_fn != len(input_tensors):
        raise ValueError(
            f"forward_chunk_fn expects {num_args_in_forward_chunk_fn} arguments, but only {len(input_tensors)} input "
            "tensors are given"
        )

    if chunk_size > 0:
        tensor_shape = input_tensors[0].shape[chunk_dim]
        for input_tensor in input_tensors:
            if input_tensor.shape[chunk_dim] != tensor_shape:
                raise ValueError(
                    f"All input tenors have to be of the same shape: {tensor_shape}, "
                    f"found shape {input_tensor.shape[chunk_dim]}"
                )

        if input_tensors[0].shape[chunk_dim] % chunk_size != 0:
            raise ValueError(
                f"The dimension to be chunked {input_tensors[0].shape[chunk_dim]} has to be a multiple of the chunk "
                f"size {chunk_size}"
            )

        num_chunks = input_tensors[0].shape[chunk_dim] // chunk_size

        # chunk input tensor into tuples
        input_tensors_chunks = tuple(input_tensor.chunk(num_chunks, dim=chunk_dim) for input_tensor in input_tensors)
        # apply forward fn to every tuple
        output_chunks = tuple(forward_fn(*input_tensors_chunk) for input_tensors_chunk in zip(*input_tensors_chunks))
        # concatenate output at same dimension
        return torch.cat(output_chunks, dim=chunk_dim)

    return forward_fn(*input_tensors)


def find_pruneable_heads_and_indices(
    heads: List[int], n_heads: int, head_size: int, already_pruned_heads: Set[int]
) -> Tuple[Set[int], torch.LongTensor]:
    """
    Finds the heads and their indices taking `already_pruned_heads` into account.

    Args:
        heads (`List[int]`): List of the indices of heads to prune.
        n_heads (`int`): The number of heads in the model.
        head_size (`int`): The size of each head.
        already_pruned_heads (`Set[int]`): A set of already pruned heads.

    Returns:
        `Tuple[Set[int], torch.LongTensor]`: A tuple with the indices of heads to prune taking `already_pruned_heads`
        into account and the indices of rows/columns to keep in the layer weight.
    """
    mask = torch.ones(n_heads, head_size)
    heads = set(heads) - already_pruned_heads  # Convert to set and remove already pruned heads
    for head in heads:
        # Compute how many pruned heads are before the head and move the index accordingly
        head = head - sum(1 if h < head else 0 for h in already_pruned_heads)
        mask[head] = 0
    mask = mask.view(-1).contiguous().eq(1)
    index: torch.LongTensor = torch.arange(len(mask))[mask].long()
    return heads, index


def meshgrid(
    *tensors: Union[torch.Tensor, List[torch.Tensor]], indexing: Optional[str] = None
) -> Tuple[torch.Tensor, ...]:
    """
    Wrapper around torch.meshgrid to avoid warning messages about the introduced `indexing` argument.

    Reference: https://pytorch.org/docs/1.13/generated/torch.meshgrid.html
    """
    return torch.meshgrid(*tensors, indexing=indexing)


def id_tensor_storage(tensor: torch.Tensor) -> Tuple[torch.device, int, int]:
    """
    Unique identifier to a tensor storage. Multiple different tensors can share the same underlying storage. For
    example, "meta" tensors all share the same storage, and thus their identifier will all be equal. This identifier is
    guaranteed to be unique and constant for this tensor's storage during its lifetime. Two tensor storages with
    non-overlapping lifetimes may have the same id.
    """
    if tensor.device.type == "xla" and is_torch_xla_available():
        # NOTE: xla tensors dont have storage
        # use some other unique id to distinguish.
        # this is a XLA tensor, it must be created using torch_xla's
        # device. So the following import is safe:
        import torch_xla

        unique_id = torch_xla._XLAC._xla_get_tensor_id(tensor)
    else:
        unique_id = storage_ptr(tensor)

    return tensor.device, unique_id, storage_size(tensor)
