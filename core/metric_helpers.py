# -*- coding: utf-8 -*-

import torch
from typing import Tuple
from functorch import jacrev, vmap

""" Helpers for Metrics.
"""

@torch.jit.script
def batched_matrix_vector_prod(u: torch.Tensor, J: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """ Compute product u @ J.t() @ v
    """
    return torch.bmm(
        torch.transpose(
            torch.bmm(J, v), 
            1,
            2
        ), u
    ).squeeze(-1).squeeze(-1) # workaround to avoid squeezing batch dimension


@torch.jit.script
def spectral_norm_power_iteration(J: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """ Compute one iteration of the power method to estimate
        the largest singular value of J
    """
    u = torch.bmm(J, v)
    u /= torch.norm(u, p=2, dim=1).unsqueeze(-1)
    v = torch.matmul(torch.transpose(J, 1, 2), u)
    v /= torch.norm(v, p=2, dim=1).unsqueeze(-1)
    return (u, v)


@torch.jit.script
def spectral_norm(J: torch.Tensor, num_steps: int, atol: float = 1e-2) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """ Compute the spectral norm of @J using @num_steps iterations
        of the power method.
        
        @return u (torch.Tensor): left-singular vector
        @return sigma (torch.Tensor): largest singular value
        @return v (torch.Tensor): right-singular vector
    """
    device = J.device
    dtype = J.dtype
    J = J.view(J.shape[0], -1, J.shape[2])
    nbatches, nindims, noutdims = J.shape[0], J.shape[1], J.shape[2]
    
    batch_indices = torch.arange(nbatches, dtype=torch.long, device=device)
    atol = torch.full((1,), fill_value=atol, device=device, dtype=dtype)
    
    v = torch.randn(nbatches, noutdims, 1, device=device, dtype=dtype)
    v /= torch.norm(v, p=2, dim=1).unsqueeze(-1)
    sigma_prev = torch.zeros(nbatches, dtype=dtype, device=device)
    u_prev = torch.zeros((nbatches, nindims), dtype=dtype, device=device)
    v_prev = torch.zeros((nbatches, noutdims), dtype=dtype, device=device)
    
    for i in range(num_steps):
        u, v = spectral_norm_power_iteration(J, v)
        sigma = batched_matrix_vector_prod(u, J, v)
        diff_indices = torch.ge(
            torch.abs(sigma.squeeze() - sigma_prev[batch_indices]), atol
        )

        if not torch.any(diff_indices):
            break
        
        sigma_prev[batch_indices[diff_indices]] = sigma[diff_indices]
        u_prev[batch_indices[diff_indices]] = u[diff_indices].squeeze(-1)
        v_prev[batch_indices[diff_indices]] = v[diff_indices].squeeze(-1)
        u = u[diff_indices]
        v = v[diff_indices]
        J = J[diff_indices]
        batch_indices = batch_indices[diff_indices]
        
    return u_prev.squeeze(), sigma_prev, v_prev.squeeze()


@torch.jit.script
def accuracy(output: torch.Tensor, targets: torch.Tensor, k: int=1) -> torch.Tensor:
    """ Compute top-K 0/1 loss, without averaging along the batch dimension
    """
    repeat_factor = output.shape[0] // targets.shape[0]
    prediction = output.topk(k, dim=1)[1].squeeze()
    acc = torch.eq(prediction, targets.repeat_interleave(repeat_factor))
    return acc


@torch.jit.script
def confidence(scores: torch.Tensor, targets: torch.Tensor, nclasses: int) -> torch.Tensor:
    """ Compute the distance of the softmax @scores from one-hot encoded @targets
    """
    repeat_factor = scores.shape[0] // targets.shape[0]
    one_hot = torch.nn.functional.one_hot(targets.repeat_interleave(repeat_factor), num_classes=nclasses)
    confidence = torch.norm(one_hot - scores, dim=-1, p=2)
    return confidence


""" Normalization and distances
"""

@torch.jit.script
def batch_normalize(x: torch.Tensor, batch_size: int) -> torch.Tensor:
    """ Normalize tensor @x of shape (N, *) by dividing each entry x[i, ...]
        by its Frobenius norm.
    """
    x = x.reshape(batch_size, -1)
    x_norm = torch.norm(x, p=2, dim=1, keepdim=True)
    return x / x_norm


""" Jacobian computation
"""

def get_jacobian_fn(model, nsamples, nclasses, bigmem=False, target_logit_only=False, normalization=None):
    """Wrapper to select Jacobian computation algorithm
    """
    scalar_output = target_logit_only or (normalization == "crossentropy")
    if target_logit_only and normalization == "crossentropy":
        raise ValueError("Unsupported combination: target logit only and crossentropy normalization.")
    elif target_logit_only:
        jacobian = jacobian_target_only
    elif normalization == "crossentropy":
        jacobian = jacobian_cross_entropy
    elif bigmem:
        jacobian = jacobian_big_mem
    else:
        jacobian = jacobian_low_mem
    
    def tile_input(x, bigmem=False):
        if bigmem and not scalar_output:
            tile_shape = (nclasses,) + (1,) * len(x.shape[1:])
            return x.repeat(tile_shape)
        else:
            return x
    
    device = next(model.parameters()).device
    if normalization == "logsoftmax":
        criterion = torch.nn.LogSoftmax(dim=1).to(device)
    elif normalization == "softmax":
        criterion = torch.nn.Softmax(dim=1).to(device)
    elif normalization == "crossentropy":
        criterion = torch.nn.CrossEntropyLoss(reduction='none').to(device)
    else:
        criterion = None
    
    if criterion is None:
        model_forward = model
    elif normalization == "crossentropy":
        def model_forward(x, targets):
            out = model(x)
            repeat_factor = out.shape[0] // targets.shape[0]
            return criterion(out, targets.repeat_interleave(repeat_factor))
    else:
        def model_forward(x):
            out = model(x)
            return criterion(out)
    
    def jacobian_fn(x, targets=None):
        x = tile_input(x, bigmem)
        x.requires_grad_(True)
        
        if targets is None:
            output = model_forward(x)
            j = jacobian(x, output, nsamples, nclasses)
        elif target_logit_only:
            output = model_forward(x)
            j = jacobian(x, output, targets, nsamples)
        else:
            output = model_forward(x, targets)
            j = jacobian(x, output, nsamples, nclasses)
        x.grad = None
        return j
    
    return jacobian_fn


@torch.jit.script
def jacobian_cross_entropy(x: torch.Tensor, predictions: torch.Tensor, nsamples: int, nclasses: int) -> torch.Tensor:
    """ Compute the Jacobian of @logits w.r.t. @input.
        
        Note: @x_in should track gradient computation before @logits
              is computed, otherwise this method will fail. @x should
              store a gradient_fn corresponding to the function used to
              produce @logits.
        
        Params:
            @x : 4D Tensor Batch of inputs with .grad attribute populated 
                 according to @logits
            @predictions: 1D Tensor Batch of network outputs at @x
            
        Return:
            Jacobian: Batch-indexed 2D torch.Tensor of shape (N,*, K).
            where N is the batch dimension, D is the (flattened) input
            space dimension, and K is the number of output dimensions
            of the network.
    """
    x.retain_grad()
    predictions.backward(gradient=torch.ones_like(predictions), retain_graph=True)
    jacobian = x.grad.data.view(nsamples, -1)
    
    return jacobian


@torch.jit.script
def jacobian_big_mem(x: torch.Tensor, logits: torch.Tensor, nsamples: int, nclasses: int) -> torch.Tensor:
    """ Compute the Jacobian of @logits w.r.t. @input.
        
        Note: @x_in should track gradient computation before @logits
              is computed, otherwise this method will fail. @x should
              store a gradient_fn corresponding to the function used to
              produce @logits.
        
        Params:
            @x : 4D Tensor Batch of inputs with .grad attribute populated 
                 according to @logits
            @logits: 2D Tensor Batch of network outputs at @x
            
        Return:
            Jacobian: Batch-indexed 2D torch.Tensor of shape (N,*, K).
            where N is the batch dimension, D is the (flattened) input
            space dimension, and K is the number of output dimensions
            of the network.
    """
    x.retain_grad()
    indexing_mask = torch.eye(nclasses, device=x.device).repeat((nsamples,1))
    
    logits.backward(gradient=indexing_mask, retain_graph=True)
    jacobian = x.grad.data.view(nsamples, nclasses, -1).transpose(1,2)
    
    return jacobian


@torch.jit.script
def jacobian_low_mem(x: torch.Tensor, logits: torch.Tensor, nsamples: int, nclasses: int) -> torch.Tensor:
    """ Compute the Jacobian of @logits w.r.t. @input.
        
        Note: @x_in should track gradient computation before @logits
              is computed, otherwise this method will fail. @x should
              store a gradient_fn corresponding to the function used to
              produce @logits.
        
        Params:
            @x : 4D Tensor Batch of inputs with .grad attribute populated 
                 according to @logits
            @logits: 2D Tensor Batch of network outputs at @x
            
        Return:
            Jacobian: Jacobian: Batch-indexed 2D torch.Tensor of shape (N,*, K).
            where N is the batch dimension, D is the (flattened) input
            space dimension, and K is the number of output dimensions
            of the network.
    """
    x.retain_grad()
    jacobian = torch.zeros(
        x.shape + (nclasses,), dtype=x.dtype, device=x.device
    )
    indexing_mask = torch.zeros_like(logits)
    indexing_mask[:, 0] = 1.
    
    for dim in range(nclasses):
        logits.backward(gradient=indexing_mask, retain_graph=True)
        jacobian[..., -dim] = x.grad.data
        x.grad.data.zero_()
        indexing_mask = torch.roll(indexing_mask, shifts=1, dims=1)
    
    return jacobian


@torch.jit.script
def jacobian_target_only(x: torch.Tensor, logits: torch.Tensor, targets: torch.Tensor, nsamples: int) -> torch.Tensor:
    """ Compute the Jacobian of @logits[targets] w.r.t. @input.
        
        Note: @x_in should track gradient computation before @logits
              is computed, otherwise this method will fail. @x should
              store a gradient_fn corresponding to the function used to
              produce @logits.
        
        Params:
            @x : 4D Tensor Batch of inputs with .grad attribute populated 
                 according to @logits
            @logits: 2D Tensor Batch of network outputs at @x
            @targets: 1D Tensor: Batch of logit indices, specifying which logit
                      to use for computing the Jacobian.
            
        Return:
            Jacobian: Jacobian: Batch-indexed 2D torch.Tensor of shape (N,D).
            where N is the batch dimension, D is the (flattened) input
            space dimension.
    """
    x.retain_grad()
    jacobian = torch.zeros_like(x)
    indices = torch.arange(nsamples, device=x.device)
    indexing_mask = torch.zeros_like(logits)
    repeat_factor = logits.shape[0] // targets.shape[0]
    indexing_mask[indices, targets.repeat_interleave(repeat_factor)] = 1.
    
    logits.backward(gradient=indexing_mask, retain_graph=True)
    jacobian = x.grad.data.view(nsamples, -1)
    return jacobian

