# -*- coding: utf-8 -*-
"""
Sparsity functions in PyTorch.
Copyright (c) 2024 Junbo Zhao
"""

from typing import Tuple
import sys
import math

import numpy as np
import torch
from torch import autograd, nn, Tensor
import torch.nn.functional as F
import torch.utils.checkpoint


def unstructured_weight_prune(weight: Tensor, ratio: float) -> Tensor:
    """
    Unstructured weight pruning based on the absolute value with a predefined ratio.

    Args:
        weight (Tensor): The weight to be pruned.
        ratio (float): The pruning ratio.

    Returns:
        Tensor: The generated mask.

    Examples::
        >>> weight = torch.randn((3, 16, 16))
        >>> mask = unstructured_weight_prune(weight, ratio = 0.5)
    """

    if ratio == 0.:
        return torch.ones_like(weight).type_as(weight)

    num_weight = weight.numel()
    num_prune = int(num_weight * ratio)
    abs_weight = weight.abs()
    threshold = torch.topk(abs_weight.view(-1), num_prune, largest = False)[0].max()
    mask = torch.gt(abs_weight, threshold).type_as(weight)
    return mask


def get_sparse_mask(weight: Tensor, ratio: float) -> Tuple[Tensor, Tensor]:
    """
    Calculate the sparsity mask.

    Args:
        weight (Tensor): The weight to be pruned.
        ratio (float): Unstructured pruning ratio.

    Returns:
        Tensor: The N:M pruned weight.
        Tensor: The corresponding N:M pruning binary mask.

    Examples::
        >>> module = nn.Conv2d(64, 128, (3, 3))
        >>> pruned_weight, mask = get_sparse_mask(module.weight, ratio = 0.5)
        >>> actual_sparsity = 1 - mask.sum().item() / mask.numel()
    """

    # PRE-DEFINED HYPER-PARAMETERS
    weight_unit = 8
    block_w = 8
    block_h = 8
    sparsity_option = [0, 1, 2, 4, 8]
    # The possible preserved weights of the block
    sparsity_array = np.array([0, 8, 16, 32, 64])

    # Step 1: Generate the mask for unstructured pruning
    # The scheme utilises this mask to determine N:M pruning settings
    unstructured_mask = unstructured_weight_prune(weight, ratio = ratio)
    # Apply the mask to get the pre-pruned weight
    unstructured_weight = weight * unstructured_mask

    # Step 2: Record the original shape of the weight
    if len(weight.shape) == 4:  # The weight belongs to a 2D-convolution
        C_out, C_in, H, W = weight.shape
        reshaped_W = C_in * H * W
    elif len(weight.shape) == 2:  # The weight belongs to a linear layer
        C_out, C_in = weight.shape
        reshaped_W = C_in
    else:  # The weight is invalid
        raise ValueError("Invalid weight shape: {}".format(weight.shape))
    reshaped_H = C_out

    # Step 3: Reshape the weight to a Tensor with shape (C_out, X)
    weight_mtx = weight.reshape(reshaped_H, reshaped_W)
    unstructured_mask_mtx = unstructured_mask.reshape(reshaped_H, reshaped_W)
    pruned_unstructured_weight = unstructured_weight.reshape(reshaped_H, reshaped_W)

    # Step 4: Pad the reshaped matrix to integral multiple of the block size
    W_pad = block_w - reshaped_W % block_w
    H_pad = block_h - reshaped_H % block_h
    weight_padded = F.pad(weight_mtx.unsqueeze(0), (0, W_pad, 0, H_pad), value = 0.).squeeze(0)
    unstructured_mask_padded = F.pad(
        unstructured_mask_mtx.unsqueeze(0), (0, W_pad, 0, H_pad), value = 0.).squeeze(0)

    # Step 5: Generate the N:M pruning mask
    mask = torch.zeros_like(weight_padded)  # Initialize the mask
    H_block_num = int((reshaped_H + H_pad) / block_h)
    W_block_num = int((reshaped_W + W_pad) / block_w)

    for i in range(H_block_num):
        for j in range(W_block_num):

            h_left, h_right = i * block_h, (i + 1) * block_h
            w_left, w_right = j * block_w, (j + 1) * block_w

            # Get the weight and mask patches
            weight_sub_mtx = weight_padded[h_left : h_right, w_left : w_right]
            unstructured_mask_sub_mtx = unstructured_mask_padded[h_left : h_right, w_left : w_right]

            # Point 1: Get the best sparsity choice
            preserved_num = unstructured_mask_sub_mtx.sum()
            sparsity_choice_idx = np.argmin(np.abs(preserved_num.item() - sparsity_array))
            sparsity_choice = sparsity_option[sparsity_choice_idx] # The selected N for this patch

            # Point 2: Generate the mask
            def get_n_m_sparse_mask(transpose: bool = False) -> Tuple[Tensor, float]:
                """
                Calculate the sparse mask of the patch.

                Args:
                    transpose (bool): Whether to prune in the reverse direction.

                Returns:
                    Tensor: The generated mask.
                    float: The similarity between the newly generated mask and unstructured mask.
                """
                frac_weight = weight_sub_mtx.T if transpose else weight_sub_mtx
                frac_weight = torch.abs(frac_weight.reshape(-1, weight_unit))
                _, sorted_indices = torch.sort(frac_weight, descending = True)
                sub_mask = torch.zeros_like(frac_weight)
                for k, indices in enumerate(sorted_indices):
                    sub_mask[k][indices[:sparsity_choice]] = 1.
                sub_mask = sub_mask.reshape(block_h, block_w)
                sub_mask = sub_mask.T if transpose else sub_mask
                confidence = (sub_mask == unstructured_mask_sub_mtx).sum().item() / sub_mask.numel()
                return sub_mask, confidence

            sub_mask_1, confidence_1 = get_n_m_sparse_mask(False)
            sub_mask_2, confidence_2 = get_n_m_sparse_mask(True)
            # Select the pruning scheme that has larger similarity with the unstructured pruning
            sub_mask = sub_mask_1 if confidence_1 > confidence_2 else sub_mask_2

            # Update the mask
            mask[h_left : h_right, w_left : w_right] = sub_mask

    # Step 7: Recover the original shape
    mask = mask[: reshaped_H, : reshaped_W]
    mask = mask.reshape(C_out, C_in, H, W) if len(weight.shape) == 4 else mask.reshape(C_out, C_in)
    pruned_weight = mask * weight

    return pruned_weight, mask


class SparseStrategy(autograd.Function):
    """
    The pruning strategy.
    """

    @staticmethod
    def forward(
        ctx,
        weight: Tensor,
        ratio: float,
        mask: Tensor = None,
        update: bool = False
        ) -> Tuple[Tensor, Tensor]:
        """
        Prune the weight in the forward phase.

        Args:
            weight (Tensor): The weight to be pruned.
            ratio (float): The pruning ratio.
            mask (Tensor): The previously generated mask. If not given, a new mask
                           will be generated and utilised. Default: `None`.
            update (bool): Whether to update the mask. Default: `False`.

        Returns:
            Tensor: The weight pruned with the previous mask.
            Tensor: The mask for the backward phase.
        """

        if mask and not update:
            pruned_weight = mask * weight
            return pruned_weight, mask

        return get_sparse_mask(weight, ratio)

    @staticmethod
    def backward(ctx, grad_output: Tensor, grad_mask: Tensor = None):
        """
        The backward function.
        """
        return grad_output, None, None, None, None, None


class SparseConv2d(nn.Conv2d):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride = 1,
        padding = 0,
        dilation = 1,
        groups = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        sparsity_rate: float = 0.5,
        mask_update_every_step_num: int = 2000
        **kwargs
    ) -> None:
        super(SparseConv2d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
            **kwargs
        )

        self.sparsity_rate = sparsity_rate
        self.mask_update_every_step_num = 2000

        self.current_step_num = 0
        self.current_epoch = 0

        self.mask = None

    def get_sparse_weights(self) -> Tensor:
        """
        Get the sparse weight.
        """
        update = self.current_step_num % self.mask_update_every_step_num == 0
        weight, self.mask = SparseStrategy.apply(self.weight, self.sparsity_rate, self.mask, update)
        return weight

    def forward(self, input: Tensor, current_step_num: int = 0, current_epoch: int = 0) -> Tensor:
        """
        Feed-forward phase.

        Args:
            input (Tensor): The input features.
            current_step_num (int): Current step number. Default: 0.
            current_epoch (int): Current epoch number. Default: 0.

        Returns:
            Tensor: The output feature.
        """
        # Record the statistics
        self.current_step_num = current_step_num
        self.current_epoch = current_epoch
        # Get the sparse weight
        w = self.get_sparse_weights()
        return F.conv2d(input, w, self.bias, self.stride, self.padding, self.dilation, self.groups)

    @property
    def actual_sparse_ratio(self) -> float:
        """
        The current actual sparse ratio.

        Returns:
            float: Current sparsity ratio.
        """
        return 1. - sum(self.mask).sum().item() / self.mask.numel()

    def __return_sparse_weights__(self) -> Tensor:
        """
        Get the sparse weight.

        Returns:
            Tensor: The pruned weight.
        """
        return self.get_sparse_weights() 


class SparseLinear(nn.Linear):

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        sparsity_rate: float = 0.5,
        **kwargs
    ) -> None:
        super(SparseLinear, self).__init__(in_features, out_features, bias = bias)
        self.sparsity_rate = sparsity_rate
        self.mask = None

    def get_sparse_weights(self) -> Tensor:
        weight, self.mask = SparseStrategy.apply(self.weight, self.sparsity_rate, self.mask)
        return weight

    def forward(self, x: Tensor, current_step_num: int = 0, current_epoch: int = 0) -> Tensor:
        w = self.get_sparse_weights()
        x = F.linear(x, w)
        return x

    def __return_sparse_weights__(self) -> Tensor:
        return self.get_sparse_weights()

    @property
    def actual_sparse_ratio(self) -> float:
        return 1. - sum(self.mask).sum().item() / self.mask.numel()


def test_sparse_linear() -> None:
    device = torch.device("cuda")
    module = SparseLinear(32, 32, True, sparsity_rate = 0.8).to(device)
    input = torch.randn((4, 32)).to(device)
    output = module(input)
    loss = output.sum()
    loss.backward()
    print(module.actual_sparse_ratio)


def test_sparse_conv() -> None:
    device = torch.device("cuda")
    module = SparseConv2d(64, 128, (3, 3), sparsity_rate = 0.6).to(device)
    input = torch.randn((64, 32, 32)).to(device)
    output = module(input)
    loss = output.sum()
    loss.backward()
    print(module.actual_sparse_ratio)


if __name__ == "__main__":
    test_sparse_linear()
    test_sparse_conv()
