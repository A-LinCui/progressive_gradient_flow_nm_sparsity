""" Sparsity functions in PyTorch

A PyTorch implement of Sparse Linear, Sparse Three Linear and Sparse Conv 2D.

The implementation has MdGf-Linear, MdGf-Exponential along with SdGf-Stepwise, SdGf-Geometric.

"""

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


def get_sparse_mask(weight: Tensor, ratio: float, **kwargs):
    """
    # Calculate the sparsity mask.

    The $i^{th}$ backpropagation and $(i+1)^{th}$ feed-forward share the same mask.

    Args:
        weight (Tensor): The weight to be pruned.
        ratio (float): Unstructured pruning ratio.

    Returns:
        Tensor: The N:M pruned weight.
        Tensor: The corresponding N:M pruning binary mask.

    Examples::
        >>> module = nn.Conv2d(64, 128, (3, 3)).to("cuda")
        >>> pruned_weight, mask = get_sparse_mask(module.weight, ratio = 0.5)
        >>> actual_sparsity = 1 - mask.sum().item() / mask.numel()
    """

    # PRE-DEFINED HYPER-PARAMETERS
    weight_unit = 8
    block_w = 2 * weight_unit
    block_h = 16
    block_size = block_w * block_h
    sparsity_option = [0, 1, 2, 4, 8]
    # The possible preserved weights of the block
    sparsity_array = np.array(sparsity_option) * 2 * block_h

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
    weight_mtx = weight.clone().reshape(reshaped_H, reshaped_W)
    unstructured_mask_mtx = unstructured_mask.clone().reshape(reshaped_H, reshaped_W)
    pruned_unstructured_weight = unstructured_weight.clone().reshape(reshaped_H, reshaped_W)

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
    # sp_opt_mtx = torch.zeros((H_block_num, W_block_num)) - 1

    for i in range(H_block_num):
        for j in range(W_block_num):

            h_left, h_right = i * block_h, (i + 1) * block_h
            w_left, w_right = j * block_w, (j + 1) * block_w

            # Divide the original weight and mask into different patches
            weight_sub_mtx = weight_padded[h_left : h_right, w_left : w_right]
            unstructured_mask_sub_mtx = unstructured_mask_padded[h_left : h_right, w_left : w_right]

            # Point 1: Get the best sparsity choice
            preserved_num = unstructured_mask_sub_mtx.sum()
            sparsity_choice_idx = np.argmin(np.abs(preserved_num.item() - sparsity_array))
            sparsity_choice = sparsity_option[sparsity_choice_idx]
            # sp_opt_mtx[i, j] = sparsity_choice

            # Point 2: Generate the mask
            def get_n_m_sparse_mask(transpose: bool = False):
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
            sub_mask = sub_mask_1 if confidence_1 > confidence_2 else sub_mask_2
            # print(confidence_1, confidence_2)
            mask[h_left : h_right, w_left : w_right] = sub_mask

    # Step 7: Recover the original shape
    mask = mask[: reshaped_H, : reshaped_W].clone()
    mask = mask.reshape(C_out, C_in, H, W) if len(weight.shape) == 4 else mask.reshaped(C_out, C_in)
    pruned_weight = mask * weight

    return pruned_weight, mask


weight = torch.randn((128, 64, 3, 3)).to("cuda")
ratio = 0.8
w, m = get_sparse_mask(weight, ratio)
print(m.sum() / m.numel())
import ipdb
ipdb.set_trace()

def get_sparse_mask(weight, N, M, sparsity_rate=0.0, isconv=False, sparse_dim = 0):

    length = weight.numel()
        
    if sparsity_rate > 0.0:
        num_sparse = int((1.0 - sparsity_rate) * length)
        M = length
        N = num_sparse
    
    group = int(length/M)

    if(isconv == False):
        if(sparse_dim == 1):        ## Column wise N:M sparse
            weight_temp = torch.transpose(weight,0,1).detach().abs().reshape(group, M)
            index = torch.argsort(weight_temp, dim=1)[:, :int(M-N)]
            w_b = torch.ones(weight_temp.shape, device=weight_temp.device)
            w_b = w_b.scatter(dim=1, index=index, value=0)
            w_b = torch.t(w_b.reshape(weight.shape[1],weight.shape[0]))
        else:                       ## Row-wise N:M sparse
            weight_temp = weight.detach().abs().reshape(group, M)
            index = torch.argsort(weight_temp, dim=1)[:, :int(M-N)]

            w_b = torch.ones(weight_temp.shape, device=weight_temp.device)
            w_b = w_b.scatter_(dim=1, index=index, value=0).reshape(weight.shape)
    else:
        weight_temp = weight.detach().abs().permute(0,2,3,1).reshape(group, M)
        index = torch.argsort(weight_temp, dim=1)[:, :int(M-N)]

        w_b = torch.ones(weight_temp.shape, device=weight_temp.device)
        w_b = w_b.scatter_(dim=1, index=index, value=0).reshape(weight.permute(0,2,3,1).shape)
        w_b = w_b.permute(0,3,1,2)

    return w_b

def get_decay_config(sparseConfig):
    structure_decay_type =  sparseConfig.structure_decay_type.lower()
    N = sparseConfig.n_sparsity
    M = sparseConfig.m_sparsity
    if(sparseConfig.structure_decay_flag == True):
        if(structure_decay_type == "custom"):
            if(sparseConfig.structure_decay_config is not None):
                structure_decay_config = sparseConfig.structure_decay_config
            else:
                print("For custom structured sparse decay type, please also specify structure-decay-config. Format : [N1:M1,N2:M2,N3:M3,N4:M4]")
                sys.exit(1) 
        elif(structure_decay_type == "sparsify"):
            num_sparse_frames = int(math.log2(M/N)) + 1
            sparse_frame_size = (sparseConfig.total_epochs - sparseConfig.dense_epochs - sparseConfig.fine_tune_epochs) / num_sparse_frames
            structure_decay_config = dict()
            for i in range(num_sparse_frames):
                structure_decay_config[int(sparseConfig.dense_epochs) + int(i*sparse_frame_size)]  = str((M-1) if (i == 0) else int(M/math.pow(2,i))) + ":" + str(M)
        elif(structure_decay_type == "densify"):
            num_sparse_frames = 4
            sparse_frame_size = (sparseConfig.total_epochs - sparseConfig.dense_epochs - sparseConfig.fine_tune_epochs) / num_sparse_frames
            structure_decay_config = dict()
            for i in range(num_sparse_frames):
                structure_decay_config[int(sparseConfig.dense_epochs) + int(i*sparse_frame_size)]  = str(N) + ":" + str(int(M*math.pow(2,num_sparse_frames-i-1)))
        elif(structure_decay_type == "fine"):
            num_sparse_frames = 4
            sparse_frame_size = (sparseConfig.total_epochs - sparseConfig.dense_epochs - sparseConfig.fine_tune_epochs) / num_sparse_frames
            structure_decay_config = dict()
            for i in range(num_sparse_frames):
                structure_decay_config[int(sparseConfig.dense_epochs) + int(i*sparse_frame_size)]  =  str(int(N*math.pow(2,num_sparse_frames-i-1))) + ":" + str(int(M*math.pow(2,num_sparse_frames-i-1)))
        else:
            print("structured sparse decay type unidentified. Use on of the following: SPARSIFY,DENSIFY,FINE,CONFIG. For CONFIG please also specify structure-decay-config")
            sys.exit(1)
        print(structure_decay_config)
    else:
        structure_decay_config = None
    
    return structure_decay_config


# SparseSRSTELinear : SR-STE
# Taken from https://github.com/aojunzz/NM-sparsity
# Base SR-STE implementation.
class SparseSRSTE(autograd.Function):
    """" Prune the unimprotant weight for the forwards phase but pass the gradient to dense weight using SR-STE in the backwards phase"""

    @staticmethod
    def forward(ctx, weight, N, M, sparsity_rate = 0.0, isconv=False, sparse_dim = 0, decay = 0.0002):
        ctx.save_for_backward(weight)

        output = weight.clone()
        
        w_b = get_sparse_mask(weight,N,M,sparsity_rate,isconv,sparse_dim)
        ctx.mask = w_b
        ctx.decay = decay

        return output * w_b 


    @staticmethod
    def backward(ctx, grad_output):
        weight, = ctx.saved_tensors
        return grad_output + ctx.decay * (1-ctx.mask) * weight, None, None, None, None, None
    


class StepDecay(autograd.Function):
    """" Prune the unimprotant weight for the forwards phase but pass the default gradient to dense weight in the backwards phase"""

    @staticmethod
    def forward(ctx, weight, N, M, sparsity_rate = 0.0,sparse_dim = 0, isconv=False):
        ctx.save_for_backward(weight)

        output = weight.clone()

        w_b = get_sparse_mask(weight,N,M,sparsity_rate,isconv,sparse_dim)

        ctx.mask = w_b
        ctx.decay = 0.0002
        return output * w_b


    @staticmethod
    def backward(ctx, grad_output):
        # weight, = ctx.saved_tensors
        return grad_output , None, None, None, None, None

class LinearDecay(autograd.Function):
    """" Prune the unimprotant weight for the forwards phase in a linear fashion but pass the default gradient to dense weight in the backwards phase"""

    @staticmethod
    def forward(ctx, weight, N, M, sparsity_rate = 0.0, linear_decay_coef = 0.01 ,current_step_num = 100, isconv=False, sparse_dim = 0):
        ctx.save_for_backward(weight)

        output = weight.clone()

        w_b = get_sparse_mask(weight,N,M,sparsity_rate,isconv,sparse_dim)
        
        ctx.mask = w_b
        ctx.decay = 0.0002
        mask_decay_value = 1.0
        mask_decay_value = max(
            mask_decay_value -
            (linear_decay_coef*current_step_num),
            0.0)
        if(current_step_num%1000==0):
            print("Linear mask decay value:", mask_decay_value )
        return output * (w_b + (1 - w_b) * mask_decay_value)


## For Decay sparse, we are sending gradients as it during back propogation,
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output , None, None, None, None, None, None, None


class ExponentialDecay(autograd.Function):
    """" Prune the unimprotant weight for the forwards phase in a linear fashion but pass the default gradient to dense weight in the backwards phase"""

    @staticmethod
    def forward(ctx, weight, N, M, sparsity_rate = 0.0, exp_decay_coef = 0.01 ,current_step_num = 100, isconv=False,sparse_dim = 0):
        ctx.save_for_backward(weight)

        output = weight.clone()

        w_b = get_sparse_mask(weight,N,M,sparsity_rate,isconv,sparse_dim)
        
        ctx.mask = w_b
        ctx.decay = 0.0002
        mask_decay_value = math.exp(-1*exp_decay_coef*current_step_num) 
        if(current_step_num%1000==0):
            print("Exponential Mask decay value:", mask_decay_value)

        return output * (w_b + (1 - w_b) * mask_decay_value)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output , None, None, None, None, None, None, None


# Decay function with linear,exponential,cosine decay. Training step as input, decay value as output.

class SparseLinear(nn.Linear):

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 sparseConfig = None,**kwargs):
        
        ### Sparsity type
        ##  DENSE = 'DENSE'
        ##  STRUCTURED_NM = 'STRUCTURED_NM'
        ##  UNSTRUCTURED = 'UNSTRUCTURED'
        ##  SRSTE = 'SRSTE'




        self.sparsity_type = sparseConfig.sparsity_type
        ### Decay type, MdGf
        ## Step :- For all steps enforce the "sparsity_type" for the whole matrix.
        ## Linear :- Linearly decay the mask value from 1 to 0, using the training step number and decay coeffecient.
        ##          Mask decay value = max(0, 1 - linear_decay_coef*(current_step_num-starting_step))
        ##
        ## Exp :- Exponentially decay the mask from 1 to 0, using training step and decay coefficients
        ##          Mask decay value = max(0, e^(-exp_decay_coef*(current_step_num-starting_step))
        ##
        self.decay_type = sparseConfig.decay_type

        ## Structure_decay , SdGf
        # Flag to enable uniform structure decay to final N:M.
        ## For example, when target sparsity pattern is 1:16, we divide (n-d-f) 
        # steps to five equal time frame, and the sparsity pattern 
        # of each time frame is 15:16, 8:16, 4:16, 2:16, and 1:16, respectively.
        self.structure_decay_flag = sparseConfig.structure_decay_flag

        ## Sparsity % for uniform sparsity
        self.N = sparseConfig.n_sparsity
        self.M = sparseConfig.m_sparsity
        self.sparsity_rate = sparseConfig.prune_rate

        self.sparse_dim=sparseConfig.sparse_dim

        print("Enabling sparsity type:",self.sparsity_type," , decay_type: ",self.decay_type,", Structure decay flag:",self.structure_decay_flag,", N:M=",self.N,self.M)
        ## Decay config params
        self.decay_coef = sparseConfig.decay_coef
        self.current_step_num = 0
        self.current_epoch = 0

        self.dense_epochs = sparseConfig.dense_epochs
        self.fine_tune_epochs = sparseConfig.fine_tune_epochs
        self.total_epochs = sparseConfig.total_epochs


        self.structure_decay_config = get_decay_config(sparseConfig)
    
        super(SparseLinear, self).__init__(in_features, out_features, bias = bias)


    def get_sparse_weights(self):
        if(self.current_epoch < self.dense_epochs or self.current_step_num<0):     ## Return dense weights
            return self.weight

        if(self.sparsity_type.lower() == "srste"):
            return SparseSRSTE.apply(self.weight, self.N, self.M, self.sparsity_rate,False,self.sparse_dim)
        else:
            if(self.structure_decay_config is not None):
                if(self.current_epoch in self.structure_decay_config ):
                    self.N =  int(self.structure_decay_config[self.current_epoch].split(':')[0])
                    self.M =  int(self.structure_decay_config[self.current_epoch].split(':')[1])

            ## When doing fine tuning, the mask is binary (0,1)
            if(self.current_epoch > (self.total_epochs-self.fine_tune_epochs)):
                self.decay_type = "step"


            if(self.decay_type.lower() == "step"):
                return StepDecay.apply(self.weight, self.N, self.M, self.sparsity_rate,False,self.sparse_dim)
            elif(self.decay_type.lower() == "linear"):
                return LinearDecay.apply(self.weight, self.N, self.M, self.sparsity_rate, self.decay_coef,(self.current_step_num),False,self.sparse_dim)
            elif(self.decay_type.lower() == "exp"):
                return ExponentialDecay.apply(self.weight, self.N, self.M, self.sparsity_rate, self.decay_coef,(self.current_step_num),False,self.sparse_dim)
            else:
                print("decay type unidentified. Use on of the following: step,linear,exp.")
                sys.exit(1)



    def forward(self, x, current_step_num = 0,current_epoch=0):
        self.current_step_num = current_step_num
        self.current_epoch = current_epoch
        w = self.get_sparse_weights()
        x = F.linear(x, w)
        return x

    def __return_sparse_weights__(self):
        return self.get_sparse_weights() 


class SparseThreeLinears(nn.Module):
    def __init__(self, in_features, out_features, bias = False, sparseConfig = None,**kwargs):
        super(SparseThreeLinears, self).__init__()

        sparseConfig.n_sparsity = sparseConfig.n_sparsity_qkv
        sparseConfig.m_sparsity = sparseConfig.m_sparsity_qkv
        sparseConfig.prune_rate = sparseConfig.prune_rate_qkv
        self.sparseConfig = sparseConfig

        if 'Q' in sparseConfig.sparsity_loc:
            self.q = SparseLinear(in_features,out_features//3 , sparseConfig = sparseConfig)
            print("Sparse Q")
        else:
            print("Dense Q")
            self.q = nn.Linear(in_features, out_features//3)
        
        if 'K' in sparseConfig.sparsity_loc:
            self.k = SparseLinear(in_features,out_features//3 , sparseConfig = sparseConfig)
            print("Sparse K")
        else:
            print("Dense K")
            self.k = nn.Linear(in_features, out_features//3)
        
        if 'V' in sparseConfig.sparsity_loc:
            self.v = SparseLinear(in_features,out_features//3 , sparseConfig = sparseConfig)
            print("Sparse V")
        else:
            print("Dense V")
            self.v = nn.Linear(in_features, out_features//3)
        
    def forward(self, x, current_step = 0,current_epoch=0):
        try:
            x1 = self.q(x,current_step_num=current_step,current_epoch=current_epoch)
        except:
            x1 = self.q(x)

        try:
            x2 = self.k(x,current_step_num=current_step,current_epoch=current_epoch)
        except:
            x2 = self.k(x)

        try:
            x3 = self.v(x,current_step_num=current_step,current_epoch=current_epoch)
        except:
            x3 = self.v(x)
        return torch.cat([x1, x2, x3], dim=-1)

    def get_weights(self):
        if 'Q' in self.sparseConfig.sparsity_loc:
            q_weight = self.q.__return_sparse_weights__()
        else:
            q_weight = self.q.weight
        
        if 'K' in self.sparseConfig.sparsity_loc:
            k_weight = self.k.__return_sparse_weights__()
        else:
            k_weight = self.k.weight
        
        if 'V' in self.sparseConfig.sparsity_loc:
            v_weight = self.v.__return_sparse_weights__()
        else:
            v_weight = self.v.weight 
        return torch.cat((q_weight,k_weight,v_weight))



class SparseConv2D(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros',
                 sparseConfig = None,**kwargs):
        
        ### Sparsity type
        ##  DENSE = 'DENSE'
        ##  STRUCTURED_NM = 'STRUCTURED_NM'
        ##  UNSTRUCTURED = 'UNSTRUCTURED'
        ##  SRSTE = 'SRSTE'
        self.sparsity_type = sparseConfig.sparsity_type



        ### Decay type
        ## Step :- For all steps enforce the "sparsity_type" for the whole matrix.
        ## Linear :- Linearly decay the mask value from 1 to 0, using the training step number and decay coeffecient.
        ##          Mask decay value = max(0, 1 - linear_decay_coef*(current_step_num-starting_step))
        ##
        ## Exp :- Exponentially decay the mask from 1 to 0, using training step and decay coefficients
        ##          Mask decay value = max(0, e^(-exp_decay_coef*(current_step_num-starting_step))
        ##
        self.decay_type = sparseConfig.decay_type

        ## Structure_decay 
        # Flag to enable uniform structure decay to final N:M.
        ## For example, when target sparsity pattern is 1:16, we divide (n-d-f) 
        # steps to five equal time frame, and the sparsity pattern 
        # of each time frame is 15:16, 8:16, 4:16, 2:16, and 1:16, respectively.
        self.structure_decay_flag = sparseConfig.structure_decay_flag

        ## Sparsity % for uniform sparsity
        self.N = sparseConfig.n_sparsity
        self.M = sparseConfig.m_sparsity
        self.sparsity_rate = sparseConfig.prune_rate

        print("Enabling sparsity type:",self.sparsity_type," , decay_type: ",self.decay_type,", Structure decay flag:",self.structure_decay_flag,", N:M=",self.N,self.M)
        ## Decay config params
        self.decay_coef = sparseConfig.decay_coef
        self.current_step_num = 0
        self.current_epoch = 0

        ## TBD, update these from real parameters in training run.
        self.dense_epochs = sparseConfig.dense_epochs
        self.fine_tune_epochs = sparseConfig.fine_tune_epochs
        self.total_epochs = sparseConfig.total_epochs

        self.structure_decay_config = get_decay_config(sparseConfig)
        


        super(SparseConv2D, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, **kwargs)

    def get_sparse_weights(self):
        if(self.current_epoch < self.dense_epochs or self.current_step_num<0):     ## Return dense weights
            return self.weight

        if(self.sparsity_type.lower() == "srste"):
            return SparseSRSTE.apply(self.weight, self.N, self.M, self.sparsity_rate, True)
        else:
            if(self.structure_decay_config is not None):
                if(self.current_epoch in self.structure_decay_config ):
                    self.N =  int(self.structure_decay_config[self.current_epoch].split(':')[0])
                    self.M =  int(self.structure_decay_config[self.current_epoch].split(':')[1])

            ## When doing fine tuning, the mask is binary (0,1)
            if(self.current_epoch > (self.total_epochs-self.fine_tune_epochs)):
                self.decay_type = "step"


            if(self.decay_type.lower() == "step"):
                return StepDecay.apply(self.weight, self.N, self.M, self.sparsity_rate, True)
            elif(self.decay_type.lower() == "linear"):
                return LinearDecay.apply(self.weight, self.N, self.M, self.sparsity_rate, self.decay_coef,(self.current_step_num), True)
            elif(self.decay_type.lower() == "exp"):
                return ExponentialDecay.apply(self.weight, self.N, self.M, self.sparsity_rate, self.decay_coef,(self.current_step_num), True)
            else:
                print("decay type unidentified. Use on of the following: step,linear,exp.")
                sys.exit(1)

    def forward(self, x, current_step_num = 0,current_epoch=0):
        self.current_step_num = current_step_num
        self.current_epoch = current_epoch
        w = self.get_sparse_weights()
        x = F.conv2d(
            x, w, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        return x