import math
import six
import numpy

import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter

TINY_VALUE = float(numpy.finfo(numpy.float32).tiny)


# ------------- ESPnet Relatives ----------------------------------------------------------------------------------------
def to_device(m, x):
    """Send tensor into the device of the module.

    Args:
        m (torch.nn.Module): Torch module.
        x (Tensor): Torch tensor.

    Returns:
        Tensor: Torch tensor located in the same place as torch module.

    """
    assert isinstance(m, torch.nn.Module)
    device = next(m.parameters()).device
    return x.to(device)

def make_pad_mask(lengths, xs=None, length_dim=-1):
    """Make mask tensor containing indices of padded part.

    Args:
        lengths (LongTensor or List): Batch of lengths (B,).
        xs (Tensor, optional): The reference tensor. If set, masks will be the same shape as this tensor.
        length_dim (int, optional): Dimension indicator of the above tensor. See the example.

    Returns:
        Tensor: Mask tensor containing indices of padded part.

    Examples:
        >>> lengths = [5, 3, 2]
        >>> make_non_pad_mask(lengths)
        masks = [[0, 0, 0, 0 ,0],
                 [0, 0, 0, 1, 1],
                 [0, 0, 1, 1, 1]]

    """
    if length_dim == 0:
        raise ValueError('length_dim cannot be 0: {}'.format(length_dim))

    if not isinstance(lengths, list):
        lengths = lengths.tolist()
    bs = int(len(lengths))
    if xs is None:
        maxlen = int(max(lengths))
    else:
        maxlen = xs.size(length_dim)

    seq_range = torch.arange(0, maxlen, dtype=torch.int64)
    seq_range_expand = seq_range.unsqueeze(0).expand(bs, maxlen)
    seq_length_expand = seq_range_expand.new(lengths).unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand

    if xs is not None:
        assert xs.size(0) == bs, (xs.size(0), bs)

        if length_dim < 0:
            length_dim = xs.dim() + length_dim
        # ind = (:, None, ..., None, :, , None, ..., None)
        ind = tuple(slice(None) if i in (0, length_dim) else None
                    for i in range(xs.dim()))
        mask = mask[ind].expand_as(xs).to(xs.device)
    return mask
    
# ------------- MoChA Relatives ----------------------------------------------------------------------------------------
def moving_sum(x, back, forward):
    """Compute the moving sum of x over a window with the provided bounds.

    x is expected to be of shape (B x T_max).
    The returned tensor x_sum is computed as
    x_sum[i, j] = x[i, j - back] + ... + x[i, j + forward]
    """
    x_pad = F.pad(x, (back,forward,0,0), "constant", 0.0)
    kernel = torch.ones(1,1,back+forward+1, dtype=x.dtype)
    return F.conv1d(x_pad.unsqueeze(1), kernel).squeeze(1)
    
def safe_cumprod(x, *args, **kwargs):
    """Computes cumprod of x in logspace using cumsum to avoid underflow.
    The cumprod function and its gradient can result in numerical instabilities
    when its argument has very small and/or zero values.  As long as the argument
    is all positive, we can instead compute the cumulative product as
    exp(cumsum(log(x))).  This function can be called identically to tf.cumprod.
    Args:
        x: Tensor to take the cumulative product of.
        *args: Passed on to cumsum; these are identical to those in cumprod.
        **kwargs: Passed on to cumsum; these are identical to those in cumprod.
    Returns:
        Cumulative product of x, the first element is 1.
    """
    cumprod = torch.exp(torch.cumsum(torch.log(torch.clamp(x[:,:-1], TINY_VALUE, 1.)), *args, **kwargs))
    exclusive_cumprod = cumprod.new_ones(x.shape)
    exclusive_cumprod[:,1:] = cumprod
    return exclusive_cumprod

class MoChA(torch.nn.Module):
    '''Monotonic chunkwise attention dropping prev attention distribution
        which is slightly different from Google's formulation but more stable during training

    :param int eprojs: # projection-units of encoder
    :param int dunits: # units of decoder
    :param int att_dim: attention dimension
    :param int win: chunk width for MoChA
    :param float scaling: scaling parameter before applying softmax
    :param float sigmoid_noise: Standard deviation of pre-sigmoid noise.
    :param float score_bias_init: Initial value for score bias scalar.
                                  It's recommended to initialize this to a negative value
                                  (e.g. -4.0) when the length of the memory is large.
    '''

    def __init__(self, eprojs, dunits, att_dim, att_win,
                 sigmoid_noise=1.0, score_bias_init=-4.0):
        super(MoChA, self).__init__()
        
        self.monotonic_mlp_enc = torch.nn.Linear(eprojs, att_dim)
        self.monotonic_mlp_dec = torch.nn.Linear(dunits, att_dim, bias=False)
        self.monotonic_gvec = torch.nn.Linear(att_dim, 1, bias=False)
        self.monotonic_factor = torch.nn.Parameter(torch.Tensor(1,1)) # don't forget to initialize this to 1.0 / math.sqrt(att_dim)
        self.monotonic_bias = torch.nn.Parameter(torch.Tensor(1,1)) # don't forget to initialize this to a negative value (e.g. -4.0)
        
        assert att_win > 0
        if att_win > 1: # Hard Monotonic Attention for att_win = 1
            self.chunk_mlp_enc = torch.nn.Linear(eprojs, att_dim)
            self.chunk_mlp_dec = torch.nn.Linear(dunits, att_dim, bias=False)
            self.chunk_gvec = torch.nn.Linear(att_dim, 1, bias=False)
        
        self.dunits = dunits
        self.eprojs = eprojs
        self.att_dim = att_dim
        self.att_win = att_win
        self.sigmoid_noise = sigmoid_noise
        self.score_bias_init = score_bias_init
        self.h_length = None
        self.enc_h = None
        self.pre_compute_enc_h = None
        self.pre_compute_chunk_enc_h = None

    def reset(self):
        '''reset states'''
        self.h_length = None
        self.enc_h = None
        self.pre_compute_enc_h = None
        self.pre_compute_chunk_enc_h = None
        self.mask = None
    
    def forward(self, enc_hs_pad, enc_hs_len, dec_z, att_prev, scaling=2.0):
        '''MoChA forward

        :param torch.Tensor enc_hs_pad: padded encoder hidden state (B x T_max x D_enc)
        :param list enc_h_len: padded encoder hidden state lenght (B)
        :param torch.Tensor dec_z: docoder hidden state (B x D_dec)
        :param torch.Tensor att_prev: previous attetion weight (B x T_max)

        :return: attentioin weighted encoder state (B, D_enc)
        :rtype: torch.Tensor
        :return: previous a (B x T_max)
        :rtype: torch.Tensor
        '''
        
        batch = len(enc_hs_pad)
        # pre-compute all h outside the decoder loop
        if self.pre_compute_enc_h is None:
            self.enc_h = enc_hs_pad  # utt x frame x hdim
            self.h_length = self.enc_h.size(1)
            # utt x frame x att_dim
            self.pre_compute_enc_h = self.monotonic_mlp_enc(self.enc_h)
            if self.att_win > 1:
                self.pre_compute_chunk_enc_h = self.chunk_mlp_enc(self.enc_h)

        if dec_z is None:
            dec_z = enc_hs_pad.new_zeros(batch, self.dunits)
        else:
            dec_z = dec_z.view(batch, self.dunits)
        # dec_z_tiled: utt x frame x att_dim
        dec_z_tiled = self.monotonic_mlp_dec(dec_z).view(batch, 1, self.att_dim)
        
        if att_prev is None:
            att_prev = enc_hs_pad.new_zeros(batch, self.h_length)
            att_prev[:,0] = 1.0 # initialize attention weights
        
        # Implements additive energy function to compute pre-sigmoid activation e.
        # Sigmoid is used to compute selection probability p, than its expectation value a.    
        # To mitigate saturating and sensitivity to offset, 
        # monotonic_factor and monotonic_bias are added here as learnable scalars
        # utt x frame x att_dim -> utt x frame
        e = self.monotonic_factor / torch.norm(self.monotonic_gvec.weight, p=2) \
          * self.monotonic_gvec(torch.tanh(self.pre_compute_enc_h + dec_z_tiled)).squeeze(2) \
          + self.monotonic_bias
        
        # NOTE consider zero padding when compute p and a
        # a: utt x frame
        if self.mask is None:
            self.mask = to_device(self, make_pad_mask(enc_hs_len))
        e.masked_fill_(self.mask, -float('inf'))
        # Optionally add pre-sigmoid noise to the scores
        e += self.sigmoid_noise * to_device(self,torch.normal(mean=torch.zeros(e.shape), std=1))
        p = torch.sigmoid(e)
        # safe_cumprod computes cumprod in logspace with numeric checks
        cumprod_1mp = safe_cumprod(1-p, dim=1)
        # Google's formulation:
        # a = p * cumprod_1mp * torch.cumsum(
        #    att_prev / torch.clamp(cumprod_1mp, 1e-10, 1.), dim=1)
        # or an approximation:
        # a = p * cumprod_1mp * torch.cumsum(att_prev, dim=1)
        # Stable MoChA:
        a = p * cumprod_1mp

        if self.att_win == 1:
            w = a.masked_fill(self.mask, 0)
        else:
            # dec_z_chunk_tiled: utt x frame x att_dim
            dec_z_chunk_tiled = self.chunk_mlp_dec(dec_z).view(batch, 1, self.att_dim)
            # dot with gvec
            # utt x frame x att_dim -> utt x frame
            u = self.chunk_gvec(torch.tanh(self.pre_compute_chunk_enc_h + dec_z_chunk_tiled)).squeeze(2)

            # NOTE consider zero padding when compute w.
            u.masked_fill_(self.mask, -float('inf'))
            exp_u = torch.exp(u * scaling)
            w = exp_u * moving_sum(a / torch.clamp(moving_sum(exp_u, self.att_win-1, 0), 1e-10, float('inf')), 0, self.att_win-1)

        # weighted sum over flames
        # utt x hdim
        # NOTE use bmm instead of sum(*)
        c = torch.sum(self.enc_h * w.view(batch, self.h_length, 1), dim=1)
                 
        return c, a

class OnlineMoChA(MoChA):
    '''MoChA for online decoding
    '''
    def __init__(self, *args, **kwargs):
        super(OnlineMoChA, self).__init__(*args, **kwargs)
        
    def reset(self):
        '''reset states'''
        self.h_length = 0
        self.enc_h = None
        self.pre_compute_enc_h = None
        self.pre_compute_chunk_enc_h = None
        self.last_offset = 0

    def forward(self, enc_hs_pad, enc_hs_len, dec_z, end_point, scaling=2.0, offset=0):
        '''MoChA forward in online scenario, only support one utterance

        :param torch.Tensor enc_hs_pad: padded encoder hidden state (B x T_max x D_enc)
        :param list enc_h_len: padded encoder hidden state lenght (B)
        :param torch.Tensor dec_z: docoder hidden state (B x D_dec)
        :param int end_point: previous end-point of MoChA
        :param int offset: the first index of new coming encoder hidden states
                           designed for streaming encoder

        :return: attentioin weighted encoder state (B, D_enc)
        :rtype: torch.Tensor
        :return: previous end-point (B x T_max)
        :rtype: torch.Tensor
        '''        
        assert len(enc_hs_pad) == 1
        batch = 1
        if self.pre_compute_enc_h is None or offset > self.last_offset:
            self.enc_h = enc_hs_pad if self.enc_h is None else torch.cat([self.enc_h, enc_hs_pad], dim=1)
            self.h_length += self.enc_h.size(1)
            # utt x frame x att_dim
            self.pre_compute_enc_h = self.monotonic_mlp_enc(self.enc_h) if self.pre_compute_enc_h is None else \
                                     torch.cat([self.pre_compute_enc_h, self.monotonic_mlp_enc(self.enc_h)], dim=1)
            if self.att_win > 1:
                self.pre_compute_chunk_enc_h = self.chunk_mlp_enc(self.enc_h) if self.pre_compute_chunk_enc_h is None else \
                                               torch.cat([self.pre_compute_chunk_enc_h, self.chunk_mlp_enc(self.enc_h)], dim=1)
            self.last_offset = offset      
        if dec_z is None:
            dec_z = enc_hs_pad.new_zeros(batch, self.dunits)
        else:
            dec_z = dec_z.view(batch, self.dunits)
        
        if end_point is None:
            end_point = 0

        # dec_z_tiled: utt x frame x att_dim
        dec_z_tiled = self.monotonic_mlp_dec(dec_z).view(batch, 1, self.att_dim)
        
        # utt x frame x att_dim -> utt x frame
        e = self.monotonic_factor / torch.norm(self.monotonic_gvec.weight, p=2) \
          * self.monotonic_gvec(torch.tanh(self.pre_compute_enc_h[:,end_point:,:] + dec_z_tiled)).squeeze(2) \
          + self.monotonic_bias

        flag = False
        for z in range(e.size(1)):
            if e[0,z] > 0:
                flag = True
                break
        #z = torch.nonzero((e > 0).to(e.dtype))
        
        if flag:
            end_point += z
            if self.att_win == 1:
                c = self.enc_h[:, end_point]
            else:
                # dec_z_chunk_tiled: utt x frame x att_dim
                dec_z_chunk_tiled = self.chunk_mlp_dec(dec_z).view(batch, 1, self.att_dim)
                # dot with gvec
                # utt x frame x att_dim -> utt x frame
                start_point = max(0, end_point - self.win + 1)
                u = self.chunk_gvec(torch.tanh(self.pre_compute_chunk_enc_h[:, start_point:end_point+1] + dec_z_chunk_tiled)).squeeze(2)

                w = F.softmax(scaling * u, dim=1)
                c = torch.sum(self.enc_h[:, start_point:end_point+1] * w.view(batch, end_point-start_point+1, 1), dim=1)
        else:
            c = enc_hs_pad.new_zeros(batch, self.eprojs)
        return c, end_point
  
class MTA(torch.nn.Module):
    '''Monotonic truncated attention 
    :param int eprojs: # projection-units of encoder
    :param int dunits: # units of decoder
    :param int att_dim: attention dimension
    :param float scaling: scaling parameter before applying softmax
    :param float sigmoid_noise: Standard deviation of pre-sigmoid noise.
    :param float score_bias_init: Initial value for score bias scalar.
                                  It's recommended to initialize this to a negative value
                                  (e.g. -4.0) when the length of the memory is large.
    '''

    def __init__(self, eprojs, dunits, att_dim,
                 sigmoid_noise=1.0, score_bias_init=-4.0):
        super(MTA, self).__init__()
        
        self.monotonic_mlp_enc = torch.nn.Linear(eprojs, att_dim)
        self.monotonic_mlp_dec = torch.nn.Linear(dunits, att_dim, bias=False)
        self.monotonic_gvec = torch.nn.Linear(att_dim, 1, bias=False)
        self.monotonic_factor = torch.nn.Parameter(torch.Tensor(1,1)) # don't forget to initialize this to 1.0 / math.sqrt(att_dim)
        self.monotonic_bias = torch.nn.Parameter(torch.Tensor(1,1)) # don't forget to initialize this to a negative value (e.g. -4.0)
        
        self.dunits = dunits
        self.eprojs = eprojs
        self.att_dim = att_dim
        self.sigmoid_noise = sigmoid_noise
        self.score_bias_init = score_bias_init
        self.h_length = None
        self.enc_h = None
        self.pre_compute_enc_h = None

    def reset(self):
        '''reset states'''
        self.h_length = None
        self.enc_h = None
        self.pre_compute_enc_h = None
        self.mask = None
    
    def forward(self, enc_hs_pad, enc_hs_len, dec_z, att_prev, scaling=2.0):
        '''MTA forward
        :param torch.Tensor enc_hs_pad: padded encoder hidden state (B x T_max x D_enc)
        :param list enc_h_len: padded encoder hidden state lenght (B)
        :param torch.Tensor dec_z: docoder hidden state (B x D_dec)
        :param torch.Tensor att_prev: previous attetion weight (B x T_max)
        :return: attentioin weighted encoder state (B, D_enc)
        :rtype: torch.Tensor
        :return: previous a (B x T_max)
        :rtype: torch.Tensor
        '''
        
        batch = len(enc_hs_pad)
        # pre-compute all h outside the decoder loop
        if self.pre_compute_enc_h is None:
            self.enc_h = enc_hs_pad  # utt x frame x hdim
            self.h_length = self.enc_h.size(1)
            # utt x frame x att_dim
            self.pre_compute_enc_h = self.monotonic_mlp_enc(self.enc_h)

        if dec_z is None:
            dec_z = enc_hs_pad.new_zeros(batch, self.dunits)
        else:
            dec_z = dec_z.view(batch, self.dunits)
        # dec_z_tiled: utt x frame x att_dim
        dec_z_tiled = self.monotonic_mlp_dec(dec_z).view(batch, 1, self.att_dim)
        
        if att_prev is None:
            att_prev = enc_hs_pad.new_zeros(batch, self.h_length)
            att_prev[:,0] = 1.0 # initialize attention weights
        
        # Implements additive energy function to compute pre-sigmoid activation e.
        # Sigmoid is used to compute selection probability p, than its expectation value a.    
        # To mitigate saturating and sensitivity to offset, 
        # monotonic_factor and monotonic_bias are added here as learnable scalars
        # utt x frame x att_dim -> utt x frame
        e = self.monotonic_factor / torch.norm(self.monotonic_gvec.weight, p=2) \
          * self.monotonic_gvec(torch.tanh(self.pre_compute_enc_h + dec_z_tiled)).squeeze(2) \
          + self.monotonic_bias
        
        # NOTE consider zero padding when compute p and a
        # a: utt x frame
        if self.mask is None:
            self.mask = to_device(self, make_pad_mask(enc_hs_len))
        e.masked_fill_(self.mask, -float('inf'))
        # Optionally add pre-sigmoid noise to the scores
        e += self.sigmoid_noise * to_device(self,torch.normal(mean=torch.zeros(e.shape), std=1))
        p = torch.sigmoid(e)
        # safe_cumprod computes cumprod in logspace with numeric checks
        cumprod_1mp = safe_cumprod(1-p, dim=1)
        a = p * cumprod_1mp        
        w = a.masked_fill(self.mask, 0)

        # weighted sum over flames
        # utt x hdim
        # NOTE use bmm instead of sum(*)
        c = torch.sum(self.enc_h * w.view(batch, self.h_length, 1), dim=1)
                 
        return c, a

class OnlineMTA(MTA):
    '''MTA for online decoding
        aim to use historical encoder outputs and simplify MoChA
    '''
    def __init__(self, *args, **kwargs):
        super(OnlineMTA, self).__init__(*args, **kwargs)
        
    def reset(self):
        '''reset states'''
        self.h_length = 0
        self.enc_h = None
        self.pre_compute_enc_h = None
        self.last_offset = 0

    def forward(self, enc_hs_pad, enc_hs_len, dec_z, end_point, scaling=2.0, offset=0):
        '''MoChA forward in online scenario, only support one utterance
        :param torch.Tensor enc_hs_pad: padded encoder hidden state (B x T_max x D_enc)
        :param list enc_h_len: padded encoder hidden state lenght (B)
        :param torch.Tensor dec_z: docoder hidden state (B x D_dec)
        :param int end_point: previous end-point of MTA (B)
        :param int offset: the first index of new coming encoder hidden states
                           designed for streaming encoder
        :return: attentioin weighted encoder state (B, D_enc)
        :rtype: torch.Tensor
        :return: previous end-point (B)
        :rtype: torch.Tensor
        '''        
        assert len(enc_hs_pad) == 1
        batch = 1
        if self.pre_compute_enc_h is None or offset > self.last_offset:
            self.enc_h = enc_hs_pad if self.enc_h is None else torch.cat([self.enc_h, enc_hs_pad], dim=1)
            self.h_length += self.enc_h.size(1)
            # utt x frame x att_dim
            self.pre_compute_enc_h = self.monotonic_mlp_enc(self.enc_h) if self.pre_compute_enc_h is None else \
                                     torch.cat([self.pre_compute_enc_h, self.monotonic_mlp_enc(self.enc_h)], dim=1)
            self.last_offset = offset      
        if dec_z is None:
            dec_z = enc_hs_pad.new_zeros(batch, self.dunits)
        else:
            dec_z = dec_z.view(batch, self.dunits)
        
        if end_point is None:
            end_point = 0

        # dec_z_tiled: utt x frame x att_dim
        dec_z_tiled = self.monotonic_mlp_dec(dec_z).view(batch, 1, self.att_dim)
        
        # utt x frame x att_dim -> utt x frame
        e = self.monotonic_factor / torch.norm(self.monotonic_gvec.weight, p=2) \
          * self.monotonic_gvec(torch.tanh(self.pre_compute_enc_h + dec_z_tiled)).squeeze(2) \
          + self.monotonic_bias

        flag = False
        for z in range(end_point, e.size(1)):
            if e[0,z] > 0:
                flag = True
                break
        #z = torch.nonzero((e > 0).to(e.dtype))
        
        if flag:
            end_point = z
            p = torch.sigmoid(e[:, :end_point+1])
            cumprod_1mp = safe_cumprod(1-p, dim=1)
            w = p * cumprod_1mp
            c = torch.sum(self.enc_h[:, :end_point+1] * w.view(batch, end_point+1, 1), dim=1)
        else:
            c = enc_hs_pad.new_zeros(batch, self.eprojs)
        return c, end_point
