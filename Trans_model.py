# -*- coding: utf-8 -*-
"""
# @copyright (c) 2023 Baidu.com, Inc. Allrights Reserved
@Time ： 2023/3/1 11:35
@Author ： Liu Tianyuan (liutianyuan02@baidu.com)
@Site ：Trans_model.py
@File ：Trans_model.py
"""

import copy
import math
import numpy as np

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from FNO_model import SpectralConv2d
from utils import activation_dict, params_initial
from collections import defaultdict
import os


# from Models.configs import *

def vanilla_attention(query, key, value,
                      mask=None, dropout=None, weight=None,
                      attention_type='softmax'):
    '''
    Simplified from
    https://nlp.seas.harvard.edu/2018/04/03/attention.html
    Compute the Scaled Dot Product Attention
    '''
    d_k = query.shape[-1]
    trans_dim = np.arange(query.dim())
    trans_dim[-2:] = trans_dim[-1:-3:-1]

    if attention_type == 'cosine':
        p_attn = F.cosine_similarity(query, key.transpose(trans_dim)) \
                 / math.sqrt(d_k)
    else:
        scores = paddle.matmul(query, key.transpose(trans_dim)) \
                 / math.sqrt(d_k)
        seq_len = scores.shape[-1]

        if attention_type == 'softmax':
            if mask is not None:
                scores = scores.masked_fill(mask == 0, -1e9)
            p_attn = F.softmax(scores, axis=-1)
        elif attention_type in ['fourier', 'integral', 'local']:
            if mask is not None:
                scores = scores.masked_fill(mask == 0, 0)
            p_attn = scores / seq_len

    if dropout is not None:
        p_attn = F.dropout(p_attn)

    out = paddle.matmul(p_attn, value)

    return out, p_attn


def linear_attention(query, key, value,
                     mask=None, dropout=None,
                     attention_type='galerkin'):
    '''
    Adapted from lucidrains' implementaion
    https://github.com/lucidrains/linear-attention-transformer/blob/master/linear_attention_transformer/linear_attention_transformer.py
    to https://nlp.seas.harvard.edu/2018/04/03/attention.html template
    linear_attn function
    Compute the Scaled Dot Product Attention globally
    '''

    seq_len = query.shape[-2]
    trans_dim = np.arange(query.dim())
    trans_dim[-2:] = trans_dim[-1:-3:-1]

    if attention_type in ['linear', 'global']:
        query = query.softmax(axis=-1)
        key = key.softmax(axis=-2)
    scores = paddle.matmul(key.transpose(trans_dim), value)

    if mask is not None:
        raise RuntimeError("linear attention does not support casual mask.")

    p_attn = scores / seq_len

    if dropout is not None:
        p_attn = F.dropout(p_attn)

    out = paddle.matmul(query, p_attn)
    return out, p_attn


class FeedForward(nn.Layer):
    """
    FeedForward layer in transformers
    """

    def __init__(self, in_dim=256,
                 dim_feedforward: int = 1024,
                 out_dim=None,
                 batch_norm=False,
                 activation='relu',
                 dropout=0.1):
        super(FeedForward, self).__init__()
        if out_dim is None:
            out_dim = in_dim
        n_hidden = dim_feedforward
        # activation = default(activation, 'relu')
        self.lr1 = nn.Linear(in_dim, n_hidden)
        self.activation = activation_dict[activation]
        self.batch_norm = batch_norm
        if self.batch_norm:
            self.bn = nn.BatchNorm1D(n_hidden)
        self.lr2 = nn.Linear(n_hidden, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        forward compute
        :param in_var: (batch, seq_len, in_dim)
        """
        x = self.activation(self.lr1(x))
        x = self.dropout(x)
        if self.batch_norm:
            x = x.permute((0, 2, 1))
            x = self.bn(x)
            x = x.permute((0, 2, 1))
        x = self.lr2(x)
        return x


class PositionalEncoding(nn.Layer):
    '''
    https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    This is not necessary if spacial coords are given
    input is (batch, seq_len, d_model)
    '''

    def __init__(self, d_model,
                 dropout=0.1,
                 max_len=2 ** 13):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)

        pe = paddle.zeros((max_len, d_model), dtype=paddle.float32)
        position = paddle.arange(0, max_len, dtype=paddle.float32).unsqueeze(1)
        div_term = paddle.exp(paddle.arange(0, d_model, 2, dtype=paddle.float32) * (-math.log(2 ** 13) / d_model))
        pe[:, 0::2] = paddle.sin(position * div_term)
        pe[:, 1::2] = paddle.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        forward compute
        :param in_var: (batch, seq_len, d_model)
        """
        x = x + self.pe[:, :x.shape[1], :]
        return self.dropout(x)


class SimpleAttention(nn.Layer):
    '''
    The attention is using a vanilla (QK^T)V or Q(K^T V) with no softmax
    For an encoder layer, the tensor size is slighly different from the official pytorch implementation

    attn_types:
        - fourier: integral, local
        - galerkin: global
        - linear: standard linearization
        - softmax: classic softmax attention

    In this implementation, output is (N, L, E).
    batch_first will be added in the next version of PyTorch: https://github.com/pytorch/pytorch/pull/55285

    Reference: code base modified from
    https://nlp.seas.harvard.edu/2018/04/03/attention.html
    - added xavier init gain
    - added layer norm <-> attn norm switch
    - added diagonal init

    In https://github.com/lucidrains/linear-attention-transformer/blob/master/linear_attention_transformer/linear_attention_transformer.py
    the linear attention in each head is implemented as an Einstein sum
    attn_matrix = paddle.einsum('bhnd,bhne->bhde', k, v)
    attn = paddle.einsum('bhnd,bhde->bhne', q, attn_matrix)
    return attn.reshape(*q.shape)
    here in our implementation this is achieved by a slower transpose+matmul
    but can conform with the template Harvard NLP gave
    '''

    def __init__(self, n_head, d_model,
                 pos_dim: int = 1,
                 attention_type='fourier',
                 dropout=0.1,
                 xavier_init=1e-4,
                 diagonal_weight=1e-2,
                 symmetric_init=False,
                 norm_add=False,
                 norm_type='layer',
                 eps=1e-5):
        super(SimpleAttention, self).__init__()
        assert d_model % n_head == 0  # n_head 可被d_model整除
        self.attention_type = attention_type
        self.d_k = d_model // n_head
        self.n_head = n_head
        self.pos_dim = pos_dim
        self.linears = nn.LayerList(
            [copy.deepcopy(nn.Linear(d_model, d_model)) for _ in range(3)])
        self.xavier_init = xavier_init
        self.diagonal_weight = diagonal_weight
        self.symmetric_init = symmetric_init
        if self.xavier_init > 0:
            self._reset_parameters()
        self.norm_add = norm_add
        self.norm_type = norm_type
        if norm_add:
            self._get_norm(eps=eps)

        if pos_dim > 0:
            self.fc = nn.Linear(d_model + n_head * pos_dim, d_model)

        self.attn_weight = None
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, pos=None, mask=None, weight=None):
        """
        forward compute
        :param query: (batch, seq_len, d_model)
        :param key: (batch, seq_len, d_model)
        :param value: (batch, seq_len, d_model)
        """
        if mask is not None:
            mask = mask.unsqueeze(1)

        bsz = query.shape[0]
        if weight is not None:
            query, key = weight * query, weight * key

        query, key, value = \
            [layer(x).reshape((bsz, -1, self.n_head, self.d_k)).transpose((0, 2, 1, 3))
             for layer, x in zip(self.linears, (query, key, value))]

        if self.norm_add:
            if self.attention_type in ['linear', 'galerkin', 'global']:
                if self.norm_type == 'instance':
                    key, value = key.transpose((0, 1, 3, 2)), value.transpose((0, 1, 3, 2))

                key = paddle.stack(
                    [norm(x) for norm, x in
                     zip(self.norm_K, (key[:, i, ...] for i in range(self.n_head)))], axis=1)
                value = paddle.stack(
                    [norm(x) for norm, x in
                     zip(self.norm_V, (value[:, i, ...] for i in range(self.n_head)))], axis=1)

                if self.norm_type == 'instance':
                    key, value = key.transpose((0, 1, 3, 2)), value.transpose((0, 1, 3, 2))
            else:
                if self.norm_type == 'instance':
                    key, query = key.transpose((0, 1, 3, 2)), query.transpose((0, 1, 3, 2))

                key = paddle.stack(
                    [norm(x) for norm, x in
                     zip(self.norm_K, (key[:, i, ...] for i in range(self.n_head)))], axis=1)
                query = paddle.stack(
                    [norm(x) for norm, x in
                     zip(self.norm_Q, (query[:, i, ...] for i in range(self.n_head)))], axis=1)

                if self.norm_type == 'instance':
                    key, query = key.transpose((0, 1, 3, 2)), query.transpose((0, 1, 3, 2))

        if pos is not None and self.pos_dim > 0:
            assert pos.shape[-1] == self.pos_dim
            pos = pos.unsqueeze(1)
            pos = pos.tile([1, self.n_head, 1, 1])
            query, key, value = [paddle.concat([pos, x], axis=-1)
                                 for x in (query, key, value)]

        if self.attention_type in ['linear', 'galerkin', 'global']:
            x, self.attn_weight = linear_attention(query, key, value,
                                                   mask=mask,
                                                   attention_type=self.attention_type,
                                                   dropout=self.dropout)
        else:
            x, self.attn_weight = vanilla_attention(query, key, value,
                                                    mask=mask,
                                                    attention_type=self.attention_type,
                                                    dropout=self.dropout)

        out_dim = self.n_head * self.d_k if pos is None else self.n_head * \
                                                             (self.d_k + self.pos_dim)
        att_output = x.transpose((0, 2, 1, 3)).reshape((bsz, -1, out_dim))

        if pos is not None and self.pos_dim > 0:
            att_output = self.fc(att_output)

        return att_output, self.attn_weight

    def _reset_parameters(self):
        """
        weight initialize
        """

        for m in self.sublayers():
            if isinstance(m, nn.Linear):
                v = params_initial('xavier_uniform', shape=m.weight.shape, gain=self.xavier_init)
                if self.diagonal_weight > 0.0:
                    v += self.diagonal_weight * \
                         np.diag(np.ones(m.weight.shape[-1], dtype=np.float32))
                if self.symmetric_init:
                    v += v.T
                m.weight.set_value(v)
                v = params_initial('constant', shape=m.bias.shape, gain=0.0)
                m.bias.set_value(v)

    def _get_norm(self, eps):
        """
        batch/layer/instance normalization
        """
        if self.attention_type in ['linear', 'galerkin', 'global']:
            if self.norm_type == 'instance':
                self.norm_K = self._get_instancenorm(self.d_k, self.n_head,
                                                     epsilon=eps)
                self.norm_V = self._get_instancenorm(self.d_k, self.n_head,
                                                     epsilon=eps)
            elif self.norm_type == 'layer':
                self.norm_K = self._get_layernorm(self.d_k, self.n_head,
                                                  epsilon=eps)
                self.norm_V = self._get_layernorm(self.d_k, self.n_head,
                                                  epsilon=eps)
        else:
            if self.norm_type == 'instance':
                self.norm_K = self._get_instancenorm(self.d_k, self.n_head,
                                                     epsilon=eps)
                self.norm_Q = self._get_instancenorm(self.d_k, self.n_head,
                                                     epsilon=eps)
            elif self.norm_type == 'layer':
                self.norm_K = self._get_layernorm(self.d_k, self.n_head,
                                                  epsilon=eps)
                self.norm_Q = self._get_layernorm(self.d_k, self.n_head,
                                                  epsilon=eps)

    @staticmethod
    def _get_layernorm(normalized_dim, n_head, **kwargs):
        """
        layer normalization
        """
        return nn.LayerList(
            [copy.deepcopy(nn.LayerNorm(normalized_dim, **kwargs)) for _ in range(n_head)])

    @staticmethod
    def _get_instancenorm(normalized_dim, n_head, **kwargs):
        """
        instance normalization
        """
        return nn.LayerList(
            [copy.deepcopy(nn.InstanceNorm1D(normalized_dim, **kwargs)) for _ in range(n_head)])


class SimpleTransformerEncoderLayer(nn.Layer):
    """
    EncoderLayer for transformer
    """

    def __init__(self,
                 d_model=96,
                 pos_dim=1,
                 n_head=2,
                 dim_feedforward=512,
                 attention_type='fourier',
                 pos_emb=False,
                 layer_norm=True,
                 attn_norm=True,
                 norm_type='layer',
                 norm_eps=1e-5,
                 batch_norm=False,
                 attn_weight=False,
                 xavier_init: float = 1e-2,
                 diagonal_weight: float = 1e-2,
                 symmetric_init=False,
                 residual_type='add',
                 activation_type='relu',
                 dropout=0.1,
                 ffn_dropout=0.1,
                 ):
        super(SimpleTransformerEncoderLayer, self).__init__()

        self.attn = SimpleAttention(n_head=n_head,
                                    d_model=d_model,
                                    attention_type=attention_type,
                                    diagonal_weight=diagonal_weight,
                                    xavier_init=xavier_init,
                                    symmetric_init=symmetric_init,
                                    pos_dim=pos_dim,
                                    norm_add=attn_norm,
                                    norm_type=norm_type,
                                    eps=norm_eps,
                                    dropout=dropout)
        self.d_model = d_model
        self.n_head = n_head
        self.pos_dim = pos_dim
        self.add_layer_norm = layer_norm
        if layer_norm:
            self.layer_norm1 = nn.LayerNorm(d_model, epsilon=norm_eps)
            self.layer_norm2 = nn.LayerNorm(d_model, epsilon=norm_eps)
        dim_feedforward = 2 * d_model
        self.ff = FeedForward(in_dim=d_model,
                              dim_feedforward=dim_feedforward,
                              batch_norm=batch_norm,
                              activation=activation_type,
                              dropout=ffn_dropout,
                              )
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.residual_type = residual_type  # plus or minus
        self.add_pos_emb = pos_emb
        if self.add_pos_emb:
            self.pos_emb = PositionalEncoding(d_model)

        self.attn_weight = attn_weight
        self.__name__ = attention_type.capitalize() + 'TransformerEncoderLayer'

    def forward(self, x, pos=None, weight=None):
        '''
        - x: node feature, (batch_size, seq_len, n_feats)
        - pos: position coords, needed in every head

        Remark:
            - for n_head=1, no need to encode positional
            information if coords are in features
        '''
        if self.add_pos_emb:
            x = x.transpose((1, 0, 2))
            x = self.pos_emb(x)
            x = x.transpose((1, 0, 2))

        if pos is not None and self.pos_dim > 0:
            att_output, attn_weight = self.attn(x, x, x, pos=pos, weight=weight)  # encoder no mask
        else:
            att_output, attn_weight = self.attn(x, x, x, weight=weight)

        if self.residual_type in ['add', 'plus'] or self.residual_type is None:
            x = x + self.dropout1(att_output)
        else:
            x = x - self.dropout1(att_output)
        if self.add_layer_norm:
            x = self.layer_norm1(x)

        x1 = self.ff(x)
        x = x + self.dropout2(x1)

        if self.add_layer_norm:
            x = self.layer_norm2(x)

        if self.attn_weight:
            return x, attn_weight
        else:
            return x


class PointwiseRegressor(nn.Layer):
    '''
    A wrapper for a simple pointwise linear layers
    '''

    def __init__(self, in_dim,  # input dimension
                 n_hidden,
                 out_dim,  # number of target dim
                 num_layers: int = 2,
                 spacial_fc: bool = False,
                 spacial_dim: int = 1,
                 dropout=0.1,
                 activation='gelu',
                 return_latent=False,
                 debug=False):
        super(PointwiseRegressor, self).__init__()

        self.spacial_fc = spacial_fc
        activ = activation_dict[activation]
        if self.spacial_fc:
            in_dim = in_dim + spacial_dim
            self.fc = nn.Linear(in_dim, n_hidden)
        else:
            self.fc = nn.Linear(in_dim, n_hidden)

        self.ff = nn.LayerList([nn.Sequential(
            nn.Linear(n_hidden, n_hidden),
            activ,
        )])
        for _ in range(num_layers - 1):
            self.ff.append(nn.Sequential(
                nn.Linear(n_hidden, n_hidden),
                activ,
            ))
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(n_hidden, out_dim)
        self.return_latent = return_latent
        self.debug = debug

    def forward(self, x, grid=None):
        '''
        2D:
            Input: (-1, n, n, in_features)
            Output: (-1, n, n, n_targets)
        1D:
            Input: (-1, n, in_features)
            Output: (-1, n, n_targets)
        '''
        if self.spacial_fc:
            x = paddle.concat([x, grid], axis=-1)
        x = self.fc(x)

        for layer in self.ff:
            x = layer(x)
            x = self.dropout(x)

        x = self.out(x)

        if self.return_latent:
            return x, None
        else:
            return x


class SpectralRegressor(nn.Layer):
    '''
    A wrapper for both SpectralConv1d and SpectralConv2d
    Ref: Li et 2020 FNO paper
    https://github.com/zongyi-li/fourier_neural_operator/blob/master/fourier_2d.py
    A new implementation incoporating all spacial-based FNO
    in_dim: input dimension, (either n_hidden or spacial dim)
    n_hidden: number of hidden features out from attention to the fourier conv
    '''

    def __init__(self, in_dim,
                 n_hidden,
                 freq_dim,
                 out_dim,
                 modes: int,
                 num_spectral_layers: int = 2,
                 n_grid=None,
                 dim_feedforward=None,
                 spacial_fc=False,
                 spacial_dim=2,
                 return_freq=False,
                 return_latent=False,
                 normalizer=None,
                 activation='gelu',
                 last_activation=True,
                 dropout=0.1,
                 debug=False):
        super(SpectralRegressor, self).__init__()

        spectral_conv = SpectralConv2d
        # activation = default(activation, 'silu')
        self.activation = activation_dict[activation]

        self.spacial_fc = spacial_fc  # False in Transformer
        if self.spacial_fc:
            self.fc = nn.Linear(in_dim + spacial_dim, n_hidden)
        else:
            self.fc = nn.Linear(in_dim, n_hidden)

        self.spectral_conv = nn.LayerList([spectral_conv(in_dim=n_hidden,
                                                         out_dim=freq_dim,
                                                         modes=(modes, modes),
                                                         dropout=dropout,
                                                         activation=activation)])
        for _ in range(num_spectral_layers - 1):
            self.spectral_conv.append(spectral_conv(in_dim=freq_dim,
                                                    out_dim=freq_dim,
                                                    modes=(modes, modes),
                                                    dropout=dropout,
                                                    activation=activation))
        if not last_activation:
            self.spectral_conv[-1].activation = nn.Identity()

        self.n_grid = n_grid  # dummy for debug
        self.dim_feedforward = 2 * spacial_dim * freq_dim
        self.regressor = nn.Sequential(
            nn.Linear(freq_dim, self.dim_feedforward),
            self.activation,
            nn.Linear(self.dim_feedforward, out_dim),
        )
        self.normalizer = normalizer
        self.return_freq = return_freq
        self.return_latent = return_latent
        self.debug = debug

    def forward(self, x, grid=None, edge=None, pos=None):
        '''
        2D:
            Input: (-1, n, n, in_features)
            Output: (-1, n, n, n_targets)
        1D:
            Input: (-1, n, in_features)
            Output: (-1, n, n_targets)
        '''
        x_latent = []
        x_fts = []

        if self.spacial_fc:
            x = paddle.concat([x, grid], axis=-1)

        x = self.fc(x)
        if len(x.shape) == 3:
            x = x.transpose((0, 2, 1))
        else:
            x = x.transpose((0, 3, 1, 2))

        for layer in self.spectral_conv:
            if self.return_freq:
                x, x_ft = layer(x)
                x_fts.append(x_ft.contiguous())
            else:
                x = layer(x)

            if self.return_latent:
                x_latent.append(x.contiguous())

        if len(x.shape) == 3:
            x = x.transpose((0, 2, 1))
        elif len(x.shape) == 4:
            x = x.transpose((0, 2, 3, 1))

        x = self.regressor(x)

        if self.normalizer:
            x = self.normalizer.inverse_transform(x)

        if self.return_freq or self.return_latent:
            return x, dict(preds_freq=x_fts, preds_latent=x_latent)
        else:
            return x


class FourierTransformer2D(nn.Layer):
    def __init__(self, **kwargs):
        super(FourierTransformer2D, self).__init__()
        self.config = defaultdict(lambda: None, **kwargs)
        self._get_setting()
        self._initialize()
        self.__name__ = self.attention_type.capitalize() + 'Transformer2D'

    def pre_process(self, x):
        """
        Args:
            x: input coordinates
        Returns:
            res: input transform
        """
        shape = x.shape
        batchsize, size_x, size_y = shape[0], shape[2], shape[3]
        gridx = paddle.linspace(0, 1, size_x, dtype=paddle.float32)
        gridx = gridx.reshape((1, size_x, 1, 1)).tile([batchsize, 1, size_y, 1])
        gridy = paddle.linspace(0, 1, size_y, dtype=paddle.float32)
        gridy = gridy.reshape((1, 1, size_y, 1)).tile([batchsize, size_x, 1, 1])
        grid = paddle.concat((gridx, gridy), axis=-1)
        return x.transpose((0, 2, 3, 1)), grid

    def forward(self, node, weight=None):
        '''
        - node: (batch_size, n, n, node_feats)
        - pos: (batch_size, n_s1*n_s2, pos_dim)
        - edge: (batch_size, n_s1*n_s2, n_s1*n_s2, edge_feats)
        - weight: (batch_size, n_s1*n_s2, n_s1*n_s2): mass matrix prefered
            or (batch_size, n_s1*n_s2) when mass matrices are not provided (lumped mass)
        - grid: (batch_size, n, n, 2) excluding boundary
        '''

        node, pos = self.pre_process(node)
        bsz = node.shape[0]
        n_s1 = node.shape[1]
        n_s2 = node.shape[2]
        x_latent = []
        attn_weights = []

        # if not self.downscaler_size:
        node = paddle.concat([node, pos], axis=-1)
        x = self.downscaler(node)
        x = x.reshape((bsz, -1, self.n_hidden))
        pos = pos.reshape((bsz, -1, pos.shape[-1]))

        x = self.feat_extract(x)
        x = self.dpo(x)

        for encoder in self.encoder_layers:
            if self.return_attn_weight and self.attention_type != 'official':
                x, attn_weight = encoder(x, pos, weight)
                attn_weights.append(attn_weight)
            elif self.attention_type != 'official':
                x = encoder(x, pos, weight)
            else:
                out_dim = self.n_head * self.pos_dim + self.n_hidden
                x = x.reshape((bsz, -1, self.n_head, self.n_hidden // self.n_head)).transpose((0, 2, 1, 3))
                x = paddle.concat([pos.tile([1, self.n_head, 1, 1]), x], axis=-1)
                x = x.transpose((0, 2, 1, 3)).reshape((bsz, -1, out_dim))
                x = encoder(x)
            if self.return_latent:
                x_latent.append(x)

        x = x.reshape((bsz, n_s1, n_s2, self.n_hidden))
        x = self.upscaler(x)

        if self.return_latent:
            x_latent.append(x)

        x = self.dpo(x)

        if self.return_latent:
            x, xr_latent = self.regressor(x)
            x_latent.append(xr_latent)
        else:
            x = self.regressor(x)

        if self.normalizer:
            x = self.normalizer.inverse_transform(x)

        return x.transpose((0, 3, 1, 2))

    def _initialize(self):
        self._get_feature()
        self._get_scaler()
        self._get_encoder()
        self._get_regressor()
        self.config = dict(self.config)

    def print_config(self):
        for a in self.config.keys():
            if not a.startswith('__'):
                print(f"{a}: \t", getattr(self, a))

    @staticmethod
    def _initialize_layer(layer, gain=1e-2):

        for m in layer.sublayers():
            v = params_initial('xavier_uniform', shape=m.weight.shape, gain=gain)
            m.weight.set_value(v)
            v = params_initial('constant', shape=m.bias.shape, gain=0.0)
            m.bias.set_value(v)

    def _get_setting(self):
        all_attr = list(self.config.keys())
        for key in all_attr:
            setattr(self, key, self.config[key])

        self.dpo = nn.Dropout(self.dropout)
        if self.decoder_type == 'attention':
            self.num_encoder_layers += 1
        self.attention_types = ['fourier', 'integral', 'local', 'global',
                                'cosine', 'galerkin', 'linear', 'softmax']

    def _get_feature(self):

        self.feat_extract = nn.Identity()

    def _get_scaler(self):
        # if self.downscaler_size:
        #     self.downscaler = DownScaler(in_dim=self.node_feats,
        #                                  out_dim=self.n_hidden,
        #                                  downsample_mode=self.downsample_mode,
        #                                  interp_size=self.downscaler_size,
        #                                  dropout=self.downscaler_dropout,
        #                                  activation_type=self.downscaler_activation)
        # else:
        self.downscaler = nn.Linear(self.node_feats + self.spacial_dim, out_features=self.n_hidden)
        # Identity(in_features=self.node_feats+self.spacial_dim, out_features=self.n_hidden)
        # if self.upscaler_size:
        #     self.upscaler = UpScaler(in_dim=self.n_hidden,
        #                              out_dim=self.n_hidden,
        #                              upsample_mode=self.upsample_mode,
        #                              interp_size=self.upscaler_size,
        #                              dropout=self.upscaler_dropout,
        #                              activation_type=self.upscaler_activation)
        # else:
        self.upscaler = nn.Identity()

    def _get_encoder(self):
        if self.attention_type in self.attention_types:
            encoder_layer = SimpleTransformerEncoderLayer(d_model=self.n_hidden,
                                                          n_head=self.n_head,
                                                          attention_type=self.attention_type,
                                                          dim_feedforward=self.dim_feedforward,
                                                          layer_norm=self.layer_norm,
                                                          attn_norm=self.attn_norm,
                                                          batch_norm=self.batch_norm,
                                                          pos_dim=self.pos_dim,
                                                          xavier_init=self.xavier_init,
                                                          diagonal_weight=self.diagonal_weight,
                                                          symmetric_init=self.symmetric_init,
                                                          attn_weight=self.return_attn_weight,
                                                          dropout=self.encoder_dropout,
                                                          ffn_dropout=self.ffn_dropout,
                                                          norm_eps=self.norm_eps)
        else:
            raise NotImplementedError("encoder type not implemented.")
        self.encoder_layers = nn.LayerList(
            [copy.deepcopy(encoder_layer) for _ in range(self.num_encoder_layers)])

    def _get_regressor(self):
        if self.decoder_type == 'pointwise':
            self.regressor = PointwiseRegressor(in_dim=self.n_hidden,
                                                n_hidden=self.n_hidden,
                                                out_dim=self.n_targets,
                                                num_layers=self.num_regressor_layers,
                                                spacial_fc=self.spacial_fc,
                                                spacial_dim=self.spacial_dim,
                                                activation=self.regressor_activation,
                                                dropout=self.decoder_dropout,
                                                return_latent=self.return_latent)
        elif self.decoder_type == 'ifft2':
            self.regressor = SpectralRegressor(in_dim=self.n_hidden,
                                               n_hidden=self.freq_dim,
                                               freq_dim=self.freq_dim,
                                               out_dim=self.n_targets,
                                               num_spectral_layers=self.num_regressor_layers,
                                               modes=self.fourier_modes,
                                               spacial_dim=self.spacial_dim,
                                               spacial_fc=self.spacial_fc,
                                               activation=self.regressor_activation,
                                               last_activation=self.last_activation,
                                               dropout=self.decoder_dropout,
                                               return_latent=self.return_latent)
        else:
            raise NotImplementedError("Decoder type not implemented")


if __name__ == '__main__':
    # Q = paddle.ones([10, 100, 512])
    # K = paddle.ones([10, 100, 512])
    # V = paddle.ones([10, 100, 512])
    #
    # out, _ = vanilla_attention(Q, K, V)
    #
    # print(out)
    #
    # out, _ = linear_attention(Q, K, V)
    # print(out)
    #
    # layer = SimpleAttention(n_head=8, d_model=512, attention_type='galerkin', norm_type='instance', norm_add=True)
    # y, _ = layer(Q, K, V)
    #
    # Postional = PositionalEncoding(d_model=512)
    # x = y.transpose((1, 0, 2))
    # y = Postional(x)
    # y = y.transpose((1, 0, 2))
    #
    # EncoderLayer = SimpleTransformerEncoderLayer(d_model=512)
    # x = paddle.ones([10, 100, 512])
    # out = EncoderLayer(x)
    # print(out)
    #
    # layer = PointwiseRegressor(in_dim=5, out_dim=3, n_hidden=128)
    # x = paddle.ones([10, 64, 64, 5], dtype=paddle.float32)
    # x = layer(x)
    # print(x.shape)

    import yaml

    with open(os.path.join('transformer_config.yml')) as f:
        config = yaml.full_load(f)

    config = config['Transformer']

    Net_model = FourierTransformer2D(**config)
    x = paddle.ones([10, 3, 64, 64], dtype=paddle.float32)
    x = Net_model(x)
    print(x.shape)
