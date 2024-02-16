import os
import sys
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import scipy
import torch.distributions as dist
import torch
import torch.nn as nn



def input_checks(adata, kinetics=True):
    if kinetics:
        if (not 'spliced' in adata.layers.keys()) or (not 'unspliced' in adata.layers.keys()):
            raise ValueError(
                f'Input anndata object need to have layers named `spliced` and `unspliced`.')
        if np.sum((adata.layers['spliced'] - adata.layers['spliced'].astype(int)))**2 != 0:
            raise ValueError('layers `spliced` includes non integer number, while count data is required for `spliced`.')

        if np.sum((adata.layers['unspliced'] - adata.layers['unspliced'].astype(int)))**2 != 0:
            raise ValueError('layers `unspliced` includes non integer number, while count data is required for `unspliced`.')
    else:
        if np.sum((adata.layers['raw_count'] - adata.layers['raw_count'].astype(int)))**2 != 0:
            raise ValueError('layers `raw_count` includes non integer number, while count data is required for `raw_count`.')

    return(adata)


def define_exp(
        adata,
        model_params={
            'x_dim': 100,
            'z_dim': 10,
            'enc_z_h_dim': 50, 'dec_z_h_dim': 50, 'enc_d_h_dim': 50,
            'num_enc_z_layers': 2, 'num_dec_z_layers': 2,
            'num_enc_d_layers': 2, 't_num': 2
        },
        lr=0.0001, val_ratio=(1/12), test_ratio=(1/12),
        batch_size=20, num_workers=1):
    # splice
    select_adata = adata[:, adata.var['LineageVAE_used']]
    if 'spliced' in select_adata.layers:
        if type(select_adata.layers['spliced']) == np.ndarray:
            s = torch.tensor(select_adata.layers['spliced']).astype('float64')
        else:
            s = torch.tensor(select_adata.layers['spliced'].astype('float64').toarray())
    else:
        if type(select_adata.layers['raw_count']) == np.ndarray:
            s = torch.tensor(select_adata.layers['raw_count']).astype('float64')
        else:
            s = torch.tensor(select_adata.layers['raw_count'].astype('float64').toarray())
    # unspliced
    if 'unspliced' in select_adata.layers:
        if type(select_adata.layers['spliced']) == np.ndarray:
            u = torch.tensor(select_adata.layers['unspliced']).astype('float64')
        else:
            u = torch.tensor(select_adata.layers['unspliced'].astype('float64').toarray())
    else:
        # Create tensor filled with zeros of the same shape as s
        u = torch.zeros_like(s)

    # day
    day = torch.tensor(np.array(select_adata.obs['Day']).astype('float64'))

    # meta df
    model_params['x_dim'] = s.shape[1]
    model_params['t_num'] = int(adata.obs['Day'].max())

    validation_ratio = val_ratio

    return (model_params, lr, s, u, day, test_ratio, batch_size, num_workers, validation_ratio)
