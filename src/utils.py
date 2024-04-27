from multiprocessing import shared_memory
import umap
from scipy import stats
import torch
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import anndata
from scipy import sparse
from sklearn import neighbors
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
import scanpy as sc
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
# from .fix_progress import FixTqdmProgress
from modules import LitSubcunet
from dataset import AnnDataDataset
# from subcunet import modules


def make_disc(values, res, min_val):
    return ((values - min_val)/ res)


def xy2gird_idxs(disc_x, disc_y, y_size):
    return disc_x * y_size + disc_y


def original_disc_x(grid_idx, y_size):
    return grid_idx // y_size


def original_disc_y(grid_idx, y_size):
    return grid_idx % y_size


# def custom_collate_fn(batch):
#     x = [item[0] for item in batch]
#     norm_mat = [item[1] for item in batch]
#     x_2d = [item[2] for item in batch]
#     x = torch.stack(x)
#     norm_mat = torch.stack(norm_mat)
#     x_2d = torch.stack(x_2d)
#     return x, norm_mat, x_2d


def custom_collate_fn(batch):
    x = [item[0] for item in batch]
    norm_mat = [item[1] for item in batch]
    x_2d = [item[2] for item in batch]
    x = torch.stack(x)
    norm_mat = torch.stack(norm_mat)
    x_2d = pad_sequence(x_2d, batch_first=True)
    return x, norm_mat, x_2d


def prepare_datasets(grid_adata, pcell_adata, c_size, unique_cells, split_ends, split_name):    
    start, end = split_ends
    grid_mask = grid_adata.obs['cell_idx'].isin(unique_cells[start:end])
    grid_indices = np.where(grid_mask)[0]
    split_adata = grid_adata[grid_mask].copy()
    unique_indices = [np.where(pcell_adata.obs['cell_idx'] == idx)[0][0] for idx in unique_cells[start:end]]
    pcell_mask_mat = pcell_adata.obsm['mask_mat'][unique_indices, :]
    pcell_mask_mat = pcell_mask_mat[:, grid_indices]
    split_pcell_adata = pcell_adata[pcell_adata.obs['cell_idx'].isin(split_adata.obs['cell_idx'].unique())].copy()
    split_pcell_adata.obsm['mask_mat'] = pcell_mask_mat.reshape(split_pcell_adata.n_obs, -1)
    return AnnDataDataset(split_adata, split_pcell_adata, c_size)


#, collate_fn=custom_collate_fn
def optimize_subcunet(grid_adata, pcell_adata, model_params, split_by_cell_idx=False, test_ratio=0.05, val_ratio=0.1, epoch=200, patience=10, gpus=1, batch_size=16, init_model_path=None, use_mask=True):

    c_size = model_params['c_size'] ** 0.5
    if split_by_cell_idx and grid_adata is not None:
        np.random.seed(42)
        unique_cells = grid_adata.obs['cell_idx'].unique()
        np.random.shuffle(unique_cells)
        total_cells = len(unique_cells)
        test_end = int(total_cells * test_ratio)
        val_end = test_end + int(total_cells * val_ratio)
        test_ds = prepare_datasets(grid_adata, pcell_adata, c_size, unique_cells, (0, test_end), 'test')
        val_ds = prepare_datasets(grid_adata, pcell_adata, c_size, unique_cells, (test_end, val_end), 'val')
        train_ds = prepare_datasets(grid_adata, pcell_adata, c_size, unique_cells, (val_end, total_cells), 'train')
        model_params['gene_num'] = train_ds.gene_num

    else:
        ds = AnnDataDataset(grid_adata, pcell_adata, c_size)
        total_size = len(ds)
        test_size = int(total_size * test_ratio)
        val_size = int(total_size * val_ratio)
        train_size = total_size - test_size - val_size
        train_ds, val_ds, test_ds = torch.utils.data.dataset.random_split(ds, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(42))
        model_params['gene_num'] = ds.gene_num

    train_loader = DataLoader(train_ds, batch_size=batch_size, collate_fn=custom_collate_fn, num_workers=16, pin_memory=True, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, collate_fn=custom_collate_fn, num_workers=16, pin_memory=True)
    #model_params['cell_num'] = ds.cell_num
    # setup trainer
    checkpoint_callback = ModelCheckpoint(dirpath='lightning_logs/ckpt', monitor='val_loss', )
    trainer = pl.Trainer(max_epochs=epoch, devices=1, accelerator="gpu", callbacks=[EarlyStopping(monitor="val_loss", patience=patience), checkpoint_callback])
    # setup pytorch_lightning module
    # lit_cubictr = modules.LitSubcunet(**model_params)
    lit_cubictr = LitSubcunet(**model_params)
    if init_model_path is not None:
        print(f'Using initial model in {init_model_path}')
        lit_cubictr.load_state_dict(torch.load(init_model_path))
    print('Start first opt')
    trainer.fit(lit_cubictr, train_loader, val_loader)
    #lit_cubictr = modules.LitSubcunet.load_from_checkpoint(**model_params, checkpoint_path=checkpoint_callback.best_model_path)
    lit_cubictr = LitSubcunet.load_from_checkpoint(**model_params, checkpoint_path=checkpoint_callback.best_model_path)
    return lit_cubictr, train_ds, val_ds, test_ds


def pickup_exp_genes(adata, cell, min_exp):
    tot_exps = pd.Series(adata[cell].X.sum(axis=0), index=adata.var_names)
    exp_genes = tot_exps.index[tot_exps > min_exp]
    return exp_genes


@torch.no_grad()
def extract_cellwise_info(model, ds, batch_size=128):
    ld_list = []
    zl_list = []
    p_list = []
    loader = DataLoader(ds, batch_size=batch_size, num_workers=16, pin_memory=False)
    for exp_vec, norm_vec, x_2d in loader:
        h, h_2d_flat = model.embed_exp(x_2d)
        qz_mu, qz = model.exp_encoder(h)
        tot_ld = model.exp_decoder(qz_mu)
        p, res = model.sample_pi(qz_mu)
        ld_list.append(tot_ld)
        zl_list.append(qz_mu)
        p_list.append(p)
    all_ld = torch.cat(ld_list, dim=0).numpy()
    all_zl = torch.cat(zl_list, dim=0).numpy()
    all_p = torch.cat(p_list, dim=0).numpy()
    return all_ld, all_zl, all_p


@torch.no_grad()
def extract_spotwise_info(model, ds, batch_size=32):
    ld_list = []
    zl_list = []
    zv_list = []
    pi_list = []
    loader = DataLoader(ds, batch_size=batch_size, num_workers=16, pin_memory=False)
    for exp_vec, norm_vec, x_2d in loader:
        pi_2d, z_2d = model.get_pi_z_2d(x_2d)
        #ld_2d = model.get_ld(x_2d)
        ld_2d, ld_2d_exp, ld_2d_mask, masks = model.get_ld(x_2d, norm_vec)
        pi = pi_2d
        ld_list.append(ld_2d)
        zl_list.append(z_2d)
        pi_list.append(pi)
    all_ld = torch.cat(ld_list, dim=0).numpy()
    all_zl = torch.cat(zl_list, dim=0).numpy()
    all_pi = torch.cat(pi_list, dim=0).numpy()
    return all_ld, all_zl, all_pi


def matrix_pearson(X, Y, axis=0):
    zx = stats.zscore(X, axis=axis)
    zy = stats.zscore(Y, axis=axis)
    return (zx * zy).mean(axis=axis)


def embed_z(z_mat, n_neighbors=30, min_dist=0.3):
    if z_mat.shape[1] != 2:
        reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist)
        z_embed = reducer.fit_transform(z_mat)
    else:
        z_embed = np.array(z_mat)
    return(z_embed)


def extract_enrich_ranks(adata, tot_adata, cluster, cluster_key='leiden'):
    totals = pd.Series(tot_adata.X.mean(axis=0), index=adata.var_names)
    target_totals = pd.Series(adata[adata.obs[cluster_key] == cluster].X.mean(axis=0), index=adata.var_names)
    scores_df = pd.DataFrame({'tareget': target_totals, 'total': totals, 'ratio': target_totals / totals}, index=adata.var_names)
    scores_df = scores_df.sort_values('ratio', ascending=False)
    return scores_df


def make_cont_table(targets, alls, geneset):
    tp_genes = np.intersect1d(targets, geneset)
    tp = tp_genes.shape[0]
    fp = targets.shape[0] - tp
    fn = np.intersect1d(geneset, alls).shape[0] - tp
    tn = alls.shape[0] - fn
    tb = np.array([[tp, fp], [fn, tn]])
    return tb, tp_genes

def fisher_geneset(targets, alls, genset):
    tb, tp_genes = make_cont_table(targets, alls, genset)
    odds, pval = stats.fisher_exact(tb)
    return pval, tp_genes


def select_apex_markers(apex_df, comp, topn):
    sub_apex_df = apex_df.loc[np.logical_and(apex_df[f'{comp}_qval'] < 0.2, apex_df[f'{comp}_b'] > 0)]
    top_genes = sub_apex_df.gene_name[sub_apex_df[f'{comp}_b'].rank(ascending=False) < topn].values
    return top_genes

def extract_apex_comps(apex_df):
    return apex_df.columns[6:][::3].str.replace('_tpm', '').values


def extract_spotwise_exp(ds, batch_size=16):
    x_2d_list = []
    loader = DataLoader(ds, batch_size=batch_size, num_workers=16, pin_memory=False)
    for exp_vec, norm_vec, x_2d in loader:
        x_2d_list.append(x_2d)
    all_x_2d = torch.cat(x_2d_list, dim=0).numpy()
    return all_x_2d


def calculate_ld_corr(ld_2d, x_2d):
    ld_2d = ld_2d.reshape(-1, ld_2d.shape[-1])
    x_2d = x_2d.reshape(-1, x_2d.shape[-1])
    tot_counts = x_2d.sum(axis=1)
    corr_vec = (stats.zscore(x_2d[tot_counts > 0], axis=0) * stats.zscore(ld_2d[tot_counts > 0], axis=0)).mean(axis=0)
    return corr_vec