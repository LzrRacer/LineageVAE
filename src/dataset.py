import random
import pandas as pd
import scanpy as sc
import numpy as np
import anndata
import torch

def select_anndata(adata, batch_size=20):

    selected_cells = []

    lineage_counts = adata.obs['Lineage'].value_counts()
    lineage_with_30_cells = lineage_counts[lineage_counts >= batch_size].index

    selected_lineages = []
    for lineage in lineage_with_30_cells:
        if any(adata.obs['Day'][adata.obs['Lineage'] == lineage] == 2):
            selected_lineages.append(lineage)

    # Iterate over selected lineages
    for lineage in selected_lineages:
        selected_batch_cells = []
        # Get cells with lineage and Day == 2
        cells_with_day2 = adata.obs.index[(adata.obs['Lineage'] == lineage) & (adata.obs['Day'] == 2)]
        if len(cells_with_day2) > 0:
            # Randomly select one cell from cells_with_day2
            selected_day2_cell = random.choice(cells_with_day2)
            # Add selected_day2_cell to selected_cells
            selected_batch_cells.append(selected_day2_cell)

        # Shuffle the selected_cells
        #random.shuffle(selected_cells)

        cells_in_lineage = adata.obs.index[(adata.obs['Lineage'] == lineage)]

        # Remove selected_day2_cell from cells_in_lineage
        cells_in_lineage = cells_in_lineage[cells_in_lineage != selected_day2_cell]

        # Randomly extract batch_size - 1 cells from cells_in_lineage
        rest_of_batch_cells = random.sample(cells_in_lineage.tolist(), (int(batch_size) - 1))

        # Add back selected_day2_cell to selected_batch_cells
        selected_batch_cells.extend(rest_of_batch_cells)

        if len(selected_batch_cells) == batch_size:
            # Extend selected_cells with selected_batch_cells
            selected_cells.extend(selected_batch_cells)
        else:
            print('lineage number error')
            print(len(selected_batch_cells))

    # Create selected_adata by concatenating selected_cells
    selected_adata = adata[selected_cells]

    return(selected_adata)


def heldout_anndata(adata, batch_size=20):

    selected_cells = []
    rest_of_cells = []
    selected_lineages = []
    # Iterate over lineages
    for lineage in adata.obs['Lineage'].unique():
        # Filter cells for the current lineage
        lineage_cells = adata.obs.index[adata.obs['Lineage'] == lineage]

        # Check if the lineage has at least one cell with ['day'] == 2
        if any(adata.obs['Day'][lineage_cells] == 2):
            # Filter cells with ['day'] == 2 and ['day'] == 6
            day2_cells = lineage_cells[adata.obs['Day'][lineage_cells] == 2]
            day6_cells = lineage_cells[adata.obs['Day'][lineage_cells] == 6]

            # Check if the lineage has more than batch_size cells with ['day'] == 2 and ['day'] == 6
            if len(day2_cells)+len(day6_cells) > batch_size:
                selected_lineages.append(lineage)


    # Iterate over selected lineages
    for lineage in selected_lineages:
        selected_batch_cells = []
        # Get cells with lineage and Day == 2
        cells_with_day2 = adata.obs.index[(adata.obs['Lineage'] == lineage) & (adata.obs['Day'] == 2)]
        if len(cells_with_day2) > 0:
            # Randomly select one cell from cells_with_day2
            selected_day2_cell = random.choice(cells_with_day2)
            # Add selected_day2_cell to selected_cells
            selected_batch_cells.append(selected_day2_cell)

        # Shuffle the selected_cells
        #random.shuffle(selected_cells)

        cells_in_lineage = adata.obs.index[(adata.obs['Lineage'] == lineage) & ((adata.obs['Day'] == 2) | (adata.obs['Day'] == 6))]

        # Remove selected_day2_cell from cells_in_lineage
        cells_in_lineage = cells_in_lineage[cells_in_lineage != selected_day2_cell]

        # Randomly extract batch_size - 1 cells from cells_in_lineage
        rest_of_batch_cells = random.sample(cells_in_lineage.tolist(), (int(batch_size) - 1))

        # Add back selected_day2_cell to selected_batch_cells
        selected_batch_cells.extend(rest_of_batch_cells)

        if len(selected_batch_cells) == batch_size:
            # Extend selected_cells with selected_batch_cells
            selected_cells.extend(selected_batch_cells)
        #else:
        #    print('lineage number error')
        #    print(len(selected_batch_cells))

    # Create selected_adata by concatenating selected_cells
    heldout_adata = adata[selected_cells]

    for lineage in selected_lineages:
        selected_batch_cells = []
        # Get cells with lineage and Day == 4
        cells_day4 = adata.obs.index[(adata.obs['Lineage'] == lineage) & (adata.obs['Day'] == 4)]
        rest_of_cells.extend(cells_day4)

    rest_of_adata = adata[rest_of_cells]

    return(heldout_adata, rest_of_adata)


def anndata_preprocessing(adata, n_top_genes=None, min_genes=None, target_sum=None):
    """
    Preprocesses AnnData object based on provided parameters.

    Parameters:
        adata (AnnData): Annotated data matrix with cells as rows and features as columns.
        n_top_genes (int or None): Number of highly variable genes to select.
        min_genes (int or None): Minimum number of genes expressed in cells to retain a cell.

    Returns:
        None (Modifies the input AnnData object in place)
    """
    adata = adata[~adata.obs['Day'].isna()] # Drop NaN
    sc.pp.log1p(adata) # logarization
    if n_top_genes is not None:
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes)
    else:
        sc.pp.highly_variable_genes(adata)
    sc.pl.highly_variable_genes(adata)
    if min_genes is not None:
        sc.pp.filter_cells(adata, min_genes=min_genes)
    else:
        sc.pp.filter_cells(adata) # Exclude cells expressing too few genes
    if target_sum is not None:
        sc.pp.normalize_total(adata, target_sum=target_sum)
    else:
        sc.pp.normalize_total(adata) # Normalization

    return(adata)


def extract_highly_variable_genes(adata):
    var_list = []
    for i, is_highly_variable in enumerate(adata.var['highly_variable']):
        if is_highly_variable:
            var_list.append(adata.var.index[i])
    return var_list


def select_dataset(adata, batch_size):

    lineage_counts = adata.obs['Lineage'].value_counts()
    lineage_with_30_cells = lineage_counts[lineage_counts >= batch_size].index
    selected_adata = []
    for lineage in lineage_with_30_cells:
        lineage_cells = adata[adata.obs['Lineage'] == lineage]

        # Get unique day values for the selected lineage
        days = np.unique(lineage_cells.obs['Day'])

        # Initialize variables for tracking the extracted cells and checking the required conditions
        selected_cells = []
        selected_days = set()

        # Iterate over unique days and collect cells
        for day in days:
            day_cells = lineage_cells[lineage_cells.obs['Day'] == day]
            random_cell_idx = np.random.choice(len(day_cells), size=1, replace=False)[0]
            selected_cells.extend(day_cells[random_cell_idx])
            selected_days.add(day)

            if len(selected_cells) >= batch_size:
                break

            if len(selected_days) == len(days):
                break

        # Ensure you have at least one lineage represented
        if len(selected_cells) < batch_size:
            additional_cells_needed = batch_size - len(selected_cells)
            selected_cell_identifiers = [cell.obs_names[0] for cell in selected_cells]
            # Filter cells from lineage_cells that are not in selected_cells
            remaining_cells = [cell for cell in lineage_cells if cell.obs_names[0] not in selected_cell_identifiers]
            remaining_cells = anndata.concat(remaining_cells)
            additional_cells_idx = np.random.choice(len(remaining_cells), size=additional_cells_needed, replace=False)[0:additional_cells_needed]
            additional_cells = remaining_cells[additional_cells_idx]
            selected_cells.extend(additional_cells)

        # Now selected_cells contains the desired batch of cells
        selected_cells = anndata.concat(selected_cells)
        selected_adata.append(selected_cells)

    return(selected_adata)


def select_undifferentiated_anndata(adata, selected_adata, undifferentiated_cell_size, initial=False):
    undifferentiated_sample_list = []
    subgroup_indices = np.where((adata.obs['Day'] == 0) | (adata.obs['Day'] == 6))[0]
    subgroup_adata = adata[subgroup_indices]

    if initial:
        for i in range(len(selected_adata)):
            initial_cell_index = random.choice(adata[adata.obs['Day'] == 0].obs.index)
            additional_cells = adata[initial_cell_index]
            undifferentiated_cell_size_i = undifferentiated_cell_size - len(additional_cells)
            selected_cell_identifiers = [cell.obs_names[0] for cell in undifferentiated_sample_list]
            # Filter cells from lineage_cells that are not in selected_cells
            remaining_cells = [cell for cell in subgroup_adata if cell.obs_names[0] not in selected_cell_identifiers]
            remaining_cells = anndata.concat(remaining_cells)
            additional_cells_idx = np.random.choice(len(remaining_cells), size=undifferentiated_cell_size_i, replace=False)[0:undifferentiated_cell_size_i]
            additional_cells = anndata.concat([additional_cells, remaining_cells[additional_cells_idx]])
            additional_cells = anndata.concat(additional_cells)
            undifferentiated_sample_list.append(additional_cells)

        initial_cells_idx = np.where(subgroup_adata.obs['Day'] == 0)[0]
        initial_cells = subgroup_adata[initial_cells_idx]
        undifferentiated_sample_list.append(initial_cells)


    else:
        for i in range(len(selected_adata)):
            selected_cell_identifiers = [cell.obs_names[0] for cell in undifferentiated_sample_list]
            # Filter cells from lineage_cells that are not in selected_cells
            remaining_cells = [cell for cell in subgroup_adata if cell.obs_names[0] not in selected_cell_identifiers]
            remaining_cells = anndata.concat(remaining_cells)
            additional_cells_idx = np.random.choice(len(remaining_cells), size=undifferentiated_cell_size, replace=False)[0:undifferentiated_cell_size]
            additional_cells = remaining_cells[additional_cells_idx]
            additional_cells = anndata.concat(additional_cells)
            undifferentiated_sample_list.append(additional_cells)

    return(undifferentiated_sample_list)


def anndata_for_dynamics_inference(selected_adata, undifferentiated_sample_list=None):
    adata_list = []

    if undifferentiated_sample_list is not None:
        for i in range(len(selected_adata)):
            adata_i = selected_adata[i]
            adata_i = adata_i.concatenate(undifferentiated_sample_list[i])
            adata_list.append(adata_i)
    else:
        for i in range(len(selected_adata)):
            adata_i = selected_adata[i]
            adata_list.append(adata_i)

    select_adata = anndata.concat(adata_list)

    return (select_adata)


class LineageVAEDataSet(torch.utils.data.Dataset):
    def __init__(self, s, u, day, norm_mat, transform=None, pre_transform=None):
        self.s = s
        self.u = u
        self.day = day
        self.norm_mat = norm_mat

    def __len__(self):
        return(self.s.shape[0])

    def __getitem__(self, idx):
        idx_s = self.s[idx]
        idx_u = self.u[idx]
        idx_day = self.day[idx]
        idx_norm_mat = self.norm_mat[idx]
        return(idx_s, idx_u, idx_day, idx_norm_mat)


class LineageVAEDataManager():
    def __init__(self, s, u, day, test_ratio, batch_size, num_workers, t_num, validation_ratio=(1/12)):
        s = s.float()
        u = u.float()
        day = day.float()
        norm_mat = torch.sum(s, dim=1).view(-1, 1) * torch.sum(s, dim=0).view(1, -1)
        norm_mat = torch.mean(s) * norm_mat / torch.mean(norm_mat)
        self.s = s
        self.u = u
        self.day = day
        self.norm_mat = norm_mat
        total_num = s.shape[0]
        validation_num = int(total_num * validation_ratio)
        test_num = int(total_num * test_ratio)
        np.random.seed(42)
        idx = np.arange(total_num)
        validation_idx, test_idx, train_idx = idx[:validation_num], idx[validation_num:(validation_num +  test_num)], idx[(validation_num +  test_num):]
        self.validation_idx, self.test_idx, self.train_idx = validation_idx, test_idx, train_idx
        self.validation_s = s[validation_idx]
        self.validation_u = u[validation_idx]
        self.validation_day = day[validation_idx]
        self.validation_norm_mat = norm_mat[validation_idx]
        self.test_s = s[test_idx]
        self.test_u = u[test_idx]
        self.test_day = day[test_idx]
        self.test_norm_mat = norm_mat[test_idx]
        self.train_eds = LineageVAEDataSet(s[train_idx], u[train_idx], day[train_idx], norm_mat[train_idx])
        self.train_loader = torch.utils.data.DataLoader(
            self.train_eds, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=True) #false
        
        

