import os
import torch
import scanpy as sc
import numpy as np
import csv
from datetime import datetime
from scipy.sparse import csr_matrix

from dataset import AnnDataDataset
from postprocess import zero_out_grids, signal_imputation, plot_data_grids, get_channel_bounds_and_coordinates, compute_signal_matrix, reconstruct_signals, plot_signal_strength_map
from utils import optimize_subcunet

# Define file paths and directories
base_directory = "/home/majima/Analysis/Data/NabulaLoom/20240424_subcellular"
results_directory = "/home/majima/Analysis/Results/NebulaLoom"
model_params = {
    "z_dim": 10,
    "p_dim": 10,
    "h_dim": 32,
    "c_size": 32**2,
    "vamp_mode": True,
    "enc_comps": 1,
    "use_mask": True
}
batch_size = 32

# Load data
grid_filepath = f"{base_directory}/2024042401_grid_adata_filled.h5ad"
pcell_filepath = f"{base_directory}/2024042401_pcell_adata.h5ad"
grid_adata = sc.read_h5ad(grid_filepath)
pcell_adata = sc.read_h5ad(pcell_filepath)

# Initialize and load model
current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
save_directory = f"{results_directory}/{current_datetime}_subcvae"
os.makedirs(save_directory, exist_ok=True)
model_save_path = os.path.join(save_directory, "opt_params.pth")

# Save the required information
csv_path = os.path.join(save_directory, "model_details.csv")
try:
    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Parameter', 'Value'])
        writer.writerows([
            ('z_dim', model_params['z_dim']),
            ('p_dim', model_params['p_dim']),
            ('h_dim', model_params['h_dim']),
            ('c_size', model_params['c_size']),
            ('vamp_mode', model_params['vamp_mode']),
            ('enc_comps', model_params['enc_comps']),
            ('use_mask', model_params['use_mask']),
            ('Batch Size', batch_size),
            ('Explanation', 'repeat_num 2000, 0.1 single mask, 0.1 betamask, Concat the pooled value to skip_connection, skip beta_anti_mask, reverse anti_mask')
        ])
except Exception as e:
    # If an error occurs, append an error message row at the end of the CSV file
    with open(csv_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Error', str(e)])

# Optimize model
lit_cubictr, train_ds, val_ds, test_ds = optimize_subcunet(grid_adata, pcell_adata, model_params, split_by_cell_idx=False, epoch=300, gpus=[0, 1], patience=20, batch_size=batch_size)
torch.save(lit_cubictr.state_dict(), model_save_path)
lit_cubictr.load_state_dict(torch.load(model_save_path))
lit_cubictr.eval()

# Prepare for down analysis Perform inference
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
exp_vec, norm_vec, exp_2d = test_ds[0]
exp_vec, norm_vec, exp_2d = exp_vec.to(device), norm_vec.to(device), exp_2d.to(device)
x_2d = exp_2d.unsqueeze(0)

# Perform down analysis with two different noise scenarios
for i in range(2):
    mask_ratio = 0.1 if i == 0 else 0
    masked_x_2d = zero_out_grids(x_2d, mask_ratio)
    exp_2d_noised = masked_x_2d.squeeze(0)
    ld_2d = signal_imputation(lit_cubictr, exp_2d_noised.unsqueeze(0), masked_x_2d, device)
    filename = os.path.join(save_directory, f"plot{i + 1}.png")
    ld_2d = ld_2d.squeeze(0)
    plot_data_grids(exp_2d, ld_2d, grid_adata, filename=filename, exp_2d_noised=exp_2d_noised, consistent_color_scale=True)

# Downsampled data processing (Example for additional scenarios)
down_grid_filepath = "/home/majima/Analysis/Data/NabulaLoom/20240331_subcellular_downsample/202403301517_grid_adata.h5ad"
down_pcell_filepath = "/home/majima/Analysis/Data/NabulaLoom/20240331_subcellular_downsample/202403301517_pcell_adata.h5ad"
grid_adata = sc.read_h5ad(down_grid_filepath)
pcell_adata = sc.read_h5ad(down_pcell_filepath)
grid_adata.layers['count'] = csr_matrix(grid_adata.X)
pcell_adata.layers['count'] = csr_matrix(pcell_adata.X)

# Example visualization and reconstruction for a selected object
selected_obj_id = 248102
row_indices = np.where(grid_adata.obs['obj_ids'] == selected_obj_id)
selected_adata = grid_adata[row_indices]
min_x, max_x, min_y, max_y, channels = get_channel_bounds_and_coordinates(selected_adata)
signal_matrix = compute_signal_matrix(selected_adata)
reconstructed_signals = reconstruct_signals(signal_matrix, lit_cubictr, batch_size)

for i in range(signal_matrix.shape[2]):
    plot_signal_strength_map(signal_matrix, i, os.path.join(save_directory, f"cell_observation_{i:04d}.png"), upper_limit=signal_matrix[:, :, i].max())
    plot_signal_strength_map(reconstructed_signals, i, os.path.join(save_directory, f"cell_reconstruction_{i:04d}.png"), upper_limit=signal_matrix[:, :, i].max())
