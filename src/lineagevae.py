
import argparse
import scanpy as sc
import pandas as pd
import torch
import datetime
import os
import joblib
import numpy as np
from workflow import run_analysis_for_hematopoiesis, check_results, latent_visualization, latent_transition_inferens, main_differentiation_destination
from umap_visualization import save_umap_scatterplot, save_custom_umap_scatterplot, save_celltype_umap_scatterplot, save_transition_scatter_plot
from tf_activity_analysis import process_tf_data, average_ann_data, process_for_tf_analysis, extract_tf_adata, calculate_tg_delta, plot_norm_of_dynamics, tfactivity_glm, create_heatmap_with_annotations

# Argument parser setup
parser = argparse.ArgumentParser(description='Execute hematopoiesis analysis.')
parser.add_argument('--input', required=True, help='Path to input dataset (.h5ad)')
parser.add_argument('--output', help='Output directory')

# Parse arguments
args = parser.parse_args()
input_file_path = args.input
output_directory = args.output

# If output directory is not provided, use current date and time for folder naming
if output_directory is None:
    dt_now = datetime.datetime.now().strftime('%Y%m%d%H%M')
    folder_path = os.path.join('Result', f'{dt_now}_LineageVAE')
else:
    folder_path = output_directory

os.makedirs(folder_path, exist_ok=True)

# Load input data
adata = sc.read(input_file_path, cache=True)
adata_input = adata.copy()
raw_adata_input = adata.copy()

# Run analysis for hematopoiesis
adata, select_adata, LineageVAE_exp, var_list = run_analysis_for_hematopoiesis(
    adata_input, raw_adata_input, select_adata_input=None, var_list_input=None, 
    undifferentiated=2, n_top_genes=1000, first_epoch=500, second_epoch=500, 
    batch_size=20, error_count_limit=3, error_count_ii_limit=2, kinetics=True
)

# Keep running analysis until results meet certain criteria
while check_results(LineageVAE_exp):
    adata, select_adata, LineageVAE_exp, var_list = run_analysis_for_hematopoiesis(
        adata_input, raw_adata_input, select_adata_input=None, var_list_input=None,
        undifferentiated=2, n_top_genes=1000, first_epoch=500, second_epoch=500,
        batch_size=20, error_count_limit=3, error_count_ii_limit=1, kinetics=True
    )

# Save model parameters
model_parameters_path = os.path.join(folder_path, f'{dt_now}_model_parameters.pth')
torch.save(LineageVAE_exp.model.state_dict(), model_parameters_path)

# Save var_list as CSV
var_list_df = pd.DataFrame({'var_names': var_list})
var_list_csv_path = os.path.join(folder_path, f'{dt_now}_var_list.csv')
var_list_df.to_csv(var_list_csv_path, index=False)

# Save var_list as CSV
var_list_df = pd.DataFrame({'var_names': var_list})
var_list_csv_path = os.path.join(folder_path, dt_now + '_var_list.csv')
var_list_df.to_csv(var_list_csv_path, index=False)

# Save select_adata
select_adata_path = os.path.join(folder_path, dt_now + '_select_adata.h5ad')
select_adata.write(select_adata_path)

# Save check_adata
adata = sc.read(input_file_path, cache=True)
adata = adata[:,var_list]
check_adata, trans = latent_visualization(adata, LineageVAE_exp, adata.var_names) 
check_adata_path = os.path.join(folder_path, dt_now + '_check_adata.h5ad')
check_adata.write(check_adata_path)

trans_path = os.path.join(folder_path, dt_now + '_trans_adata.h5ad')
trans_path = os.path.join(folder_path, f'{dt_now}_trans.joblib')
joblib.dump(trans, trans_path)
