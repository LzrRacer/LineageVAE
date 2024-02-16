import pandas as pd
import numpy as np
import anndata as ad
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
import time
import os
from workflow import latent_transition_inferens


def process_tf_data(tf_target, inf_adata):
    # Modify TF and Target symbols
    tf_target['TF_Symbol'] = tf_target['TF_Symbol'].apply(lambda x: x[0] + x[1:].lower())
    tf_target['Target_Symbol'] = tf_target['Target_Symbol'].apply(lambda x: x[0] + x[1:].lower())

    # Get TFs and TGs
    TFs = list(set(inf_adata.var_names.tolist()) & set(tf_target['TF_Symbol'].tolist()))
    TGs = list(set(inf_adata.var_names.tolist()) & set(tf_target['Target_Symbol'].tolist()))

    # Filter adata for relevant genes and cells
    tf_adata = inf_adata[:, list(set(TGs) | set(TFs))]
    day_six_cell_ids = tf_adata[tf_adata.obs['Day'] == 6].obs['CellID']
    tf_adata = tf_adata[tf_adata.obs['CellID'].isin(day_six_cell_ids)]
    days = np.unique(tf_adata.obs['Day'])
    days = days[days >= 0]
    tf_adata = tf_adata[tf_adata.obs['Day'].isin(days)]
    tf_adata = tf_adata[tf_adata.obs['reconst'] == 1]

    # Filter TF target data for relevant genes
    tf_mask = np.isin(tf_target['TF_Symbol'], TFs)
    filtered_tf_target = tf_target[tf_mask]
    targets_mask = np.isin(filtered_tf_target['Target_Symbol'], TGs)
    tf_target = filtered_tf_target[targets_mask]

    # Get TG matrix
    tg_indices = [idx for idx, gene in enumerate(tf_adata.var_names) if gene in TGs]
    tg_matrix = tf_adata.X[:, tg_indices].toarray()
    tg_matrix = pd.DataFrame(tg_matrix, index=tf_adata.obs_names, columns=TGs)

    # Get regulation matrix
    matrix = np.zeros((len(TFs), len(TGs)))
    for i, tf in enumerate(TFs):
        for j, target in enumerate(TGs):
            if target in tf_target.loc[tf_target['TF_Symbol'] == tf]['Target_Symbol'].tolist():
                matrix[i, j] = 1
    regulation_matrix = pd.DataFrame(matrix, index=TFs, columns=TGs)

    # Get TF expression data
    tf_indices = [idx for idx, gene in enumerate(tf_adata.var_names) if gene in TFs]
    tf_expression = tf_adata.X[:, tf_indices].toarray()
    tf_expression_df = pd.DataFrame(tf_expression, index=tf_adata.obs_names, columns=TFs)

    return regulation_matrix, tf_expression_df, tg_matrix


import anndata as ad
from scipy.sparse import csr_matrix, issparse, vstack

def average_ann_data(anndata_list):
    # Check if the list is not empty
    if not anndata_list:
        return None

    # Initialize variables for summing X matrices and counting the number of matrices
    sum_X = None
    num_matrices = 0

    for adata in anndata_list:
        # Check if the X attribute is a sparse matrix (CSR format)
        if adata.X is not None and issparse(adata.X):
            if sum_X is None:
                sum_X = adata.X.copy()
            else:
                sum_X += adata.X
            num_matrices += 1

    # Calculate the average X by dividing the sum by the number of matrices
    average_X = sum_X / num_matrices

    # Create a new Anndata object with the averaged X
    average_adata = ad.AnnData(X=average_X)

    # Copy other attributes from the first input Anndata object
    if anndata_list[0].obs is not None:
        average_adata.obs = anndata_list[0].obs.copy()
    if anndata_list[0].var is not None:
        average_adata.var = anndata_list[0].var.copy()
    if anndata_list[0].uns is not None:
        average_adata.uns = anndata_list[0].uns.copy()

    # Calculate the average values for obsm
    if any(adata.obsm is not None for adata in anndata_list):
        average_adata.obsm = {}
        for key in anndata_list[0].obsm.keys():
            obsm_values = [adata.obsm[key] for adata in anndata_list if key in adata.obsm]
            average_adata.obsm[key] = sum(obsm_values) / num_matrices

    # Calculate the average values for layers
    if any(adata.layers is not None for adata in anndata_list):
        average_adata.layers = {}
        for key in anndata_list[0].layers.keys():
            layers_values = [adata.layers[key] for adata in anndata_list if key in adata.layers]
            average_adata.layers[key] = sum(layers_values) / num_matrices

    return average_adata


def process_for_tf_analysis(adata_6, LineageVAE_exp, adata_var_names, tf_target, trans, sample_num=100):
    tf_expression_df_list = []
    tg_matrix_list = []
    inf_adata_list = []

    for i in range(sample_num): 
        inf_adata_sample, _ = latent_transition_inferens(adata_6, LineageVAE_exp, adata_var_names, dynamics=True, trans=trans)  # input
        regulation_matrix, tf_expression_df, tg_matrix = process_tf_data(tf_target, inf_adata_sample)
        tf_expression_df_list.append(tf_expression_df)
        tg_matrix_list.append(tg_matrix)
        inf_adata_list.append(inf_adata_sample)

    # Calculate the average TF expression DataFrame
    average_tf_expression_df = pd.concat(tf_expression_df_list, axis=1).groupby(level=0, axis=1).mean()

    # Calculate the average TG matrix
    average_tg_matrix = sum(tg_matrix_list) / len(tg_matrix_list)

    average_inf_adata = average_ann_data(inf_adata_list)

    return average_inf_adata, average_tf_expression_df, average_tg_matrix, regulation_matrix


def extract_tf_adata(tf_target, inf_adata):
    tf_target['TF_Symbol'] = tf_target['TF_Symbol'].apply(lambda x: x[0] + x[1:].lower())
    tf_target['Target_Symbol'] = tf_target['Target_Symbol'].apply(lambda x: x[0] + x[1:].lower())

    # Filter TFs and TGs based on the intersection with adata var_names
    TFs = list(set(inf_adata.var_names.tolist()) & set(tf_target['TF_Symbol'].tolist()))
    TGs = list(set(inf_adata.var_names.tolist()) & set(tf_target['Target_Symbol'].tolist()))

    # Subset adata to include only TFs and TGs
    tf_adata = inf_adata[:, list(set(TGs) | set(TFs))]

    # Filter cells at day 6
    day_six_cell_ids = tf_adata[tf_adata.obs['Day'] == 6].obs['CellID']
    tf_adata = tf_adata[tf_adata.obs['CellID'].isin(day_six_cell_ids)]

    # Filter days >= 0
    days = np.unique(tf_adata.obs['Day'])
    days = days[days >= 0]
    tf_adata = tf_adata[tf_adata.obs['Day'].isin(days)]

    # Filter cells with 'reconst' == 1
    tf_adata = tf_adata[tf_adata.obs['reconst'] == 1]

    return tf_adata, TFs, TGs



def calculate_tg_delta(inf_adata, TFs, TGs, tg_delta_path):
    # Subset adata to include only TFs and TGs
    delta_adata = inf_adata[:, list(set(TGs) | set(TFs))]

    # Filter cells at initial day 6 and with 'reconst' == 1
    delta_adata = delta_adata[delta_adata.obs['initial'] == 6]
    delta_adata = delta_adata[delta_adata.obs['reconst'] == 1]

    # Get unique days
    days = np.unique(delta_adata.obs['Day'])

    # Initialize an empty array to store TG deltas
    tg_delta = np.empty((0, len(np.unique(delta_adata.var_names))))

    # Calculate TG deltas for each day
    for i in range(np.max(days)):
        delta_i = delta_adata[delta_adata.obs['Day'] == i].X.toarray() - delta_adata[delta_adata.obs['Day'] == i + 1].X.toarray()
        tg_delta = np.vstack((tg_delta, delta_i))

    # Save the result to a CSV file
    np.savetxt(tg_delta_path, tg_delta, delimiter=',')

    return tg_delta


def plot_norm_of_dynamics(average_inf_adata, norm_fig_path):
    # Assuming you have the velocity data in average_inf_adata.obsm['dl']
    velocity_data = average_inf_adata.obsm['dl']

    # Get the unique cell types from average_inf_adata.obs['differentiation_destination']
    unique_cell_types = np.unique(average_inf_adata.obs['differentiation_destination'])
    unique_cell_types = unique_cell_types[unique_cell_types != 'Undifferentiated']

    # Create a dictionary to store data for each cell type
    cell_type_data_dict = {}

    # Create a single plot for all cell types with period-on-period ratios
    fig, ax = plt.subplots(figsize=(10, 6), facecolor='white', edgecolor='black')  # Set figure facecolor and edgecolor

    for cell_type in unique_cell_types:
        # Filter data for the current cell type
        cell_type_mask = average_inf_adata.obs['differentiation_destination'] == cell_type
        cell_type_data = velocity_data[cell_type_mask]

        # Get the unique time points from average_inf_adata.obs['Day']
        unique_time_points = np.unique(average_inf_adata.obs['Day'])

        # Remove the maximum value from unique_time_points
        unique_time_points = unique_time_points[unique_time_points >= 0]
        unique_time_points = unique_time_points[unique_time_points != np.max(unique_time_points)]

        # Initialize lists to store time points and period-on-period ratios
        time_points_list = []
        period_ratios_list = []

        # Sort unique_time_points in ascending order
        sorted_time_points = np.sort(unique_time_points)

        # Loop over each time point and calculate period-on-period ratios
        for i in range(len(sorted_time_points)):
            current_time_point = sorted_time_points[i]

            if i == 0:
                # For Day 0, set period_ratio to 1
                period_ratio = 1.0
            else:
                previous_time_point = sorted_time_points[i - 1]

                # Filter data for the current and previous time points
                current_time_mask = (average_inf_adata.obs['Day'] == current_time_point) & cell_type_mask
                previous_time_mask = (average_inf_adata.obs['Day'] == previous_time_point) & cell_type_mask

                current_time_data = velocity_data[current_time_mask]
                previous_time_data = velocity_data[previous_time_mask]

                # Calculate the norm for each velocity vector at each time step
                current_norms = np.linalg.norm(current_time_data, axis=1)
                previous_norms = np.linalg.norm(previous_time_data, axis=1)

                # Calculate the average norm for the current and previous time steps
                current_average_norm = np.mean(current_norms)
                previous_average_norm = np.mean(previous_norms)

                # Calculate the period-on-period ratio
                period_ratio = current_average_norm / previous_average_norm

            # Append time point and period-on-period ratio to the lists
            time_points_list.append(int(current_time_point))  # Convert to integer
            period_ratios_list.append(period_ratio)

        # Store data for the current cell type in the dictionary
        cell_type_data_dict[cell_type] = (time_points_list, period_ratios_list)

        # Plot for the current cell type
        ax.plot(time_points_list, period_ratios_list, label=f'{cell_type} Period-on-Period Ratio', marker='o', linestyle='--')

    # Add labels and legend
    ax.set_xlabel('Time Point')
    ax.set_ylabel('Period-on-Period Ratio')
    ax.set_title('Period-on-Period Ratio of Average Norm for Different Cell Types')
    ax.grid(True)

    # Set x-axis ticks to integers
    ax.set_xticks(sorted_time_points)
    # Set x-axis tick labels
    ax.set_xticklabels([f'Day {int(day)}' for day in sorted_time_points])

    # Set the background color of the subplot to white
    ax.set_facecolor('white')

    # Add legend with white background
    legend = ax.legend(loc='upper left', bbox_to_anchor=(1, 1), frameon=True, facecolor='white')
    frame = legend.get_frame()
    frame.set_edgecolor('black')  # Set the edge color of the legend frame

    # Add black border around the entire figure
    for spine in ax.spines.values():
        spine.set_edgecolor('black')

    # Save the figure as a PNG file
    plt.savefig(norm_fig_path, bbox_inches='tight')
    plt.close()

    return cell_type_data_dict

def get_dynamics_max_value_days(cell_type_data_dict):
    dynamics_max_value_days = {}

    for cell_type, (days, values) in cell_type_data_dict.items():
        dynamics_max_value_day = days[np.argmax(values)]
        dynamics_max_value_days[cell_type] = dynamics_max_value_day

    return dynamics_max_value_days


def tf_activity_regression(inf_adata, regulation_matrix, tf_expression_df, tg_matrix):

    num_days = num_days = inf_adata.obs['Day'].max() #-1
    n, m = regulation_matrix.shape
    W = np.zeros((n, m*num_days))
    beta = np.zeros((m, num_days))

    for t in range(num_days):
        for i in range(m):
            id = regulation_matrix.index.values[regulation_matrix[tg_matrix.columns[i]]>0.5]
            if id.size > 0:
                clf = linear_model.PoissonRegressor(alpha=0) # This regressor uses the ‘log’ link function.
                clf.fit(np.log(tf_expression_df[id].iloc[(t*len(inf_adata[inf_adata.obs['Day'] == 0])):((t+1)*len(inf_adata[inf_adata.obs['Day'] == 0])),:]+1), tg_matrix[tg_matrix.columns[i]].iloc[((t+1)*len(inf_adata[inf_adata.obs['Day'] == 0])):((t+2)*len(inf_adata[inf_adata.obs['Day'] == 0]))])
                f = regulation_matrix[tg_matrix.columns[i]]>=1
                W[f,i+t*m] = clf.coef_
                beta[i,t] = clf.intercept_

    return num_days, n, m, W, beta


def tfactivity_glm(cell_type, average_inf_adata, tf_target, regulation_matrix, tf_activity_fig_path):
    # Filter data for the specific cell type
    cell_type_data = average_inf_adata[average_inf_adata.obs['differentiation_destination'] == cell_type]
    cell_type_data = cell_type_data[cell_type_data.obs['reconst'] == 1]
    cell_type_data = cell_type_data[cell_type_data.obs['Day'] >= 0]

    _, ave_tf_expression_df, ave_tg_matrix = process_tf_data(tf_target, cell_type_data)
    num_days, n, m, W, beta = tf_activity_regression(cell_type_data, regulation_matrix, ave_tf_expression_df, ave_tg_matrix)

    # Initialize a list to store the sum of absolute values for each day
    sums_per_day = [0] * num_days

    # Iterate over the values and calculate the sum of absolute values for each day
    for day in range(num_days):
        start_index = day * m
        end_index = (day + 1) * m
        day_values = W[:, start_index:end_index]
        abs_sum_per_row = np.sum(np.abs(day_values), axis=1)
        sums_per_day[day] = abs_sum_per_row

    # Assuming your data is stored in the "data" variable
    data = np.array(sums_per_day).T
    scaled_data = data / data.max()

    # Define the TF names
    tf_names = regulation_matrix.index

    # Remove rows where all values are zero
    non_zero_rows = np.any(scaled_data != 0, axis=1)
    filtered_data = scaled_data[non_zero_rows]
    filtered_tf_names = np.array(tf_names)[non_zero_rows]

    # Create a heatmap with TF names on the vertical axis and days on the horizontal axis
    plt.figure(figsize=(num_days, len(filtered_tf_names) / 3))  # Adjust the figure size as needed
    heatmap = sns.heatmap(filtered_data, annot=False, fmt='.2f', cmap='viridis', cbar=True, linewidths=0.5)
    heatmap.set_yticklabels(heatmap.get_yticklabels(), rotation=0, fontsize=8)

    plt.title(f'Heatmap of the TF activity for {cell_type}')
    plt.xlabel('Time point')
    plt.ylabel('TFs')

    # Set the axis labels to display the day numbers in the middle of each square
    plt.xticks(np.arange(num_days) + 0.5, [f'Day {i}' for i in range(num_days)])

    # Adjust y-axis tick positions to display labels in the middle
    plt.yticks(np.arange(len(filtered_tf_names)) + 0.5, filtered_tf_names)

    for i in range(len(filtered_tf_names)):
        for j in range(num_days):
            value = filtered_data[i, j]
            text_color = 'black' if value > 0.5 else 'white'
            plt.text(j + 0.5, i + 0.5, f'{value:.2f}', ha='center', va='center', fontsize=8, color=text_color)


    # Ensure that the folderpath does not end with a slash
    plt.savefig(tf_activity_fig_path, bbox_inches='tight')
    plt.close()
    return(filtered_data, W, beta, data)


def create_heatmap_with_annotations(scaled_data, regulation_matrix, scaled_tf_activity_fig_path):
    num_days = scaled_data.shape[1]
    tf_names = regulation_matrix.index
    annotation_df = scaled_data.copy()
    percentile_10 = np.percentile(scaled_data, 10)
    percentile_90 = np.percentile(scaled_data, 90)
    percentile_95 = np.percentile(scaled_data, 95)

    top_5_percent = scaled_data > percentile_95
    scaled_data[top_5_percent] = percentile_95

    plt.figure(figsize=(num_days, len(tf_names) / 3))
    heatmap = sns.heatmap(scaled_data, cmap='viridis', cbar=True, linewidths=0.5)
    heatmap.set_yticklabels(heatmap.get_yticklabels(), rotation=0, fontsize=8)
    plt.title('Heatmap of the TF activity')
    plt.xlabel('Time point', fontsize=10)
    plt.ylabel('TFs', fontsize=10)

    plt.xticks(np.arange(num_days) + 0.5, [f'Day {i}' for i in range(num_days)], fontsize=8)
    plt.yticks(np.arange(len(tf_names)) + 0.5, tf_names, fontsize=8)

    annotation_df = pd.DataFrame(annotation_df)

    for y in range(annotation_df.shape[0]):
        for x in range(annotation_df.shape[1]):
            value = annotation_df.iloc[y, x]
            if value > percentile_90:
                text_color = 'black'
            elif value <= percentile_10:
                text_color = 'white'
            else:
                text_color = 'white'

            plt.text(x + 0.5, y + 0.5, f'{value:.2f}', color=text_color,
                     ha='center', va='center', fontsize=8)

    plt.savefig(scaled_tf_activity_fig_path, bbox_inches='tight')
    plt.close()