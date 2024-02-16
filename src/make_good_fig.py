
import scanpy as sc
import pandas as pd
import torch
import datetime
import os
import joblib
import numpy as np
from workflow import run_analysis_for_hematopoiesis, check_results, latent_visualization, latent_transition_inferens, main_differentiation_destination
from umap_visualization import save_umap_scatterplot, save_custom_umap_scatterplot, save_celltype_umap_scatterplot, save_transition_scatter_plot
from tf_activity_analysis import process_tf_data, average_ann_data, process_for_tf_analysis, extract_tf_adata, calculate_tg_delta, plot_norm_of_dynamics, tfactivity_glm, create_heatmap_with_annotations, get_dynamics_max_value_days



restart = True

while restart:
    # Reset data
    # Set the path for the new folder
    dt_now = datetime.datetime.now().strftime('%Y%m%d%H%M')
    folder_path = datetime.datetime.now().strftime('%Y%m%d%H%M_LineageVAE')
    folder_path = os.path.join('Result',folder_path)
    os.makedirs(folder_path, exist_ok=True)

    file_path = '../Data/20230413_total_adata.h5ad'
    adata = sc.read(file_path, cache=True)
    #adata = sc.read("20230413_total_adata.h5ad", cache=True)
    adata_input = adata.copy()
    raw_adata_input = adata.copy()
    adata, select_adata, LineageVAE_exp, var_list = run_analysis_for_hematopoiesis(
        adata_input, raw_adata_input, select_adata_input=None, var_list_input=None, 
        undifferentiated=2, n_top_genes=1000, first_epoch=100, second_epoch=100, 
        batch_size=20, error_count_limit=3, error_count_ii_limit=2, kinetics=True
    )


    while check_results(LineageVAE_exp):
        adata, select_adata, LineageVAE_exp, var_list = run_analysis_for_hematopoiesis(
            adata_input, raw_adata_input, select_adata_input=None, var_list_input=None,
            undifferentiated=2, n_top_genes=1000, first_epoch=100, second_epoch=100,
            batch_size=20, error_count_limit=3, error_count_ii_limit=1, kinetics=True
        )


    # Save model parameters
    model_parameters_path = os.path.join(f'Result/{dt_now}_LineageVAE', dt_now + '_model_parameters.pth')
    torch.save(LineageVAE_exp.model.state_dict(), model_parameters_path)

    # Save var_list as CSV
    var_list_df = pd.DataFrame({'var_names': var_list})
    var_list_csv_path = os.path.join(f'Result/{dt_now}_LineageVAE', dt_now + '_var_list.csv')
    var_list_df.to_csv(var_list_csv_path, index=False)

    # Save select_adata
    select_adata_path = os.path.join(f'Result/{dt_now}_LineageVAE', dt_now + '_select_adata.h5ad')
    select_adata.write(select_adata_path)

    # Save check_adata
    adata = sc.read(file_path, cache=True)
    adata = adata[:,var_list]
    check_adata, trans = latent_visualization(adata, LineageVAE_exp, adata.var_names) 
    check_adata_path = os.path.join(f'Result/{dt_now}_LineageVAE', dt_now + '_check_adata.h5ad')
    check_adata.write(check_adata_path)

    trans_path = os.path.join(f'Result/{dt_now}_LineageVAE', dt_now + '_trans_adata.h5ad')
    trans_path = os.path.join(folder_path, f'{dt_now}_trans.joblib')
    joblib.dump(trans, trans_path)

    # UMAP Visualization
    check_adata_umap_path = os.path.join(f'Result/{dt_now}_LineageVAE', dt_now + '_check_adata_umap_path.png')
    save_umap_scatterplot(check_adata, c_map="Spectral", output_file=check_adata_umap_path)
    check_adata_custom_umap_path = os.path.join(f'Result/{dt_now}_LineageVAE', dt_now + '_check_adata_custom_umap_path.png')
    save_custom_umap_scatterplot(check_adata, dot_size=2, output_file=check_adata_custom_umap_path)
    check_adata_celltype_umap_path = os.path.join(f'Result/{dt_now}_LineageVAE', dt_now + '_check_adata_celltype_umap_path.png')
    custom_palette = {
        'Baso': 'red',
        'Ccr7_DC': 'magenta',
        'Eos': 'green',
        'Erythroid': 'purple',
        'Lymphoid': 'cyan',
        'Mast': 'pink',
        'Meg': 'brown',
        'Monocyte': 'orange',
        'Neutrophil': 'blue',
        'Undifferentiated': 'gray',
        'pDC': 'lime'
    }
    save_celltype_umap_scatterplot(check_adata, output_file=check_adata_celltype_umap_path, custom_palette=custom_palette)


    # dynamics inference
    select_check_adata, trans = latent_visualization(select_adata, LineageVAE_exp, adata.var_names, trans=trans)

    # UMAP Visualization
    select_adata_custom_umap_path = os.path.join(f'Result/{dt_now}_LineageVAE', dt_now + '_select_adata_custom_umap_path.png')
    save_custom_umap_scatterplot(select_check_adata, dot_size=20, output_file=select_adata_custom_umap_path)
    select_check_adata_celltype_umap_path = os.path.join(f'Result/{dt_now}_LineageVAE', dt_now + '_check_adata_celltype_umap_path.png')
    custom_palette = {
        'Baso': 'red',
        'Ccr7_DC': 'magenta',
        'Eos': 'green',
        'Erythroid': 'purple',
        'Lymphoid': 'cyan',
        'Mast': 'pink',
        'Meg': 'brown',
        'Monocyte': 'orange',
        'Neutrophil': 'blue',
        'Undifferentiated': 'gray',
        'pDC': 'lime'
    }

    save_celltype_umap_scatterplot(select_check_adata, output_file=select_check_adata_celltype_umap_path, custom_palette=custom_palette)


    # dynamics inference for Day 6
    adata_6 = select_adata
    adata_6 = adata_6[adata_6.obs['Day']==6]
    adata_6 = adata_6[:,var_list]
    inf_adata, trans = latent_transition_inferens(adata_6, LineageVAE_exp, adata.var_names, dynamics=True, trans=trans)
    inf_adata_path = os.path.join(f'Result/{dt_now}_LineageVAE', dt_now + '_inf_adata.h5ad')
    inf_adata.write(inf_adata_path)

    inf_adata_transition_umap_path = os.path.join(f'Result/{dt_now}_LineageVAE', dt_now + '_inf_adata_transition_umap.png')
    save_transition_scatter_plot(inf_adata, c_map="Spectral", save_path=inf_adata_transition_umap_path)


    # TF Activity Analysis
    inf_adata = main_differentiation_destination(inf_adata, 0.4)
    tf_target = pd.read_csv("../Data/tf_target.txt" , sep='\t')
    regulation_matrix, tf_expression_df, tg_matrix = process_tf_data(tf_target, inf_adata)
    average_inf_adata, average_tf_expression_df, average_tg_matrix = inf_adata, tf_expression_df, tg_matrix
    #average_inf_adata, average_tf_expression_df, average_tg_matrix, regulation_matrix = process_for_tf_analysis(adata_6, LineageVAE_exp, inf_adata.var_names, tf_target, trans, sample_num=1)
    #average_inf_adata_path = os.path.join(f'Result/{dt_now}_LineageVAE', dt_now + "_average_inf_adata.h5ad")
    #average_inf_adata.write_h5ad(average_inf_adata_path)
    tf_adata, TFs, TGs = extract_tf_adata(tf_target, inf_adata)
    tg_delta_path = os.path.join(f'Result/{dt_now}_LineageVAE', dt_now + "_tg_delta.h5ad")
    tg_delta = calculate_tg_delta(inf_adata, TFs, TGs, tg_delta_path)
    norm_fig_path = os.path.join(f'Result/{dt_now}_LineageVAE', dt_now + "_norm_fig.png")
    cell_type_data_dict = plot_norm_of_dynamics(average_inf_adata, norm_fig_path)
    cell_types_to_plot = ['Baso', 'Monocyte', 'Neutrophil']
    filtered_data_list = []
    W_list = []
    for cell_type in cell_types_to_plot:
        tf_activity_fig_path = os.path.join(f'Result/{dt_now}_LineageVAE', f'{dt_now}_tf_activity_{cell_type}.png')
        filtered_data, W, beta, data = tfactivity_glm(cell_type, average_inf_adata, tf_target, regulation_matrix, tf_activity_fig_path)
        tf_activity_matrix_path = os.path.join(f'Result/{dt_now}_LineageVAE', f'{dt_now}_tf_activity_{cell_type}.csv')
        np.savetxt(tf_activity_matrix_path, filtered_data, delimiter=',')
        filtered_data_list.append(filtered_data)
        tf_activity_coef_path  = os.path.join(f'Result/{dt_now}_LineageVAE', f'{dt_now}_tf_activity_coef_{cell_type}.csv') 
        np.savetxt(tf_activity_coef_path, W, delimiter=',')
        tf_activity_intercept_path  = os.path.join(f'Result/{dt_now}_LineageVAE', f'{dt_now}_tf_activity_intercept_{cell_type}.csv')
        np.savetxt(tf_activity_intercept_path, beta, delimiter=',')
        scaled_tf_activity_fig_path = os.path.join(f'Result/{dt_now}_LineageVAE', f'{dt_now}_scaled_tf_activity_{cell_type}.png')
        create_heatmap_with_annotations(filtered_data, regulation_matrix, scaled_tf_activity_fig_path)

    dynamics_max_value_days = get_dynamics_max_value_days(cell_type_data_dict)
    dynamics_dict = dict(zip(cell_types_to_plot, dynamics_max_value_days))
    restart = False

    for cell_type in cell_types_to_plot:
        # Find the index of the cell type in the list of cell types
        cell_type_index = cell_types_to_plot.index(cell_type)
        filtered_data_i = filtered_data_list[cell_type_index]

        # Find the day with the largest value
        max_value_day = np.argmax(filtered_data_i.max(axis=0))

        # Find the day with the largest sum
        sum_per_day = np.sum(filtered_data_i, axis=0)
        max_sum_day = np.argmax(sum_per_day)
        max_sum_day_label = f'Day {max_sum_day}'
        max_dynamics_day = dynamics_max_value_days[cell_type]


        # Check conditions and restart if needed
        if (
            max_value_day > max_dynamics_day
            or max_sum_day > max_dynamics_day
            or (
                cell_type == 'Neutrophil'
                and dynamics_dict['Neutrophil'] > dynamics_dict['Baso']
            )
            or (
                cell_type == 'Monocyte'
                and dynamics_dict['Monocyte'] > dynamics_dict['Baso']
            )
            or (
                cell_type == 'Neutrophil'
                and max_sum_day > np.argmax(
                    np.sum(filtered_data_list[cell_types_to_plot.index('Baso')], axis=0)
                )
            )
            or (
                cell_type == 'Monocyte'
                and max_sum_day > np.argmax(
                    np.sum(filtered_data_list[cell_types_to_plot.index('Baso')], axis=0)
                )
            )
            or (
                cell_type == 'Neutrophil'
                and max_value_day > np.argmax(
                    filtered_data_list[cell_types_to_plot.index('Baso')].max(axis=0)
                )
            )
            or (
                cell_type == 'Monocyte'
                and max_value_day > np.argmax(
                    filtered_data_list[cell_types_to_plot.index('Baso')].max(axis=0)
                )
            )
        ):
            restart = True
            break

if not restart:
    print("All clear")