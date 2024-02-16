import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os


def save_umap_scatterplot(adata, c_map="Spectral", output_file="umap_scatterplot.png"):

    y = adata.obs['Day']
    embedding_x = adata.obsm['umap'][:, 0]
    embedding_y = adata.obsm['umap'][:, 1]

    # Convert Anndata observation metadata to a pandas DataFrame
    obs_df = pd.DataFrame(adata.obs)

    # Filter the data based on the condition
    filtered_indices = np.where(obs_df['Day'] == obs_df['initial'])[0]

    # Filter embedding_x and embedding_y using the filtered indices
    filtered_embedding_x = embedding_x[filtered_indices]
    filtered_embedding_y = embedding_y[filtered_indices]

    # Get y values for the filtered data
    filtered_y = y[filtered_indices]

    # Create a rainbow colormap using RdYlBu_r
    cmap = plt.cm.get_cmap(c_map, len(filtered_y))

    # Create the scatter plot using the filtered data and rainbow colormap
    ax = sns.scatterplot(x=filtered_embedding_x, y=filtered_embedding_y, hue=filtered_y, palette=cmap, s=5)

    # Get the unique values from filtered_y
    unique_values = sorted(filtered_y.unique())
    color_palette = sns.color_palette(c_map, len(unique_values))

    # Remove the existing legend
    ax.get_legend().remove()

    # Create a new legend with formatted labels ("Day x")
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker='o',
            color='w',
            markerfacecolor=color_palette[i],
            markersize=10,
            label=f'Day {unique_values[i]:.0f}'
        )
        for i in range(len(unique_values))
    ]

    plt.legend(handles=handles, title='Observation', loc='upper left', bbox_to_anchor=(1, 1))

    # Save the plot as a PNG file
    plt.savefig(output_file, bbox_inches='tight')
    plt.close()


def save_custom_umap_scatterplot(adata, dot_size=2, output_file="custom_umap_scatterplot.png"):

    y = adata.obs['Day']
    embedding_x = adata.obsm['umap'][:, 0]
    embedding_y = adata.obsm['umap'][:, 1]

    # Convert Anndata observation metadata to a pandas DataFrame
    obs_df = pd.DataFrame(adata.obs)

    # Define a custom color mapping dictionary
    color_mapping = {2: '#fed481', 4: '#d6ee9b', 6: '#3d95b8'}
    color_mapping_face = ['#fed481', '#d6ee9b', '#3d95b8']

    # Create a list of colors based on the custom color mapping
    point_colors = [color_mapping[val] for val in y]

    # Create the scatter plot for Day 6
    filtered_indices_6 = np.where(obs_df['Day'] == 6)[0]
    filtered_embedding_x_6 = embedding_x[filtered_indices_6]
    filtered_embedding_y_6 = embedding_y[filtered_indices_6]
    filtered_y_6 = y[filtered_indices_6]

    # Create the scatter plot for Day 4
    filtered_indices_4 = np.where(obs_df['Day'] == 4)[0]
    filtered_embedding_x_4 = embedding_x[filtered_indices_4]
    filtered_embedding_y_4 = embedding_y[filtered_indices_4]
    filtered_y_4 = y[filtered_indices_4]

    # Create the scatter plot for Day 2
    filtered_indices_2 = np.where(obs_df['Day'] == 2)[0]
    filtered_embedding_x_2 = embedding_x[filtered_indices_2]
    filtered_embedding_y_2 = embedding_y[filtered_indices_2]
    filtered_y_2 = y[filtered_indices_2]

    # Create the main scatter plot with Day 6 data
    plt.scatter(filtered_embedding_x_6, filtered_embedding_y_6, c=[color_mapping[val] for val in filtered_y_6], s=dot_size, label='Day 6')

    # Superimpose scatter plots for Day 4 and Day 2
    plt.scatter(filtered_embedding_x_4, filtered_embedding_y_4, c=[color_mapping[val] for val in filtered_y_4], s=dot_size, label='Day 4') #, alpha=0.5
    plt.scatter(filtered_embedding_x_2, filtered_embedding_y_2, c=[color_mapping[val] for val in filtered_y_2], s=dot_size, label='Day 2') #, alpha=0.5

    plt.grid(False)
    plt.gca().set_facecolor('white')

    # Add legend
    legend = plt.legend(title='Time point', loc='upper left', bbox_to_anchor=(1, 1))
    legend.set_frame_on(True)
    legend.get_frame().set_facecolor('white')

    # Save the plot as a PNG file
    plt.savefig(output_file, bbox_inches='tight')
    plt.close()



def save_celltype_umap_scatterplot(check_adata, output_file="celltype_umap_scatterplot.png", custom_palette=None):

    y = check_adata.obs['Celltype']
    embedding_x = check_adata.obsm['umap'][:, 0]
    embedding_y = check_adata.obsm['umap'][:, 1]

    if custom_palette is None:
        ax = sns.scatterplot(x=embedding_x, y=embedding_y, hue=y, s=5)
    else:
        ax = sns.scatterplot(x=embedding_x, y=embedding_y, hue=y, palette=custom_palette, s=5)

    plt.grid(False)
    plt.gca().set_facecolor('white')

    # Get the legend and set its background color
    legend = plt.legend(title='Cell type', loc='upper left', bbox_to_anchor=(1, 1))
    legend.set_frame_on(True)
    legend.get_frame().set_facecolor('white')

    # Save the plot as a PNG file
    plt.savefig(output_file, bbox_inches='tight')
    plt.close()


def save_transition_scatter_plot(inf_adata, c_map="Spectral", save_path="scatter_plot.png"):
    # Filter the data to include only cells with positive 'Day' values
    filtered_indices = inf_adata.obs['Day'] > -1
    filtered_embedding_x = inf_adata.obsm['umap'][filtered_indices, 0]
    filtered_embedding_y = inf_adata.obsm['umap'][filtered_indices, 1]
    filtered_y = inf_adata.obs['Day'][filtered_indices]

    # Create the scatter plot using the filtered data and the specified colormap
    ax = sns.scatterplot(x=filtered_embedding_x, y=filtered_embedding_y, hue=filtered_y, palette=c_map, s=20)

    # Remove the existing legend
    ax.legend_.remove()

    # Create a custom legend with formatted labels ("Day x")
    unique_values = sorted(filtered_y.unique())
    color_palette = sns.color_palette(c_map, len(unique_values))
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker='o',
            color='w',
            markerfacecolor=color_palette[i],
            markersize=10,
            label=f'Day {unique_values[i]:.0f}'
        )
        for i in range(len(unique_values))
    ]

    # Create the legend
    plt.legend(handles=handles, title='Inference', loc='upper left', bbox_to_anchor=(1, 1))

    # Save the plot
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()