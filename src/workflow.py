import torch
import scvelo as scv
import scanpy as sc
import numpy as np
import umap
import anndata as ad
from lineagevae import LineageVAEExperiment
import torch.distributions as dist
import torch.nn.functional as F
from utils import define_exp, input_checks
from dataset import select_anndata, select_dataset, select_undifferentiated_anndata, anndata_for_dynamics_inference, extract_highly_variable_genes


def estimate_embedding(
        adata, use_genes=None, first_epoch=100, param_path='.LineageVAE_opt.pt',
        model_params = {
            'x_dim': 100,
            'z_dim': 10,
            'enc_z_h_dim': 50, 'dec_z_h_dim': 50,  'enc_d_h_dim': 50,
            'num_enc_z_layers': 2, 'num_dec_z_layers': 2,
            'num_enc_d_layers': 2, 't_num': 2
        },
        lr=0.00001, val_ratio=(1/12), test_ratio=(1/12),
        batch_size=20, num_workers=1, sample_num=10, undifferentiated=None, kinetics=False):
    model_params['t_num'] = adata.obs['Day'].max()
    if use_genes is None:
        use_genes = adata.var_names

    if kinetics:
        input_checks(adata, kinetics=True)  # Call input_checks with kinetics=True
    else:
        input_checks(adata, kinetics=False)  # Call input_checks with kinetics=False

    adata.var['LineageVAE_used'] = use_genes
    model_params, lr, s, u, day, test_ratio, batch_size, num_workers, validation_ratio = define_exp(
        adata,
        model_params=model_params,
        lr=lr, val_ratio=val_ratio, test_ratio=test_ratio,
        batch_size=batch_size, num_workers=num_workers)

    embed_exp = LineageVAEExperiment(model_params, lr, s, u, day, test_ratio, batch_size, num_workers, validation_ratio, undifferentiated, kinetics)
    embed_exp.model.no_d_kld = True
    embed_exp.model.no_lu = True
    print(f'Loss:{embed_exp.embed_test()}')
    print('Start first opt')
    for param in embed_exp.model.enc_d.parameters():
        param.requires_grad = False
    embed_exp.init_optimizer(0.0001)
    embed_exp.embed_train_total(first_epoch)
    print('Done first opt')
    print(f'Loss:{embed_exp.embed_test()}')
    return adata, embed_exp

def estimate_dynamics(embed_exp,
        adata, use_genes=None, second_epoch=100, param_path='.LineageVAE_opt.pt',
        model_params = {
            'x_dim': 100,
            'z_dim': 10,
            'enc_z_h_dim': 50, 'dec_z_h_dim': 50,  'enc_d_h_dim': 50,
            'num_enc_z_layers': 2, 'num_dec_z_layers': 2,
            'num_enc_d_layers': 2, 't_num': 2
        },
        lr=0.0001, val_ratio=0.05, test_ratio=0.1,
        batch_size=20, num_workers=1, sample_num=10, undifferentiated=None, kinetics=False):
    model_params['t_num'] = int(adata.obs['Day'].max())
    #
    if use_genes == None:
        use_genes = adata.var_names
    if kinetics:
        input_checks(adata, kinetics=True)
    else:
        input_checks(adata, kinetics=False)
    adata.var['LineageVAE_used'] = use_genes
    model_params, lr, s, u, day, test_ratio, batch_size, num_workers, validation_ratio = define_exp(
        adata,
        model_params= model_params,
        lr=lr, val_ratio=val_ratio, test_ratio=test_ratio,
        batch_size=batch_size, num_workers=num_workers)
    #LineageVAE_exp.edm = LineageVAEDataManager(s, u, day, test_ratio, batch_size, num_workers, validation_ratio, undifferentiated, kinetics)
    LineageVAE_exp = LineageVAEExperiment(model_params, lr, s, u, day, test_ratio, batch_size, num_workers, validation_ratio, undifferentiated, kinetics)
    LineageVAE_exp.model.enc_z = embed_exp.model.enc_z
    LineageVAE_exp.model.dec_z = embed_exp.model.dec_z
    #
    print('Start second opt')
    for param in LineageVAE_exp.model.enc_z.parameters():
        param.requires_grad = False
    for param in LineageVAE_exp.model.dec_z.parameters():
        param.requires_grad = False
    for param in LineageVAE_exp.model.enc_d.parameters():
        param.requires_grad = True
    LineageVAE_exp.train_total(second_epoch)
    print('Done second opt')
    print(f'Loss:{LineageVAE_exp.test()}')
    torch.save(LineageVAE_exp.model.state_dict(), param_path)
    adata.uns['param_path'] = param_path
    return(adata, LineageVAE_exp)


def run_analysis_for_hematopoiesis(adata_input, raw_adata_input, select_adata_input=None, var_list_input=None, undifferentiated=None, n_top_genes=1000, first_epoch=100, second_epoch=100, batch_size=20, error_count_limit=3, error_count_ii_limit=1, kinetics=True):
    while True:
        # Check if select_adata_input is provided
        adata = adata_input
        if select_adata_input is not None:
            select_adata = select_adata_input.copy()
            raw_select_adata = select_adata_input.copy()
        # Calculate select_adata based on some logic involving adata_input, raw_adata_input, and var_list
        else:
            select_adata = select_anndata(adata)
            raw_select_adata = select_adata.copy()
            scv.pp.filter_and_normalize(select_adata, min_shared_counts=20, n_top_genes=n_top_genes)
            scv.pp.moments(select_adata, n_pcs=30, n_neighbors=30)
        if var_list_input is not None:
            var_list = var_list_input
        else:
            var_list = select_adata.var_names.tolist()

        error_count = 0  # Initialize an error count
        while True:

            try:
                adata = adata_input
                raw_adata = raw_adata_input
                adata = adata[:, var_list]
                adata.layers['spliced'] = raw_adata[adata.obs_names, adata.var_names].layers['spliced']
                adata.layers['unspliced'] = raw_adata[adata.obs_names, adata.var_names].layers['unspliced']
                embed_adata, embed_exp = estimate_embedding(adata=adata, first_epoch=first_epoch, batch_size=batch_size, kinetics=True)
                break  # If successful, exit the loop
            except ValueError as e:
                # Handle the error if necessary, or simply retry
                print(f"An error occurred in #i): {e}")
                # Optionally, you can add a delay before retrying to avoid continuous retries
                error_count += 1
                print(f"error count): {error_count}")
                if error_count >= error_count_limit:
                    break
                import time
                time.sleep(3)  # Sleep for 3 seconds before retrying

        if error_count >= error_count_limit:
            continue

        else:
            # Initialize the error count for #ii)
            error_count_ii = 0 # Reset the error count

            while True:
                try:
                    select_adata = raw_select_adata
                    raw_adata = raw_adata_input
                    select_adata = select_adata[:, var_list]
                    select_adata = select_adata[:, var_list].copy()
                    select_adata.layers['spliced'] = raw_adata[select_adata.obs_names, var_list].layers['spliced']
                    select_adata.layers['unspliced'] = raw_adata[select_adata.obs_names, var_list].layers['unspliced']
                    select_adata.X = raw_adata[select_adata.obs_names, var_list].X
                    adata, LineageVAE_exp = estimate_dynamics(embed_exp=embed_exp, adata=select_adata, second_epoch=second_epoch, batch_size=batch_size, undifferentiated=2, kinetics=True)
                    break  # If successful, exit the loop
                except ValueError as e:
                    # Increment the error count for #ii)
                    error_count_ii += 1
                    if error_count_ii >= error_count_ii_limit:
                        # error_count_ii = 0  # Reset the error count
                        print("Restarting from #i) due to repeated errors in #ii)")
                        break  # Break from the inner loop to restart from #i)

                    # Handle the error if necessary, or simply retry
                    print(f"An error occurred in #ii): {e}")
                    # Optionally, you can add a delay before retrying to avoid continuous retries
                    import time
                    time.sleep(3)  # Sleep for 3 seconds before retrying

                # If both #i) and #ii) have succeeded, exit the outer loop
            if error_count_ii < error_count_ii_limit:
                break
            else:
                continue
    return adata, select_adata, LineageVAE_exp, var_list



def run_analysis_for_reprogramming(adata_input, raw_adata_input, select_adata_input=None, var_list_input=None, undifferentiated=None, n_top_genes=1000, first_epoch=100, second_epoch=100, batch_size=30, error_count_limit=3, error_count_ii_limit=1, kinetics=True):
    while True:
        # Check if select_adata_input is provided
        adata = adata_input
        if select_adata_input is not None:
            select_adata = select_adata_input.copy()
            raw_select_adata = select_adata_input.copy()
        # Calculate select_adata based on some logic involving adata_input, raw_adata_input, and var_list
        else:
            selected_adata = select_dataset(adata, batch_size=30)
            undifferentiated_sample_list = select_undifferentiated_anndata(adata, selected_adata, undifferentiated_cell_size=10, initial=True)
            select_adata = anndata_for_dynamics_inference(selected_adata, undifferentiated_sample_list)
            raw_select_adata = select_adata.copy()
            var_list = extract_highly_variable_genes(adata)
        if var_list_input is not None:
            var_list = var_list_input
        else:
            var_list = select_adata.var_names.tolist()

        error_count = 0  # Initialize an error count

        while True:

            try:
                adata = adata_input
                raw_adata = raw_adata_input
                adata = adata[:, var_list]
                embed_adata, embed_exp = estimate_embedding(adata, first_epoch=first_epoch, batch_size=batch_size, kinetic=False)
                break  # If successful, exit the loop
            except ValueError as e:
                # Handle the error if necessary, or simply retry
                print(f"An error occurred in #i): {e}")
                # Optionally, you can add a delay before retrying to avoid continuous retries
                error_count += 1
                print(f"error count): {error_count}")
                if error_count >= error_count_limit:
                    break
                import time
                time.sleep(3)  # Sleep for 3 seconds before retrying

        if error_count >= error_count_limit:
            continue

        else:
            # Initialize the error count for #ii)
            error_count_ii = 0 # Reset the error count

            while True:
                try:
                    select_adata = raw_select_adata
                    raw_adata = raw_adata_input
                    select_adata = select_adata[:, var_list]
                    select_adata = select_adata[:, var_list].copy()
                    #select_adata.X = raw_adata[select_adata.obs_names, var_list].X
                    adata, LineageVAE_exp = estimate_dynamics(embed_exp=embed_exp, adata=select_adata, second_epoch=second_epoch, batch_size=batch_size, undifferentiated=undifferentiated, kinetics=False)
                    break  # If successful, exit the loop
                except ValueError as e:
                    # Increment the error count for #ii)
                    error_count_ii += 1
                    if error_count_ii >= error_count_ii_limit:
                        # error_count_ii = 0  # Reset the error count
                        print("Restarting from #i) due to repeated errors in #ii)")
                        break  # Break from the inner loop to restart from #i)

                    # Handle the error if necessary, or simply retry
                    print(f"An error occurred in #ii): {e}")
                    # Optionally, you can add a delay before retrying to avoid continuous retries
                    import time
                    time.sleep(3)  # Sleep for 3 seconds before retrying

                # If both #i) and #ii) have succeeded, exit the outer loop
            if error_count_ii < error_count_ii_limit:
                break
            else:
                continue
    return adata, select_adata, LineageVAE_exp, var_list


def check_results(LineageVAE_exp, loss_threshold=1.0e+7):
    if float(LineageVAE_exp.test_loss_list[-1]) > loss_threshold:
        print(f'loss is greater than the threshold value {loss_threshold}')
        return True
    else:
        print(f'loss is less than the threshold value {loss_threshold}')
        # Check if variance_prog is smaller than variance_obs in all dimensions
        variance_obs = np.var(LineageVAE_exp.test_z_list[0][0].detach().cpu().numpy(), axis=0)
        variance_prog = np.var(LineageVAE_exp.test_z_list[0][LineageVAE_exp.test_z_list[0].shape[0] - 1].detach().cpu().numpy(), axis=0)
        is_smaller = np.all(variance_prog < variance_obs)
        if is_smaller:
            print("variance_prog is smaller than variance_obs in all dimensions.")
            return False
        else:
            print("variance_prog is not smaller than variance_obs in all dimensions.")
            return True


def latent_visualization(adata, LineageVAE_exp, LineageVAE_used, sigma=0.05, n_neighbors=30, min_dist=0.1, dz_var_prop=0.05, sample_num=10, dynamics=False, trans=None):
    LineageVAE_exp.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    LineageVAE_exp.model = LineageVAE_exp.model.to(LineageVAE_exp.device)

    select_adata = adata[:, LineageVAE_used]
    if 'spliced' in select_adata.layers:
        if type(select_adata.layers['spliced']) == np.ndarray:
            s = torch.tensor(select_adata.layers['spliced']).astype('float64').to(LineageVAE_exp.device)
        else:
            s = torch.tensor(select_adata.layers['spliced'].astype('float64').toarray()).to(LineageVAE_exp.device)
    else:
        if type(select_adata.layers['raw_count']) == np.ndarray:
            s = torch.tensor(select_adata.layers['raw_count']).astype('float64').to(LineageVAE_exp.device)
        else:
            s = torch.tensor(select_adata.layers['raw_count'].astype('float64').toarray()).to(LineageVAE_exp.device)
    # unspliced
    if 'unspliced' in select_adata.layers:
        if type(select_adata.layers['spliced']) == np.ndarray:
            u = torch.tensor(select_adata.layers['unspliced']).astype('float64').to(LineageVAE_exp.device)
        else:
            u = torch.tensor(select_adata.layers['unspliced'].astype('float64').toarray()).to(LineageVAE_exp.device)
    else:
        # Create tensor filled with zeros of the same shape as s
        u = torch.zeros_like(s)

    # day
    day = torch.tensor(np.array(select_adata.obs['Day']).astype('float64')).to(LineageVAE_exp.device)
    batch = torch.ones(s.shape[0]).to(LineageVAE_exp.device)
    batch = batch*day.float()
    s = s.float()
    norm_mat = torch.sum(s, dim=1).view(-1, 1) * torch.sum(s, dim=0).view(1, -1)
    norm_mat = torch.mean(s) * norm_mat / torch.mean(norm_mat)
    qz_mu_T0, qz_logvar_T0 = LineageVAE_exp.model.enc_z(s, batch)
    qz_T0 = dist.Normal(qz_mu_T0, LineageVAE_exp.model.softplus(qz_logvar_T0))
    z_T0 = qz_T0.rsample()
    qz_mu = qz_mu_T0
    qz_logvar = qz_logvar_T0
    z = z_T0
    zl = qz_mu_T0
    inf_adata = select_adata.copy()
    inf_adata.obs['initial'] = select_adata.obs['Day']
    inf_adata.obs['Celltype'] = select_adata.obs['Celltype']
    inf_adata.obs['differentiation_destination'] = select_adata.obs['Celltype']
    inf_adata.obs['reconst'] = 0
    inf_adata.obsm['z'] = z.to('cpu').detach().numpy().copy()
    inf_adata.obsm['zl'] = zl.to('cpu').detach().numpy().copy()
    n_obs, n_vars = adata.shape
    inf_adata.obsm['d'] = np.empty((n_obs, n_vars))
    inf_adata.obsm['dl'] = np.empty((n_obs, n_vars))
    inf_adata.obsm['ds'] = np.empty((n_obs, n_vars))
    if trans is None:
        trans = umap.UMAP(n_neighbors=5, random_state=42).fit(inf_adata.obsm['z'])
    test_embedding = trans.transform(inf_adata.obsm['z'])
    inf_adata.obsm['umap'] = test_embedding

    #reconst
    zs = torch.ones(z.shape[0]).to(LineageVAE_exp.device)
    zs = zs * day.to(LineageVAE_exp.device)
    ld = LineageVAE_exp.model.dec_z(z, zs)
    ld = dist.Poisson(ld.to(LineageVAE_exp.device) * norm_mat.to(LineageVAE_exp.device)).sample()
    ld = ld.to('cpu').detach().numpy().copy()
    inf_adata_t = sc.AnnData(ld, select_adata.obs, select_adata.var)
    #inf_adata_t.obs['Day'] = adata.obs['Day']
    inf_adata_t.obs = select_adata.obs.copy()
    inf_adata_t.var = select_adata.var.copy()
    inf_adata_t.uns = select_adata.uns.copy()
    inf_adata_t.obsm = select_adata.obsm.copy()
    inf_adata_t.varm = select_adata.varm.copy()
    inf_adata_t.layers = select_adata.layers.copy()
    inf_adata_t.obsp = select_adata.obsp.copy()
    inf_adata_t.obs['initial'] = select_adata.obs['Day']
    inf_adata_t.obs['differentiation_destination'] = select_adata.obs['Celltype']
    inf_adata_t.obs['reconst'] = 1
    inf_adata_t.obsm['z'] = z.to('cpu').detach().numpy().copy()
    inf_adata_t.obsm['zl'] = zl.to('cpu').detach().numpy().copy()
    inf_adata_t.obsm['d'] = np.empty((n_obs, n_vars))
    inf_adata_t.obsm['dl'] = np.empty((n_obs, n_vars))
    inf_adata_t.obsm['ds'] = np.empty((n_obs, n_vars))
    test_embedding = trans.transform(inf_adata_t.obsm['z'])
    inf_adata_t.obsm['umap'] = test_embedding
    del inf_adata_t.obs['Celltype']
    inf_adata = inf_adata.concatenate(inf_adata_t)

    if dynamics:
        t_num = int(adata.obs['Day'].max())
        xs = torch.ones(s.shape[0]).to(LineageVAE_exp.device)
        for t in range(t_num + 1):
            qd_loc, qd_scale = LineageVAE_exp.model.enc_d(z)

            qd_scale = qd_scale.to(LineageVAE_exp.device)
            qd_scale = F.softplus(qd_scale)
            qd_scale = qd_scale + 1e-5
            qd = dist.Normal(qd_loc, qd_scale)

            d_coeff = LineageVAE_exp.model.d_coeff.to(LineageVAE_exp.device)
            d = d_coeff * qd.rsample()
            dl = d_coeff * qd_loc
            ds = d_coeff * qd_scale

            z = z - d
            zl = zl - qd_loc

            zs = torch.ones(z.shape[0]).to(LineageVAE_exp.device)
            zs = (zs - t)*day.to(LineageVAE_exp.device)
            ld = LineageVAE_exp.model.dec_z(z, zs)
            ld = dist.Poisson(ld.to(LineageVAE_exp.device) * norm_mat.to(LineageVAE_exp.device)).sample()
            ld = ld.to('cpu').detach().numpy().copy()
            inf_adata_t = sc.AnnData(ld, select_adata.obs, select_adata.var)
            inf_adata_t.obs = select_adata.obs.copy()
            inf_adata_t.var = select_adata.var.copy()
            inf_adata_t.uns = select_adata.uns.copy()
            inf_adata_t.obsm = select_adata.obsm.copy()
            inf_adata_t.varm = select_adata.varm.copy()
            inf_adata_t.layers = select_adata.layers.copy()
            inf_adata_t.obsp = select_adata.obsp.copy()
            inf_adata_t.obs['Day'] = select_adata.obs['Day'] - (t+1)
            inf_adata_t.obs['initial'] = select_adata.obs['Day']
            inf_adata_t.obs['differentiation_destination'] = select_adata.obs['Celltype']
            inf_adata_t.obs['reconst'] = 1
            inf_adata_t.obsm['z'] = z.to('cpu').detach().numpy().copy()
            inf_adata_t.obsm['zl'] = zl.to('cpu').detach().numpy().copy()
            inf_adata_t.obsm['d'] = d.to('cpu').detach().numpy().copy()
            inf_adata_t.obsm['dl'] = dl.to('cpu').detach().numpy().copy()
            inf_adata_t.obsm['ds'] = ds.to('cpu').detach().numpy().copy()
            test_embedding = trans.transform(inf_adata_t.obsm['z'])
            inf_adata_t.obsm['umap'] = test_embedding
            del inf_adata_t.obs['Celltype']
            inf_adata = inf_adata.concatenate(inf_adata_t)

    return(inf_adata, trans)


def latent_transition_inferens(adata_6, LineageVAE_exp, LineageVAE_used, dynamics=True, trans=None):
    inf_adata, trans = latent_visualization(adata_6, LineageVAE_exp, LineageVAE_used, dynamics=dynamics, trans=trans) #input

    # Store the cell name that is the start point of the inference in ['CellID']
    a = len(inf_adata[inf_adata.obs['Day'] == -1])
    b = int(len(inf_adata.obs) / a)
    inf_adata.obs['CellID'] = None
    for i in range(b):
        start_idx = i * a
        end_idx = (i + 1) * a
        inf_adata.obs['CellID'][start_idx:end_idx] = inf_adata[inf_adata.obs['Day'] == -1].obs.index

    return inf_adata, trans


def main_differentiation_destination(adata, dif_ratio, timepoint=None):
    lineage_list = adata.obs['Lineage'].unique()
    for lineage in lineage_list:
        main_differentiation_destination = 'Undifferentiated'
        a_subset = adata[(adata.obs['Lineage'] == lineage)]
        a_day6 = a_subset[a_subset.obs['Day'] == 6]

        # Count the number of cells that are not 'Undifferentiated' at Day 6
        a_day6_not_undiff = a_day6[a_day6.obs['Celltype'] != 'Undifferentiated']
        a = a_day6_not_undiff.shape[0]

        if a != 0:
            celltype_list = adata.obs['Celltype'].unique()
            celltype_list = [x for x in celltype_list if x != 'Undifferentiated']
            for celltype in celltype_list:
                b = a_day6[a_day6.obs['Celltype'] == celltype].shape[0]
                if b/a > dif_ratio:
                    main_differentiation_destination = celltype

            if timepoint is not None:
                adata.obs.loc[(adata.obs['Lineage'] == lineage) & (adata.obs['inference'] == timepoint), 'main_differentiation_destination'] = main_differentiation_destination
            else:
                adata.obs.loc[adata.obs['Lineage'] == lineage, 'main_differentiation_destination'] = main_differentiation_destination

    return adata