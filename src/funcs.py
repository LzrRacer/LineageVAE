import torch.distributions as dist
import random
import torch
import torch.nn.functional as F

def calc_kld(qz):
    kld = -0.5 * (1 + qz.scale.pow(2).log() - qz.loc.pow(2) - qz.scale.pow(2))
    return(kld)


def calc_poisson_loss(ld, norm_mat, obs):
    p_z = dist.Poisson(ld * norm_mat)
    l = - p_z.log_prob(obs)
    return(l)


def calc_nb_loss(ld, norm_mat, theta, obs):
    ld = norm_mat * ld
    p =  ld / (ld + theta)
    p_z = dist.NegativeBinomial(theta, p)
    l = - p_z.log_prob(obs)
    return(l)


def calc_normal_loss(z0, mu, sigma):
    p_z0 = dist.Normal(mu, sigma)
    l = - p_z0.log_prob(z0)
    return(l)


def calc_lptr(z_list, sigma, day):
    tr_num = z_list.shape[0] - 2
    lptr_list = []
    for i in range (tr_num):
        lptr = calc_normal_loss(z_list[i], z_list[i+1], sigma)
        lptr_list.append(lptr)
    cell = 0
    lptr = 0
    for cell, t in enumerate(day):
        t = int(t) - 1
        for i in range(t):
            lptr_t = lptr_list[i][cell]
            lptr += torch.sum(lptr_t, dim=-1)
    return(lptr)


def take_t(t_list, day_maxs, t):
    take_list = []
    for i, day_max in enumerate(day_maxs):
        day_max = day_max
        t_idx = int(day_max - t)
        a = t_list[t_idx][i]
        take_list.append(a)
    take_list = torch.stack(take_list)
    return(take_list)


def qz_series(qz_mu_list, qz_logvar_list, day, initial):
    cell = 0
    qz_series_list = []
    for t in day:
        t = int(t) - initial
        for i in range(t):
            qz = dist.Normal(qz_mu_list[i][cell], F.softplus(qz_logvar_list[i][cell]))
            qz_series_list.append(qz)
        cell += 1
    return(qz_series_list)


def stratified_sampling(day):
    stratified_list = []
    day_list = []
    for a in torch.unique(day).to(torch.int32).tolist():
        a1 = random.choice([i for i, x in enumerate(day) if x == a])
        day_list.append(a1)
    return(day_list)


def upsampling_z(qz_mu_T0, qz_logvar_T0, day, timepoint, upsample_num, z_dim):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    upsampled_day = torch.concat((day, timepoint*torch.ones(upsample_num).to(device)),0).to(device)
    pseudo_cell_num = qz_mu_T0[day == timepoint].shape[0]
    if pseudo_cell_num == 0:
        qz2_mu_T0 = torch.zeros(upsample_num, z_dim).to(device)
        qz2_logvar_T0 = torch.zeros(upsample_num, z_dim).to(device)
        z2_T0 = torch.zeros(upsample_num, z_dim).to(device)
    else:
        param_list = []
        for i in range(upsample_num):
            param_list.append(random.randint(0, pseudo_cell_num-1))
        qz2_mu_T0_list = []
        qz2_logvar_T0_list = []
        for k in param_list:
            qz2_mu_T0_list.append(qz_mu_T0[day == timepoint][k])
            qz2_logvar_T0_list.append(qz_logvar_T0[day == timepoint][k])
        qz2_mu_T0 = torch.stack(qz2_mu_T0_list, dim=0).to(device)
        qz2_logvar_T0 = torch.stack(qz2_logvar_T0_list, dim=0).to(device)
        qz2_T0 = dist.Normal(qz2_mu_T0, qz2_logvar_T0)
        z2_T0 = qz2_T0.rsample()
        z2_T0 = z2_T0.to(device)
    return(qz2_mu_T0, qz2_logvar_T0, z2_T0)

