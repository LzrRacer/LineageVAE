import torch.distributions as dist
import random


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


def stratified_sampling(day):
    stratified_list = []
    day_list = []
    for a in torch.unique(day).to(torch.int32).tolist():
        a1 = random.choice([i for i, x in enumerate(day) if x == a])
        day_list.append(a1)
    return(day_list)