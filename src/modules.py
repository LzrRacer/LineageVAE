import pandas as pd
import torch
import torch.distributions as dist
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch import functional as F
from torch.distributions.kl import kl_divergence
from torch.nn import init
import numpy as np
import random
from funcs import calc_poisson_loss, upsampling_z, take_t, stratified_sampling, calc_normal_loss, calc_lptr, qz_series


class LinearReLU(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearReLU, self).__init__()
        self.f = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim, elementwise_affine=False),
            nn.ReLU(True))

    def forward(self, x):
        h = self.f(x)
        return(h)


class SeqNN(nn.Module):
    def __init__(self, num_steps, dim):
        super(SeqNN, self).__init__()
        modules = [
            LinearReLU(dim, dim)
            for _ in range(num_steps)
        ]
        self.f = nn.Sequential(*modules)

    def forward(self, pre_h):
        post_h = self.f(pre_h)
        return(post_h)


class Encoder(nn.Module):
    def __init__(self, num_h_layers, x_dim, h_dim, z_dim):
        super(Encoder, self).__init__()
        self.x2h = LinearReLU(x_dim, h_dim)
        self.seq_nn = SeqNN(num_h_layers - 1, h_dim)
        self.h2mu = nn.Linear(h_dim, z_dim)
        self.h2logvar = nn.Linear(h_dim, z_dim)

    def forward(self, x):
        pre_h = self.x2h(x)
        post_h = self.seq_nn(pre_h)
        mu = self.h2mu(post_h)
        logvar = self.h2logvar(post_h)
        return(mu, logvar)


class Decoder(nn.Module):
    def __init__(self, num_h_layers, z_dim, h_dim, x_dim):
        super(Decoder, self).__init__()
        self.z2h = LinearReLU(z_dim, h_dim)
        self.seq_nn = SeqNN(num_h_layers - 1, h_dim)
        self.h2ld = nn.Linear(h_dim, x_dim)
        self.softplus = nn.Softplus()

    def forward(self, z):
        pre_h = self.z2h(z)
        post_h = self.seq_nn(pre_h)
        ld = self.h2ld(post_h)
        correct_ld = self.softplus(ld)
        return(correct_ld)

#
class VectorField(nn.Module):
    def __init__(self, num_h_layers, z_dim, h_dim):
        super(VectorField, self).__init__()
        self.x2h = LinearReLU(z_dim, h_dim)
        self.seq_nn = SeqNN(num_h_layers - 1, h_dim)
        self.h2v = nn.Linear(h_dim, z_dim)

    def forward(self, x):
        pre_h = self.x2h(x)
        post_h = self.seq_nn(pre_h)
        v = self.h2v(post_h)
        return(v)

#
class StochasticVectorField(nn.Module):
    def __init__(self, num_h_layers, z_dim, h_dim):
        super(StochasticVectorField, self).__init__()
        self.z2sigmu = Encoder(num_h_layers, z_dim, h_dim, z_dim)
        self.softplus = nn.Softplus()

    def forward(self, z):
        qd_mu, qd_logvar  = self.z2sigmu(z)
        qd = dist.Normal(qd_mu, self.softplus(qd_logvar))
        d = qd.rsample()
        return(d, qd.loc, qd.scale)
#
def apply_with_batch(f, x, s):
    xs = torch.cat([x, s], dim=-1)
    out = f(xs)
    return out

class EncoderBatch(nn.Module):
    def __init__(self, num_enc_z_h_layers, x_dim, enc_z_h_dim, z_dim, use_batch_bias=False):
        super(EncoderBatch, self).__init__()
        if use_batch_bias:
            self.encode_z = Encoder(num_enc_z_h_layers, x_dim, enc_z_h_dim, z_dim)
            self.main = self.encode_no_batch_z
        else:
            self.encode_z = Encoder(num_enc_z_h_layers, x_dim, enc_z_h_dim, z_dim)
            self.main = self.encode_batch_z_concat

    def encode_batch_z_concat(self, x, s):
        mu, logvar = apply_with_batch(self.encode_z, x, s)
        return mu, logvar

    def encode_no_batch_z(self, x, s):
        mu, logvar = self.encode_z(x)
        return mu, logvar

    def forward(self, x, s):
        mu, logvar = self.main(x, s)
        return(mu, logvar)

class DecoderBatch(nn.Module):
    def __init__(self, num_h_layers, z_dim, h_dim, x_dim, use_batch_bias=False):
        super(DecoderBatch, self).__init__()
        if use_batch_bias:
            self.decode_z = Decoder(num_h_layers, z_dim, h_dim, x_dim)
            self.main = self.decode_no_batch_z
        else:
            self.decode_z = Decoder(num_h_layers, z_dim, h_dim, x_dim)
            self.main = self.decode_batch_z_concat

    def decode_no_batch_z(self, z, s):
        px_z_ld = self.decode_z(z)
        return px_z_ld

    def decode_batch_z_concat(self, z, s):
        px_z_ld = apply_with_batch(self.decode_z, z, s)
        return px_z_ld

    def forward(self, x, s):
        px_z_ld = self.main(x, s)
        return(px_z_ld)


class LineageVAE(nn.Module):
    def __init__(
            self,
            x_dim, z_dim,
            enc_z_h_dim, dec_z_h_dim, enc_d_h_dim,
            num_enc_z_layers, num_dec_z_layers, num_enc_d_layers,
            t_num, d_mode='stochastic', undifferentiated=None, norm_input=False, kinetics=False):
        super(LineageVAE, self).__init__()
        #self.enc_z = Encoder(num_enc_z_layers, x_dim, enc_z_h_dim, z_dim)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.enc_z = EncoderBatch(num_enc_z_layers, x_dim, enc_z_h_dim, z_dim, use_batch_bias=True)
        self.enc_d = Encoder(num_enc_d_layers, z_dim, enc_d_h_dim, z_dim)
        #self.dec_z = Decoder(num_enc_z_layers, z_dim, dec_z_h_dim, x_dim)
        self.dec_z = DecoderBatch(num_enc_z_layers, z_dim, dec_z_h_dim, x_dim, use_batch_bias=True)
        self.enc_s = Encoder(num_enc_z_layers, z_dim, enc_z_h_dim, 1)

        self.upsample_num = 20
        self.dt = 1
        self.gamma_mean = 0.05
        self.d_coeff = Parameter(torch.Tensor(1))
        self.loggamma = Parameter(torch.Tensor(x_dim))
        self.logbeta = Parameter(torch.Tensor(x_dim))
        self.tr_scale = Parameter(torch.Tensor(1))
        self.z0_scale = Parameter(torch.Tensor(1))
        self.softplus = nn.Softplus()
        self.reset_parameters()
        self.no_lu = False
        self.no_d_kld = False
        self.no_z_kld = False
        self.norm_input = norm_input
        self.z_dim = z_dim
        self.t_num = t_num
        self.undifferentiated = undifferentiated
        self.kinetics= False

    def reset_parameters(self):
        init.normal_(self.loggamma)
        init.normal_(self.logbeta)
        init.normal_(self.tr_scale)
        init.normal_(self.z0_scale)

    def forward(self, x, day):
        # encode z
        x = x.to(self.device)
        day = day.to(self.device)
        batch = torch.ones(x.shape[0]).to(self.device)
        batch = batch*day.to(self.device)
        qz_mu_T0, qz_logvar_T0 = self.enc_z(x, batch)
        qz_mu_T0 = qz_mu_T0.to(self.device)
        qz_logvar_T0 = qz_logvar_T0.to(self.device)
        qz_logvar_T0 = self.softplus(qz_logvar_T0)
        qz_logvar_T0 += 1e-5
        qz_T0 = dist.Normal(qz_mu_T0, qz_logvar_T0)
        z_T0 = qz_T0.rsample().to(self.device)
        if self.undifferentiated is None:
            qz_mu = qz_mu_T0.to(self.device)
            qz_logvar = qz_logvar_T0.to(self.device)
            z = z_T0.to(self.device)
        else:
            qz_mu = qz_mu_T0.to(self.device)
            qz_logvar = qz_logvar_T0.to(self.device)
            z = z_T0.to(self.device)
            qz2_mu_T0, qz2_logvar_T0, z2_T0 = upsampling_z(qz_mu_T0, qz_logvar_T0, day, self.undifferentiated, self.upsample_num, self.z_dim)
            qz2_mu_T0 = qz2_mu_T0.detach().to(self.device)
            qz2_logvar_T0 = qz2_logvar_T0.detach().to(self.device)
            z2_T0 = z2_T0.detach().to(self.device)
            z = torch.cat((z, z2_T0), 0).to(self.device)
        # decode z
        batch = torch.ones(z_T0.shape[0]).to(self.device)
        batch = batch*day.to(self.device)
        ld = self.dec_z(z_T0, batch)
        if self.undifferentiated is None:
            # generate z series
            qz_mu_list = [qz_mu]
            z_list = [z]
            z2_list = []
            qz_logvar_list = [qz_logvar]
        else:
            #concatenate upsampled z
            qz_mu = torch.cat((qz_mu, qz2_mu_T0))
            qz_logvar = torch.cat((qz_logvar, qz2_logvar_T0))
            z_list = [z]
            qz_mu_list = [qz_mu]
            z2_list = [qz2_mu_T0]
            qz_logvar_list = [qz_logvar]
        qd_loc_list = []
        qd_scale_list = []

        for t in range(self.t_num + 1):
            qd_loc, qd_scale = self.enc_d(z)
            qd_scale = qd_scale.to(self.device)
            qd_scale = self.softplus(qd_scale)
            qd_scale += 1e-5
            qd = dist.Normal(qd_loc, qd_scale)
            d_coeff = self.d_coeff.to(self.device)
            d = d_coeff * qd.rsample()
            qz_mu = qz_mu - qd_loc
            z = z - d
            batch = day.cpu()
            qz_mu_list.append(qz_mu)
            qz_logvar_list.append(qz_logvar) #????
            z_list.append(z)
            qd_loc_list.append(qd_loc)
            qd_scale_list.append(self.softplus(qd_scale))
        qz_mu_list = torch.stack(qz_mu_list, dim=0)
        qz_logvar_list = torch.stack(qz_logvar_list, dim=0)
        z_list = torch.stack(z_list, dim=0)
        if self.undifferentiated is None:
            z2_list = torch.zeros(z_list.shape)
        else:
            z2_list = torch.stack(z2_list, dim=0)
        z0_list = take_t(z_list, day, 0)
        qd_loc_list = torch.stack(qd_loc_list, dim=0)
        qd_scale_list = torch.stack(qd_scale_list, dim=0)
        #velocity
        batch = torch.ones(z_T0.shape[0]).to(self.device)
        batch = batch*day.to(self.device)
        velocity = ld - self.dec_z(z_T0, batch)
        gamma = self.softplus(self.loggamma)
        beta = self.softplus(self.logbeta) * self.dt
        pu_zd_ld = self.softplus(velocity + ld * gamma) / beta
        #batch = torch.ones(z_list.shape[1]-self.upsample_num).to(self.device)
        #batch = batch*day.to(self.device)
        #px_z_ld = self.dec_z(z_list[0][:-self.upsample_num], batch)
        #velocity = px_z_ld - self.dec_z(z_list[1][:-self.upsample_num], (batch - 1))
        #gamma = self.softplus(self.loggamma)
        #beta = self.softplus(self.logbeta) * self.dt
        #pu_zd_ld = self.softplus(velocity + px_z_ld * gamma) / beta
        return(z_T0, qz_T0, ld, pu_zd_ld, qz_mu_list, qz_logvar_list, z_list, z0_list, z2_list, qd_loc_list, qd_scale_list)

    def embed_loss(self, x, u, day, norm_mat, turn_on_d_kld=False, kinetics=False):
        if self.norm_input:
            in_x = x / norm_mat
        else:
            in_x = x
        z_T0, qz_T0, ld, pu_zd_ld, qz_mu_list, qz_logvar_list, z_list, z0_list, z2_list, qd_loc_list, qd_scale_list = self(in_x, day)
        #log p(x|zT)
        lx = calc_poisson_loss(ld, norm_mat, x)
        kld = -0.5 * (1 + qz_T0.scale.pow(2).log() - qz_T0.loc.pow(2) - qz_T0.scale.pow(2))
        embed_loss = torch.sum(lx) + torch.sum(kld)
        return(embed_loss)

    def elbo_loss(self, x, u, day, norm_mat, turn_on_d_kld=False, undifferentiated=None, kinetics=False):
        if self.norm_input:
            in_x = x / norm_mat
        else:
            in_x = x
        z_T0, qz_T0, ld, pu_zd_ld, qz_mu_list, qz_logvar_list, z_list, z0_list, z2_list, qd_loc_list, qd_scale_list = self(in_x, day)
        #log p(x|zT)
        lx = calc_poisson_loss(ld, norm_mat, x)
        #log p(z0)
          #z0 = random.choice(z0_list)
          #loc = torch.zeros(z0.shape).to(device)
          #scale = torch.ones(z0.shape).to(device)
          #lpz0 = calc_normal_loss(z0, loc, scale)
        z0_list.to(self.device)
        stratified_z0_list = []
        lpz0 = 0
        if self.undifferentiated is None:
            upsampled_day = day
        else:
            upsampled_day = torch.concat((day, undifferentiated*torch.ones(self.upsample_num).to(self.device)),0)
        day_list = stratified_sampling(day)
        for i in day_list:
            stratified_z0_list.append(z0_list[i])
        for z0 in stratified_z0_list:
            if self.undifferentiated is None:
                loc = torch.zeros(z0.shape).to(self.device)
            else:
                if z2_list.shape[1] == 0:
                    #loc = take_t(z_list, day, 2).mean(axis=0)
                    loc = take_t(z_list, upsampled_day, 2).mean(axis=0)
                else:
                    loc = z2_list[0].mean(axis=0) #z2のmeanをz0のmeanに
            scale = torch.ones(z0.shape).to(self.device)
            z0_scale = self.softplus(self.z0_scale)
            scale = scale * z0_scale
            lpz0 += calc_normal_loss(z0, loc, scale)
        lpz0 = 1/len(stratified_z0_list) * lpz0
        #log p(z1|z0)
        lpz1 = 0
        #from IPython.core.debugger import Pdb; Pdb().set_trace()
        z1_list = take_t(z_list, upsampled_day, 0)
        tr_scale = self.softplus(self.tr_scale)
        #tr_scale = 1
        scale = torch.ones(z0.shape).to(self.device)
        scale = scale * tr_scale
        for z0 in stratified_z0_list:
            lpz1 += calc_normal_loss(z1_list, z0, scale)
        lpz1 = 1/len(stratified_z0_list) * lpz1
        #log p(zt|zt-1)
        lptr = calc_lptr(z_list, scale, upsampled_day)
        #log q(z0|z1)
        qd1_loc_list = take_t(qd_loc_list, upsampled_day, 1)
        qd1_scale_list = take_t(qd_scale_list, upsampled_day, 1)
        lqz0 = - calc_normal_loss(z0, z1_list - qd1_loc_list, qd1_scale_list).sum(dim=1)
        lqz0 = - torch.logsumexp(lqz0, dim=0)
        #log q(zt-1|zt)
        qz_series_list = qz_series(qz_mu_list, qz_logvar_list, day, 1)
        qz_series_entropy = 0
        for i in range(len(qz_series_list)):
            qz_series_entropy += torch.sum(qz_series_list[i].entropy(), dim=-1)
        if kinetics:
            #log q(z|x)
            lu = calc_poisson_loss(pu_zd_ld, norm_mat, u)
            elbo_loss = torch.sum(lx) + torch.sum(lu) + torch.sum(lpz0) + torch.sum(lpz1) + torch.sum(lptr) - torch.sum(qz_series_entropy) - torch.sum(lqz0)
        else:
            elbo_loss = torch.sum(lx) + torch.sum(lpz0) + torch.sum(lpz1) + torch.sum(lptr) - torch.sum(qz_series_entropy) - torch.sum(lqz0)
        return(elbo_loss, z_list)
    

    