import torch
from torch.utils.data import DataLoader
import numpy as np
import copy
from dataset import LineageVAEDataManager
from modules import LineageVAE


class LineageVAEExperiment:
    def __init__(self, model_params, lr, s, u, day, test_ratio, batch_size, num_workers, validation_ratio, undifferentiated=None, kinetics=False):
        self.edm = LineageVAEDataManager(s, u, day, test_ratio, batch_size, num_workers, validation_ratio)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = LineageVAE(**model_params, undifferentiated=undifferentiated, kinetics=kinetics)
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.train_loss_list = []
        self.test_loss_list = []
        self.train_z_list = []
        self.test_z_list = []
        self.best_loss = None
        self.undifferentiated=undifferentiated

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        entry_num = 0
        for s, u, day, norm_mat in self.edm.train_loader:
            s = s.to(self.device)
            u = u.to(self.device)
            day = day.to(self.device)
            norm_mat = norm_mat.to(self.device)
            self.optimizer.zero_grad()
            loss, z_list = self.model.elbo_loss(
                s, u, day, norm_mat, undifferentiated=self.undifferentiated, kinetics=False)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            entry_num += s.shape[0]
        loss_val = total_loss / entry_num
        self.train_loss_list.append(loss_val)
        self.train_z_list.append(z_list)
        return(total_loss / entry_num)

    def evaluate(self):
        self.model.eval()
        s = self.edm.validation_s.to(self.device)
        u = self.edm.validation_u.to(self.device)
        day = self.edm.validation_day.to(self.device)
        norm_mat = self.edm.validation_norm_mat.to(self.device)
        loss, z_list = self.model.elbo_loss(
            s, u, day, norm_mat, undifferentiated=self.undifferentiated, kinetics=False)
        entry_num = s.shape[0]
        loss_val = loss / entry_num
        return(loss_val)

    def test(self):
        self.model.eval()
        s = self.edm.test_s.to(self.device)
        u = self.edm.test_u.to(self.device)
        day = self.edm.test_day.to(self.device)
        norm_mat = self.edm.test_norm_mat.to(self.device)
        loss, z_list = self.model.elbo_loss(
            s, u, day, norm_mat, undifferentiated=self.undifferentiated, kinetics=False)
        entry_num = s.shape[0]
        loss_val = loss / entry_num
        self.test_loss_list.append(loss_val)
        self.test_z_list.append(z_list)
        return(loss_val)

    def train_total(self, epoch_num):
        for epoch in range(epoch_num):
            state_dict = copy.deepcopy(self.model.state_dict())
            loss = self.train_epoch()
            if np.isnan(loss):
                self.model.load_state_dict(state_dict)
                break
            if epoch % 10 == 0:
                print(f'loss at epoch {epoch} is {loss}')

    def embed_train_epoch(self):
        self.model.train()
        total_loss = 0
        entry_num = 0
        for s, u, day, norm_mat in self.edm.train_loader:
            s = s.to(self.device)
            u = u.to(self.device)
            day = day.to(self.device)
            norm_mat = norm_mat.to(self.device)
            self.optimizer.zero_grad()
            loss = self.model.embed_loss(
                s, u, day, norm_mat)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            entry_num += s.shape[0]
        return(total_loss / entry_num)

    def embed_test(self):
        self.model.eval()
        s = self.edm.test_s.to(self.device)
        u = self.edm.test_u.to(self.device)
        day = self.edm.test_day.to(self.device)
        norm_mat = self.edm.test_norm_mat.to(self.device)
        loss = self.model.embed_loss(
            s, u, day, norm_mat)
        entry_num = s.shape[0]
        loss_val = loss / entry_num
        return(loss_val)

    def embed_train_total(self, epoch_num):
        for epoch in range(epoch_num):
            state_dict = copy.deepcopy(self.model.state_dict())
            loss = self.embed_train_epoch()
            if np.isnan(loss):
                self.model.load_state_dict(state_dict)
                break
            if epoch % 10 == 0:
                print(f'loss at epoch {epoch} is {loss}')

    def get_forward(self, s):
        z_T0, qz_T0, ld, pu_zd_ld, qz_mu_list, qz_logvar_list, z_list, z0_list, qd_loc_list, qd_scale_list = self.model(s)
        return(z_T0, qz_T0, ld, pu_zd_ld, qz_mu_list, qz_logvar_list, z_list, z0_list, qd_loc_list, qd_scale_list)

    def init_optimizer(self, lr):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    