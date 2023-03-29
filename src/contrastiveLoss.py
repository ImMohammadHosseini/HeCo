
import torch
from torch import nn

class contrastiveLoss(nn.Module):
    def __init__(self,  lam, tau):
        super(contrastiveLoss, self).__init__()
        self.lam = lam
        self.tau = tau
        
    def sim(self, z1, z2):
        z1_norm = torch.norm(z1, dim=-1, keepdim=True)
        z2_norm = torch.norm(z2, dim=-1, keepdim=True)
        dot_numerator = torch.mm(z1, z2.t())
        dot_denominator = torch.mm(z1_norm, z2_norm.t())
        sim_matrix = torch.exp(dot_numerator / dot_denominator / self.tau)
        return sim_matrix
    
    def forward (self, z_mp, z_sc, pos):
        matrix_mp2sc = self.sim(z_mp, z_sc)
        matrix_sc2mp = matrix_mp2sc.t()
    
        matrix_mp2sc = matrix_mp2sc/(torch.sum(matrix_mp2sc, dim=1).view(-1, 1) + 1e-8)
        print('loss')
        print(z_mp.size())
        print(z_sc.size())
        print(matrix_mp2sc.size())
        print(pos.size())
        lori_mp = -torch.log(matrix_mp2sc.mul(pos.to_dense()).sum(dim=-1)).mean()

        matrix_sc2mp = matrix_sc2mp / (torch.sum(matrix_sc2mp, dim=1).view(-1, 1) + 1e-8)
        lori_sc = -torch.log(matrix_sc2mp.mul(pos.to_dense()).sum(dim=-1)).mean()
        return self.lam * lori_mp + (1 - self.lam) * lori_sc
