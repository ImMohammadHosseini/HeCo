
import torch.nn as nn
import torch.nn.functional as F

from .mp_encoder import Mp_encoder
from .sc_encoder import Sc_encoder
from copy import deepcopy


class HeCo (nn.Module):
    def __init__ (self, hidden_dim, nodes_type, projection_drop, attn_drop, P):    
        super(HeCo, self).__init__()
        self.nodes_type = nodes_type
        
        self.node_project = nn.ModuleList([nn.LazyLinear(hidden_dim, bias=True)
                                         for _ in range(len(nodes_type))])
        
        #for projection in self.node_project:
        #    nn.init.xavier_normal_(projection.weight, gain=1.414)
            
        if projection_drop > 0:
            self.feat_drop = nn.Dropout(projection_drop)
        else:
            self.feat_drop = lambda x: x
         
        self.mp = Mp_encoder(P, hidden_dim, attn_drop)
        self.sc = Sc_encoder(hidden_dim, attn_drop)
        
        self.contrastiveProj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    def forward (self, graph, mps, mode="test"):
        for i, ntype in enumerate(self.nodes_type):
            graph[ntype].x = F.elu(self.feat_drop(
                self.node_project[i](graph[ntype].x)))
        
        z_mp = self.mp(deepcopy(graph['author'].x), mps)
        z_sc = self.sc(graph)
        
        if mode == "train":
            z_proj_mp = self.proj(z_mp)
            z_proj_sc = self.proj(z_sc)
            return z_proj_mp, z_proj_sc
        else: return z_mp.detach() 