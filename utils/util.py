
import os
from torch_geometric.datasets.dblp import DBLP
from torch_geometric.datasets.aminer import AMiner


def set_defaults(parser):
    opts, args = parser.parse_args()
    dataset = opts.dataset
    if dataset == "acm":
        pass
    elif dataset == "dblp":
        parser.set_defaults(seed=53, eva_lr=0.01, patience=30, lr=0.0008, 
                            tau=0.9, feat_drop=0.4, attn_drop=0.35, 
                            sample_rate=[6])
    elif dataset == "aminer":
        parser.set_defaults(seed=4, eva_lr=0.01, patience=40, lr=0.003, 
                            tau=0.5, feat_drop=0.5, attn_drop=0.5, 
                            sample_rate=[3,8])
    elif dataset == "freebase":pass

        
def load_dataset(name):
    if name == "acm":pass
    elif name == "dblp":
        return DBLP(os.getcwd()+'/datasets/dblp').data
    elif name == "aminer":
        return AMiner(os.getcwd()+'/datasets/aminer').data
    elif name == "freebase":pass
