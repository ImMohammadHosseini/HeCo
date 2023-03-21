
import os
import torch
from torch_geometric.datasets.dblp import DBLP
from torch_geometric.datasets.aminer import AMiner
import numpy as np
import itertools
from copy import deepcopy

class mp_type:
    DBLP = ['AUTHOR_PAPER_AUTHOR', 'AUTHOR_PAPER_CONFERENCE_PAPER_AUTHOR',
            'AUTHOR_PAPER_TERM_PAPER_AUTHOR']
    ACM = []
    
    
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


def getMPs (graphData, mpTypes):
    mps = []
    for mpt in mpTypes:
        nodes = [node for node in mpt.split('_')[int(len(mpt.split('_'))/2):]]
        new_nei = []
        for i in range(0, len(nodes)-1):
            for graphEdge in graphData.edge_types:
                if graphEdge[0] == nodes[i] and graphEdge[2] == nodes[i+1]:
                    source = graphData[graphEdge].edge_index[0]
                    source = source.detach().numpy()
                    dest = graphData[graphEdge].edge_index[1]
                    dest = dest.detach().numpy()
                    break
            if i == 0:
                unique_source = np.unique(source)
                for us in unique_source:
                    indx = np.where(source == us)
                    if len(indx[0]) > 1:
                        new_nei+=list(itertools.combinations(dest[indx[0]],2))
                
            else:
                half_nei = deepcopy(new_nei)
                new_nei = []
                #half_source = deepcopy(new_source)
                #half_dest = deepcopy(new_source)
                #unique_source = np.unique(half_source)
                for ind in range(len(half_nei[0])):
                    first_part = half_nei[0][ind]
                    fp_indx = np.where(source == first_part)
                    second_part = half_nei[1][ind]
                    sp_indx = np.where(source == second_part)
                    new_nei+=[(ns,nd) for ns in dest[fp_indx] for nd in dest[sp_indx]]
            new_nei=list(np.array(new_nei).T)
            new_source=list(new_nei[0])+list(new_nei[1])
            new_dest=list(new_nei[1])+list(new_nei[0])
            new_nei=np.unique([new_source,new_dest],axis=1)
        mps.append(torch.tensor(new_nei))        
    return mps
    
def load_dataset(name):
    if name == "acm":pass
    elif name == "dblp":
        graphData = DBLP(os.getcwd()+'/datasets/dblp').data
        return graphData, getMPs(graphData, mp_type.DBLP)
    elif name == "aminer":
        return AMiner(os.getcwd()+'/datasets/aminer').data
    elif name == "freebase":pass
