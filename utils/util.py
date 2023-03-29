
import os
from os import path, makedirs
import torch
from torch_geometric.datasets.dblp import DBLP
from torch_geometric.datasets.aminer import AMiner
import numpy as np
import itertools
from copy import deepcopy
from scipy.sparse import csr_matrix, save_npz, load_npz

class mp_type:
    DBLP = ['author_paper_author', 'author_paper_conference_paper_author',
            'author_paper_term_paper_author']
    ACM = []
    

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

    
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


def getMatrixs (graphData, mpTypes, savePath):
    makedirs(savePath)
    pos = csr_matrix((4057, 4057), dtype=np.int8)
    mps = []
    #newLinkType = []
    for mpt in mpTypes:
        nodes = [node for node in mpt.split('_')[int(len(mpt.split('_'))/2):]]
        new_nei = []
        sourceName=nodes[-1]; destName=nodes[-1] 
        linkName=nodes[1]+'_'+nodes[0]+'_'+nodes[1] if len(nodes)>2 else nodes[0]
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
                for i in new_nei: pos[i] += 1
                
            else:
                half_nei = deepcopy(new_nei)
                new_nei = []
                #half_source = deepcopy(new_source)
                #half_dest = deepcopy(new_source)
                #unique_source = np.unique(half_source)
                for ind in range(len(half_nei[0])):
                    first_part = half_nei[0][ind]s
                    fp_indx = np.where(source == first_part)
                    second_part = half_nei[1][ind]
                    sp_indx = np.where(source == second_part)
                    for ns in dest[fp_indx]: 
                        for nd in dest[sp_indx]:
                            new_nei.append((ns,nd))
                            pos[ns,nd] += 1
                    #new_nei+=[(ns,nd) for ns in dest[fp_indx] for nd in dest[sp_indx]]
            new_nei=list(np.array(new_nei).T)
            new_source=list(new_nei[0])+list(new_nei[1])
            new_dest=list(new_nei[1])+list(new_nei[0])
            new_nei=np.unique([new_source,new_dest],axis=1)
        torch.save(torch.tensor(new_nei), savePath+'/'+mpt+'.pt')
        mps.append(torch.tensor(new_nei)) 
        #graphData[sourceName, linkName, destName].edge_index = torch.tensor(
        #                                                                new_nei)
        #newLinkType.append([sourceName, linkName, destName])
    save_npz(savePath+'/'+'pos.pt', pos)
    return mps, pos
    #return newLinkType
    
def loadMatrixs (mpTypes, savePath):
    mps=[]
    pos = load_npz(savePath + "pos.npz")
    for mpt in mpTypes:
        mps.append(torch.load(savePath+'/'+mpt+'.pt'))
    return mps, pos
    
def load_dataset(name, pos_threshold):
    if name == "acm":pass
    elif name == "dblp":
        datasetPath=os.getcwd()+'/datasets/dblp'
        graphData = DBLP(datasetPath).data
        if not path.exists(datasetPath+'/matix'):mps=getMatrixs(graphData, 
                                                              mp_type.DBLP, 
                                                              datasetPath+'/matix')
        else : mps, pos=loadMatrixs(mp_type.DBLP, datasetPath+'/matix')
        pos = pos > pos_threshold
        print (pos)
        pos = sparse_mx_to_torch_sparse_tensor(pos)
        return graphData, mps, pos
    elif name == "aminer":
        return AMiner(os.getcwd()+'/datasets/aminer').data
    elif name == "freebase":pass
