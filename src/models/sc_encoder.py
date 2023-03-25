
import torch
import torch.nn as nn
from torch_geometric.nn import HeteroConv, GATConv


class inter_att(nn.Module):
    def __init__(self, hidden_dim, attn_drop):
        super(inter_att, self).__init__()
        self.fc = nn.Linear(hidden_dim, hidden_dim, bias=True)
        nn.init.xavier_normal_(self.fc.weight, gain=1.414)

        self.tanh = nn.Tanh()
        self.att = nn.Parameter(torch.empty(size=(1, hidden_dim)), requires_grad=True)
        nn.init.xavier_normal_(self.att.data, gain=1.414)

        self.softmax = nn.Softmax()
        if attn_drop:
            self.attn_drop = nn.Dropout(attn_drop)
        else:
            self.attn_drop = lambda x: x

    def forward(self, embeds):
        beta = []
        attn_curr = self.attn_drop(self.att)
        for embed in embeds:
            sp = self.tanh(self.fc(embed)).mean(dim=0)
            beta.append(attn_curr.matmul(sp.t()))
        beta = torch.cat(beta, dim=-1).view(-1)
        beta = self.softmax(beta)
        print("sc ", beta.data.cpu().numpy())  # type-level attention
        z_mc = 0
        for i in range(len(embeds)):
            z_mc += embeds[i] * beta[i]
        return z_mc



class Sc_encoder(nn.Module):
    #TODO check sample_rate
    def __init__(self, hidden_dim, attn_drop):
        super(Sc_encoder, self).__init__()
        
        self.conv = HeteroConv({('paper', 'to', 'author'):GATConv((-1,-1), 
                    hidden_dim, dropout=attn_drop, add_self_loops=False),
            }, aggr='sum')
        
        self.inter = inter_att(hidden_dim, attn_drop)
        
    def forward (self, graph):
        #TODO SAMPLE
        x_dict = graph.x_dict
        edge_index_dict = graph.edge_index_dict
        
        x_dict = self.conv(x_dict, edge_index_dict)
        x_dict = {key: x.relu() for key, x in x_dict.items()}
            
        authorNode = x_dict['author']
        z_mc = self.inter(authorNode)
        return z_mc