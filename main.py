

import numpy
import torch
from utils.util import load_dataset, set_defaults
from src.models.heco import HeCo
from src.trainer.trainer import EmbeddingTrainer
import warnings
import datetime
import pickle as pkl
import os
import random
import optparse


usage = "usage: python main.py -d acm -n <nn>"

parser = optparse.OptionParser(usage=usage)
parser.add_option("-d", "--dataset", dest="dataset", default = 'dblp',
				  help="can suport: (default=acm)")
parser.add_option('--save_emb', action="store_true")
parser.add_option('--turn', type=int, default=0)
parser.add_option('--ratio', type=int, default=[20, 40, 60])
parser.add_option('--seed', type=int)
parser.add_option('--hidden_dim', type=int, default=64)
parser.add_option('--nb_epochs', type=int, default=10000)
    
# The parameters of evaluation
parser.add_option('--eva_lr', type=float)
parser.add_option('--eva_wd', type=float, default=0)
    
# The parameters of learning process
parser.add_option('--patience', type=int)
parser.add_option('--lr', type=float)
parser.add_option('--l2_coef', type=float, default=0)
    
# model-specific parameters
parser.add_option('--tau', type=float)
parser.add_option('--feat_drop', type=float)
parser.add_option('--attn_drop', type=float)
parser.add_option('--sample_rate', nargs='+', type=int,)
parser.add_option('--lam', type=float, default=0.5)
parser.add_option('--eva_lr', type=float)

set_defaults(parser)
opts, args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

seed = opts.seed
numpy.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

if __name__ == "__main__":
    data = load_dataset (opts.dataset)
    
    model = HeCo()
    embeddingTrainer = EmbeddingTrainer(model)



