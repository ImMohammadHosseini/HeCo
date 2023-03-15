
import torch.nn as nn
from mp_encoder import Mp_encoder
from sc_encoder import Sc_encoder

class HeCo (nn.Module):
    def __init__ (self):    
        super(HeCo, self).__init__()
        