
import torch.nn as nn


class Mp_encoder(nn.Module):
    def __init__(self, P, hidden_dim, attn_drop):
        super(Mp_encoder, self).__init__()

