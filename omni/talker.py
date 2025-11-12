
import torch
from torch import nn
from omni.utils import RMSNorm

class TalkerTiny(nn.Module):
    """ AR Transformer that predicts 2 RVQ codebooks per frame (MTP=2). """
    def __init__(self, d=384, n_layers=8, n_heads=6, ff=1536, codebooks=2, codebook_size=128, dropout=0.1):
        super().__init__()
        self.emb = nn.Embedding(codebook_size, d)
        self.start = nn.Parameter(torch.zeros(1,1,d))
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d, n_heads, ff, dropout, batch_first=True, norm_first=True, activation='gelu')
            for _ in range(n_layers)
        ])
        self.norm = RMSNorm(d)
        self.base_head = nn.Linear(d, codebook_size)
        self.res_head  = nn.Linear(d, codebook_size)

    def forward(self, prev_codes):
        # prev_codes: (B, T, 2) previous frame codes (base,residual)
        B,T,_ = prev_codes.shape
        x = self.emb(prev_codes[:,:,0]) + self.emb(prev_codes[:,:,1])
        start = self.start.expand(B, -1, -1)
        x = torch.cat([start, x], dim=1)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        logits = (self.base_head(x[:,1:,:]), self.res_head(x[:,1:,:]))
        return logits
