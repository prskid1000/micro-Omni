
import math, torch
from torch import nn
from omni.utils import RMSNorm, RoPE, make_positions
from einops import rearrange

class MLP(nn.Module):
    def __init__(self, d, ff):
        super().__init__()
        self.fc1 = nn.Linear(d, ff)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(ff, d)
    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

class Attention(nn.Module):
    def __init__(self, d, heads, rope_theta=10000.0, dropout=0.0):
        super().__init__()
        self.h = heads
        self.d = d
        self.dk = d // heads
        self.qkv = nn.Linear(d, 3*d, bias=False)
        self.o  = nn.Linear(d, d, bias=False)
        self.rope = RoPE(self.dk, theta=rope_theta)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask=None, pos=None):
        B, T, D = x.shape
        qkv = self.qkv(x).chunk(3, dim=-1)  # 3 * (B,T,D)
        q, k, v = [rearrange(t, "b t (h d) -> b h t d", h=self.h) for t in qkv]
        if pos is None: pos = make_positions(T, x.device)
        q, k = self.rope(q, k, pos)
        att = torch.einsum("bhtd,bhTd->bhtT", q, k) / math.sqrt(self.dk)
        if mask is not None:
            att = att.masked_fill(mask==0, float("-inf"))
        att = att.softmax(dim=-1)
        att = self.drop(att)
        y = torch.einsum("bhtT,bhTd->bhtd", att, v)
        y = rearrange(y, "b h t d -> b t (h d)")
        return self.o(y)

class Block(nn.Module):
    def __init__(self, d, heads, ff, rope_theta, dropout):
        super().__init__()
        self.norm1 = RMSNorm(d)
        self.attn = Attention(d, heads, rope_theta=rope_theta, dropout=dropout)
        self.norm2 = RMSNorm(d)
        self.mlp = MLP(d, ff)
        self.drop = nn.Dropout(dropout)
    def forward(self, x, mask=None, pos=None):
        x = x + self.drop(self.attn(self.norm1(x), mask=mask, pos=pos))
        x = x + self.drop(self.mlp(self.norm2(x)))
        return x

class ThinkerLM(nn.Module):
    def __init__(self, vocab, n_layers=16, d=512, heads=8, ff=2048, dropout=0.1, rope_theta=10000, ctx=1024):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab, d)
        self.pos_cache = None
        self.blocks = nn.ModuleList([Block(d, heads, ff, rope_theta, dropout) for _ in range(n_layers)])
        self.norm = RMSNorm(d)
        self.lm_head = nn.Linear(d, vocab, bias=False)
        self.ctx = ctx

    def forward(self, idx, attn_mask=None):
        # idx: (B,T)
        x = self.tok_emb(idx)
        T = idx.shape[1]
        pos = make_positions(T, x.device)
        mask = torch.tril(torch.ones(T, T, device=idx.device)).unsqueeze(0).unsqueeze(0)  # causal
        if attn_mask is not None:
            mask = mask * attn_mask
        for blk in self.blocks:
            x = blk(x, mask=mask, pos=pos)
        x = self.norm(x)
        logits = self.lm_head(x)
        return logits
