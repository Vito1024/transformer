import torch
from torch import nn
from torch.nn import functional as F

embedding_size = 2**10 # 输入x的长度
n_head = 2**3


class Head(nn.Module):

    def __init__(self, head_size):
        super().__init__()
        
        # weight of query
        self.query_weight = nn.Linear(embedding_size, head_size, bias=False)
        # weight of key
        self.key_weight = nn.Linear(embedding_size, head_size, bias=False)
        # weight of value
        self.value_weight = nn.Linear(embedding_size, head_size, bias=False)

        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # input of size(batch, time-step, channels)
        # output of size(batch, time-step, hs: head size)
        B, T, C = x.shape # C == embedding_size

        Q = self.query_weight(x) # x @ self.query_weight => (B, T, hs); channels = embedding_size
        K = self.key_weight(x) # x @ self.key_weight => (B, T, hs)
        V = self.value_weight(x) # x @ self.value_weight => (B, T, hs)

        weights = F.softmax(Q @ K.transpose(-2, -1) * K.shape[-1]**0.5)
        weights = self.dropout(weights)
        attention = weights @ V # (B, T, T) @ (B, T, hs) => (B, T, hs)

        return attention


class MultiHeadAttention(nn.Module):

    def __init__(self, num_head, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_head)])

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return out


class FeedForward(nn.Module):

    def __init__(self, embedding_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_size, 4 * embedding_size),
            nn.ReLU(),
            nn.Linear(embedding_size * 4, embedding_size),
            nn.Dropout(0.2),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    
    def __init__(self, embedding_size, n_head):
        super().__init__()
        head_size = embedding_size // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(embedding_size)
        self.ln1 = nn.LayerNorm(embedding_size)
        self.ln2 = nn.LayerNorm(embedding_size)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
