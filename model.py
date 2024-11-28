import torch
import torch.nn as nn
import math

from utils import read_config_yaml


class Embedding(nn.Module):
    def __init__(self, vocab_size : int, embed_size : int):
        super(Embedding, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size)

    def forward(self, x):
        return self.embedding(x)
    



class PositionalEncoding(nn.Module):
    def __init__(self, embed_size : int, seq_len : int):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(seq_len, embed_size)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_size, 2).float() * -(math.log(10000)/embed_size))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]



class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size : int, num_heads : int, qkv_bias : bool = False):
        super(MultiHeadAttention, self).__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads
        self.qkv_bias = qkv_bias
        
        self.query = nn.Linear(in_features=embed_size, out_features=embed_size, bias=qkv_bias)
        self.key = nn.Linear(in_features=embed_size, out_features=embed_size, bias=qkv_bias)
        self.value = nn.Linear(in_features=embed_size, out_features=embed_size, bias=qkv_bias)

        self.out = nn.Linear(in_features=embed_size, out_features=embed_size)

    def forward(self, x, mask=None):
        batch_size = x.shape[0]

        q = self.query(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        attention = torch.matmul(q, k.transpose(-1, -2))
        scaled_attention = attention / math.sqrt(self.head_dim)

        if mask is not None:
            scaled_attention = scaled_attention.masked_fill(mask==0, float('inf'))

        scaled_attention = torch.softmax(scaled_attention, dim=-1)

        out = torch.matmul(scaled_attention, v)

        out = out.transpose(1, 2)
        out = out.contiguous().view(batch_size, -1, self.embed_size)

        return out



class MLP(nn.Module):
    def __init__(self, embed_size : int, hidden_size : int):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_features=embed_size, out_features=hidden_size)
        self.fc2 = nn.Linear(in_features=hidden_size, out_features=embed_size)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.gelu(self.fc1(x))
        return self.fc2(x)
    


class TransformerBlock(nn.Module):
    def __init__(self, num_heads : int, embed_size : int, hidden_size : int, dropout : float = 0.1, qkv_bias : bool = False):
        super(TransformerBlock, self).__init__()
        self.mha = MultiHeadAttention(embed_size, num_heads, qkv_bias)
        self.ln1 = nn.LayerNorm(embed_size)
        self.ln2 = nn.LayerNorm(embed_size)
        self.mlp = MLP(embed_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attention_output = self.ln1(x + self.dropout(self.mha(x, mask)))
        proj = self.ln2(attention_output + self.dropout(self.mlp(attention_output)))
        return proj


class GPT2(nn.Module):
    def __init__(self, config):
        super(GPT2, self).__init__()
        self.embedding = Embedding(vocab_size=config['vocab_size'], embed_size=config['embed_size'])
        self.pos_encoding = PositionalEncoding(embed_size=config['embed_size'], seq_len=config['seq_len'])

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(config['num_heads'], config['embed_size'], config['embed_size']*4, config['dropout'], config['qkv_bias'])
            for _ in range(config['num_layers'])
        ])

        self.projection = nn.Linear(config['embed_size'], config['vocab_size'])
        self.dropout = nn.Dropout(config['dropout'])

    def forward(self, x, mask=None):
        x = self.dropout(self.pos_encoding(self.embedding(x)))

        for block in self.transformer_blocks:
            x = block(x, mask)

        return self.projection(x)








"""
GPT_CONFIG_124M = read_config_yaml(r'GPT_CONFIG_124M.yaml')

batch_size = 1

embed_layer = Embedding(GPT_CONFIG_124M['vocab_size'], GPT_CONFIG_124M['embed_size'])
x = torch.randint(0, GPT_CONFIG_124M['vocab_size'], (batch_size, GPT_CONFIG_124M['seq_len']))
print(x.shape)
embed = embed_layer(x)
print(embed.shape)

pos_emb = PositionalEncoding(GPT_CONFIG_124M['embed_size'], GPT_CONFIG_124M['seq_len'])
pos_emb = pos_emb(embed)
print(pos_emb.shape)

transformer = TransformerBlock(num_heads=GPT_CONFIG_124M['num_heads'], embed_size=GPT_CONFIG_124M['embed_size'], hidden_size=GPT_CONFIG_124M['embed_size']*4)
out = transformer(pos_emb)
print(out.shape)



gpt = GPT2(config=GPT_CONFIG_124M)
x = gpt(x)
print(x.shape)
"""