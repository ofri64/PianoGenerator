import torch
from torch import nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, k, num_heads):
        super().__init__()
        self.k = k
        self.num_heads = num_heads
        self.to_keys = nn.Linear(in_features=k, out_features=num_heads * k, bias=False)
        self.to_queries = nn.Linear(in_features=k, out_features=num_heads * k, bias=False)
        self.to_values = nn.Linear(in_features=k, out_features=num_heads * k, bias=False)
        self.heads_unifier = nn.Linear(in_features=k * num_heads, out_features=k, bias=False)

    def forward(self, x):
        batch, seq, _ = x.size()
        h = self.num_heads
        k = self.k

        keys = self.to_keys(x).view(batch, seq, h, k)
        queries = self.to_queries(x).view(batch, seq, h, k)
        values = self.to_values(x).view(batch, seq, h, k)

        # fold heads into the batch dimension
        keys = keys.transpose(1, 2).contiguous().view(batch * h, seq, k)
        queries = queries.transpose(1, 2).contiguous().view(batch * h, seq, k)
        values = values.transpose(1, 2).contiguous().view(batch * h, seq, k)

        # scale the queries and keys dot product
        queries = queries / (self.k ** (1 / 4))
        keys = keys / (self.k ** (1/4))

        # get dot product of queries and keys, and scale
        # dot has size (batch * h, seq, seq) containing raw weights (h sequences, one for each head)
        dot = torch.bmm(queries, keys.transpose(1, 2))

        # add masking for auto regressive model
        mask = torch.tril(torch.ones(seq, seq)) == 1
        dot[:, ~mask] = float('-inf')
        # mask_indices = torch.triu_indices(k, k, offset=0)
        # dot[:, mask_indices[0], mask_indices[1]] = float('-inf')

        # apply softmax to create final normalized weight
        weights = F.softmax(dot, dim=2)

        # apply the self attention to the values
        heads_out = torch.bmm(weights, values).view(batch, h, seq, k)

        # concatenate heads
        heads_out = heads_out.transpose(1, 2).contiguous().view(batch, seq, h * k)

        # unify heads (output has dimension (batch, seq, features)
        output = self.heads_unifier(heads_out)

        return output


class TransformerBlock(nn.Module):
    def __init__(self, k, num_heads):
        super().__init__()
        self.attention_layer = SelfAttention(k, num_heads)
        self.norm1 = nn.LayerNorm(k)
        self.norm2 = nn.LayerNorm(k)
        self.ff = nn.Sequential(
            nn.Linear(in_features=k, out_features=4*k),
            nn.ReLU(),
            nn.Linear(in_features=4*k, out_features=k)
        )

    def forward(self, x):
        attended = self.attention_layer(x)
        x = self.norm1(attended + x)
        fed_forward = self.ff(x)
        x = self.norm2(fed_forward + x)

        return x


class TransformerLayer(nn.Module):
    def __init__(self, embedding_dim, num_mappings, num_heads, num_blocks, seq_length, padding_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=num_mappings, embedding_dim=embedding_dim, padding_idx=padding_idx)
        self.position_embedding = nn.Embedding(num_embeddings=seq_length, embedding_dim=embedding_dim)

        transformer_blocks = []
        for i in range(num_blocks):
            transformer_blocks.append(TransformerBlock(embedding_dim, num_heads))

        self.transformer_blocks = nn.Sequential(*transformer_blocks)

    def forward(self, x):
        # create discrete tokens embeddings
        embeddings = self.embedding(x)

        # create positional embeddings
        batch, seq, features = embeddings.size()
        positions = torch.arange(seq).to(embeddings.device)
        position_embeddings = self.position_embedding(positions)[None, :, :].expand(batch, seq, features)

        # add tokens embeddings and positional embeddings
        x = embeddings + position_embeddings

        # run through transformer blocks
        out = self.transformer_blocks(x)

        return out





