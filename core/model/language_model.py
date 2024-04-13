import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class GPTConfig:
    device: torch.device
    vocab_size: int = 25
    num_layers: int = 3
    num_heads: int = 3
    embed_size: int = 512
    dropout: float = 0.0
    ctx_size: int = 512


class Head(nn.Module):
    def __init__(self, embed_size, head_embed_size, ctx_size, dropout=0):
        super().__init__()
        self.key = nn.Linear(embed_size, head_embed_size, bias=False)
        self.query = nn.Linear(embed_size, head_embed_size, bias=False)
        self.value = nn.Linear(embed_size, head_embed_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("tril", torch.tril(torch.ones((ctx_size, ctx_size))))

    def forward(self, inputs):
        _, T, _ = inputs.shape  # batch, ctx, vocab_size
        k = self.key(inputs)  # batch, ctx, head_size
        q = self.query(inputs)  # batch, ctx, head_size
        mask = q @ k.transpose(-2, -1)  # batch, ctx, ctx
        mask = mask.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        mask = F.softmax(mask, dim=-1)
        mask = self.dropout(mask)
        v = self.value(inputs)  # batch, ctx, head_size
        out = mask @ v  # batch, ctx, head_size
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, embed_size, ctx_size, dropout=0):
        super().__init__()
        head_embed_size = embed_size // num_heads
        self.heads = nn.ModuleList([Head(embed_size, head_embed_size, ctx_size, dropout) for _ in range(num_heads)])
        self.ff = nn.Linear(head_embed_size * num_heads, embed_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs):
        # batch, ctx, embed_size
        inputs = torch.cat([head(inputs) for head in self.heads],
                           dim=-1)  # batch, ctx, num_head*head_size --> batch, ctx, embed_size
        inputs = self.ff(inputs)  # batch, ctx, embed_size
        inputs = self.dropout(inputs)
        return inputs


class Block(nn.Module):
    def __init__(self, num_heads, embed_size, ctx_size, dropout=0):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_size)
        self.mha = MultiHeadAttention(num_heads, embed_size, ctx_size, dropout)
        self.ln2 = nn.LayerNorm(embed_size)
        self.ff = nn.Sequential(
            nn.Linear(embed_size, embed_size),
            nn.ReLU(),
            nn.Linear(embed_size, embed_size),
            nn.Dropout(dropout)
        )

    def forward(self, inputs):
        inputs = self.mha(self.ln1(inputs)) + inputs  # batch, ctx, embed_size
        inputs = self.ff(self.ln2(inputs)) + inputs  # batch, ctx, embed_size
        return inputs


class LanguageModel(nn.Module):
    def __init__(self, config, stoi, itos):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.embed_size)
        self.pos_embedding = nn.Embedding(config.ctx_size, config.embed_size)
        self.blocks = nn.Sequential(*[Block(config.num_heads, config.embed_size, config.ctx_size, config.dropout) for _ in range(config.num_layers)])
        self.ln = nn.LayerNorm(config.embed_size)
        self.ff = nn.Linear(config.embed_size, config.vocab_size)
        self.stoi = stoi
        self.itos = itos

    def forward(self, inputs, targets=None):
        # batch, ctx, vocab_size
        tok_emb = self.embedding(inputs) # batch, ctx, embed_size
        pos_emb = self.pos_embedding(torch.arange(inputs.size(1), device=self.config.device))
        logits = tok_emb + pos_emb
        logits = self.blocks(logits)  # batch, ctx, embed_size
        logits = self.ln(logits)  # batch, ctx, embed_size
        logits = self.ff(logits)  # batch, ctx, vocab_size

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            lv = logits.view((B * T, C))
            tv = targets.view((B * T,))
            loss = F.cross_entropy(lv, tv)

        return logits, loss

    def generate(self, tokens, decode, n=256):
        for _ in range(n):
            logits, loss = self(tokens[:, -self.config.ctx_size:])
            logits = logits[:, -1, :]
            logits = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(logits, num_samples=1)
            if decode(next_token[0].cpu().numpy()) == '.':
                break
            tokens = torch.cat((tokens, next_token), dim=1)
        return tokens

    def calculate_loss(self, batch_tokens, targets, mode='psm', indexes=None, device=None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        if mode == 'psm':
            mask = torch.zeros_like(batch_tokens).to(device)

            hash_index = torch.tensor(self.stoi['#']).to(device)
            indices = (batch_tokens == hash_index).nonzero()[:, 1] + 1

            for b, idx in enumerate(indices):
                mask[b, idx:] = 1

            pad_index = torch.tensor(self.stoi['0']).to(device)
            mask = (mask & (batch_tokens != pad_index)).to(torch.long)

            logits, _ = self(batch_tokens)
            logits = logits[:, -batch_tokens.shape[1] + 1:, :]
            loss = F.cross_entropy(logits.permute(0, 2, 1), targets[:, 1:], reduction='none')
            loss = (loss * mask[:, 1:]).sum() / mask.sum()

        elif mode == 'pms':
            mask = torch.zeros_like(batch_tokens).to(device)

            for b, idx in enumerate(indexes):
                mask[b, idx:] = 1

            pad_index = torch.tensor(self.stoi['0']).to(device)
            mask = (mask & (batch_tokens != pad_index)).to(torch.long)

            logits, _ = self(batch_tokens)
            logits = logits[:, -batch_tokens.shape[1] + 1:, :]
            loss = F.cross_entropy(logits.permute(0, 2, 1), targets[:, 1:], reduction='none')

            loss = (loss * mask[:, 1:]).sum() / mask.sum()

        elif mode == 'default':
            pad_index = torch.tensor(self.stoi['0']).to(device)
            mask = (batch_tokens != pad_index).to(torch.long)
            logits, _ = self(batch_tokens)
            logits = logits[:, -batch_tokens.shape[1] + 1:, :]
            loss = F.cross_entropy(logits.permute(0, 2, 1), targets[:, 1:], reduction='none')
            loss = (loss * mask[:, 1:]).sum() / mask.sum()

        return loss
