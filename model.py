"""A decoder-only transformer (GPT-style).

Same pre-norm block structure as the seq2seq decoder in the Xuyevo transformer,
with the cross-attention sub-layer removed: there is no encoder to attend to,
so each block is just causal self-attention + feed-forward.
"""

import inspect
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 50304   # gpt2's 50257 padded to a multiple of 64
    block_size: int = 512     # context length
    n_layer: int = 8          # number of transformer blocks
    n_head: int = 8           # number of attention heads in multi head attention layer
    n_embd: int = 512         # each token embedding vector is this many dimensions
    d_ff: int = 2048          # 4 * n_embd, inner dimension of the feed-forward layer
    dropout: float = 0.0      # raise to ~0.1 if you train well past ~2 epochs
    bias: bool = False


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention where each position sees only the past."""

    def __init__(self, cfg):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        self.dropout = cfg.dropout

        # One fused projection for Q, K and V — a third of the matmul launches.
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.resid_dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        B, T, C = x.shape

        q, k, v = self.qkv(x).split(self.n_embd, dim=2)
        # (B, T, C) -> (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # is_causal builds the mask internally and never materialises the
        # (B, n_head, T, T) score matrix, which is what makes this fit in RAM.
        y = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.proj(y))


class FeedForward(nn.Module):
    """Position-wise feed-forward network."""

    def __init__(self, cfg):
        super().__init__()
        self.fc = nn.Linear(cfg.n_embd, cfg.d_ff, bias=cfg.bias)
        self.proj = nn.Linear(cfg.d_ff, cfg.n_embd, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.dropout(self.proj(F.gelu(self.fc(x))))


class Block(nn.Module):
    """Pre-norm transformer block: x + attn(norm(x)), then x + ffn(norm(x))."""

    def __init__(self, cfg):
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.attn = CausalSelfAttention(cfg)
        self.norm2 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.ffn = FeedForward(cfg)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)   # token embeddings
        self.wpe = nn.Embedding(cfg.block_size, cfg.n_embd)   # learned positions
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.norm = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

        # Weight tying (Press & Wolf, 2017). Saves 12.9M params at this size, 
        # which is most of the model.
        self.lm_head.weight = self.wte.weight 

        self.apply(self._init_weights)
        # Scale the residual projections so the residual stream variance does
        # not grow with depth (GPT-2 §2.3).
        for name, p in self.named_parameters():
            if name.endswith('proj.weight'):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self, non_embedding=True):
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.wpe.weight.numel()  # wte is tied to lm_head, so counted once
        return n

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        """AdamW with decay on matmul weights only.

        Biases and LayerNorm gains are 1-D and decaying them just fights the
        normalisation; embeddings are excluded for the same reason.
        """
        params = [p for p in self.parameters() if p.requires_grad]
        decay = [p for p in params if p.dim() >= 2]
        no_decay = [p for p in params if p.dim() < 2]
        groups = [
            {'params': decay, 'weight_decay': weight_decay},
            {'params': no_decay, 'weight_decay': 0.0},
        ]
        # The fused kernel is a sizeable win on CUDA and unavailable elsewhere.
        fused = device_type == 'cuda' and 'fused' in inspect.signature(torch.optim.AdamW).parameters
        return torch.optim.AdamW(groups, lr=learning_rate, betas=betas, fused=fused)

    def estimate_mfu(self, tokens_per_iter, dt, flops_promised):
        """Model FLOPs utilisation: fraction of the GPU's peak we're actually using."""
        n = self.num_params(non_embedding=False)
        flops_per_token = 6 * n + 12 * self.cfg.n_layer * self.cfg.n_head * \
            (self.cfg.n_embd // self.cfg.n_head) * self.cfg.block_size
        return (flops_per_token * tokens_per_iter / dt) / flops_promised

    def forward(self, idx, targets=None):
        """
        Args:
            idx:     Token ids — (batch, seq_len), values in [0, vocab_size)
            targets: Same shape, each position's next token; None when generating.
                     -100 marks a position to skip, which is how SFT trains on
                     assistant turns only.
        Returns:
            (logits, loss). loss is None when targets is None.
        """
        B, T = idx.shape
        assert T <= self.cfg.block_size, f"sequence length {T} exceeds block_size {self.cfg.block_size}"

        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.wte(idx) + self.wpe(pos))

        for block in self.blocks:
            x = block(x)
        x = self.norm(x)

        if targets is None:
            # Generating: only the last position's logits matter, so skip the
            # rest of the vocab_size projection entirely.
            return self.lm_head(x[:, [-1], :]), None

        logits = self.lm_head(x)
        # ignore_index drops masked positions from both the numerator and the
        # denominator, so the loss stays a per-supervised-token mean. Pretraining
        # passes no -100s, so this is a no-op there.
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1),
                               ignore_index=-100)
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, eot_token=None):
        """Sample continuations autoregressively. idx is (batch, seq_len)."""
        self.eval()
        finished = torch.zeros(idx.size(0), dtype=torch.bool, device=idx.device)
        for _ in range(max_new_tokens):
            # Crop to the last block_size tokens; positions beyond that are unlearned.
            idx_cond = idx[:, -self.cfg.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]

            if temperature == 0.0:
                # Greedy. The zero-temperature limit *is* the argmax, but
                # reaching it by dividing gives inf/nan, so take it directly.
                next_id = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float('-inf')
                probs = F.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)

            if eot_token is not None:
                # A row that already emitted EOT is done: pin it to EOT so its
                # sampled tokens never re-enter the context on later steps.
                next_id[finished] = eot_token
                finished |= next_id.squeeze(1) == eot_token

            idx = torch.cat((idx, next_id), dim=1)

            if finished.all():
                break
        return idx
if __name__ == "__main__":
    cfg = GPTConfig()
    model = GPT(cfg)
    print(f"Model size: {model.num_params() / 1e6:.2f}M params")