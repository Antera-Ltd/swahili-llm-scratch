"""
Model definition for the PyTorch Swahili LLM — a GPT-style decoder-only
Transformer. Shared by train.py and generate.py.

Architecture is read from model_config.json so a single file controls the size.
"""
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(HERE, "model_config.json")


def load_config(path=CONFIG_PATH):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def create_causal_mask(seq_len, device):
    mask = torch.full((seq_len, seq_len), float("-inf"), device=device)
    return torch.triu(mask, diagonal=1)


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        h, eps = cfg["hidden_size"], cfg["layer_norm_epsilon"]
        self.attn = nn.MultiheadAttention(
            embed_dim=h, num_heads=cfg["num_attention_heads"],
            dropout=cfg["dropout"], batch_first=True,
        )
        self.norm1 = nn.LayerNorm(h, eps=eps)
        self.norm2 = nn.LayerNorm(h, eps=eps)
        self.ffn = nn.Sequential(
            nn.Linear(h, cfg["intermediate_size"]), nn.GELU(),
            nn.Linear(cfg["intermediate_size"], h), nn.Dropout(cfg["dropout"]),
        )
        self.dropout = nn.Dropout(cfg["dropout"])

    def forward(self, x, mask=None):
        n = self.norm1(x)
        a, _ = self.attn(n, n, n, attn_mask=mask, need_weights=False)
        x = x + self.dropout(a)
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


class KiswahiliLLM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        h = cfg["hidden_size"]
        self.embedding = nn.Embedding(cfg["vocab_size"], h)
        self.pos_embedding = nn.Embedding(cfg["max_seq_len"], h)
        self.layers = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg["num_layers"])])
        self.norm = nn.LayerNorm(h, eps=cfg["layer_norm_epsilon"])
        self.output = nn.Linear(h, cfg["vocab_size"])

    def forward(self, input_ids):
        T = input_ids.shape[1]
        pos = torch.arange(T, device=input_ids.device)[None, :]
        x = self.embedding(input_ids) + self.pos_embedding(pos)
        mask = create_causal_mask(T, input_ids.device)
        for layer in self.layers:
            x = layer(x, mask=mask)
        return self.output(self.norm(x))


def load_model(weights_path, cfg=None, device="cpu"):
    """Load model weights from a plain state_dict or a training checkpoint dict."""
    cfg = cfg or load_config()
    state = torch.load(weights_path, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    state = {k: (v.float() if torch.is_floating_point(v) else v) for k, v in state.items()}
    model = KiswahiliLLM(cfg).to(device)
    model.load_state_dict(state)
    model.eval()
    return model


@torch.no_grad()
def generate(model, sp, prompt, max_new_tokens=80, temperature=0.8,
             top_k=40, top_p=0.9, device="cpu"):
    ids = torch.tensor(sp.encode(prompt), dtype=torch.long, device=device)[None, :]
    for _ in range(max_new_tokens):
        logits = model(ids)[:, -1, :] / temperature
        if top_k:
            kth = torch.topk(logits, min(top_k, logits.shape[-1])).values[:, -1, None]
            logits = torch.where(logits < kth, torch.full_like(logits, float("-inf")), logits)
        if top_p < 1.0:
            sl, si = torch.sort(logits, descending=True)
            cum = torch.cumsum(F.softmax(sl, dim=-1), dim=-1)
            rm = cum > top_p
            rm[:, 1:] = rm[:, :-1].clone(); rm[:, 0] = False
            sl[rm] = float("-inf")
            logits = sl.gather(-1, si.argsort(-1))
        nxt = torch.multinomial(F.softmax(logits, dim=-1), 1)
        ids = torch.cat([ids, nxt], dim=1)
        if nxt.item() == sp.eos_id():
            break
    return sp.decode(ids[0].tolist()).replace("⁇", "")
