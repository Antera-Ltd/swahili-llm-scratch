"""
Language Model Training
Author: Shadrackovsky
"""

import os
import json
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from tqdm import tqdm
import sentencepiece as spm

with open("model_config.json", "r", encoding="utf-8") as f:
    config = json.load(f)

VOCAB_SIZE = config["vocab_size"]
NUM_LAYERS = config["num_layers"]
NUM_HEADS = config["num_attention_heads"]
HIDDEN_SIZE = config["hidden_size"]
INTERMEDIATE_SIZE = config["intermediate_size"]
MAX_SEQ_LEN = config["max_seq_len"]
DROPOUT = config["dropout"]
LN_EPS = config["layer_norm_epsilon"]


class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.MultiHeadAttention(dims=HIDDEN_SIZE, num_heads=NUM_HEADS)
        self.norm1 = nn.LayerNorm(dims=HIDDEN_SIZE, eps=LN_EPS)
        self.norm2 = nn.LayerNorm(dims=HIDDEN_SIZE, eps=LN_EPS)
        self.ffn = nn.Sequential(
            nn.Linear(HIDDEN_SIZE, INTERMEDIATE_SIZE),
            nn.GELU(),
            nn.Linear(INTERMEDIATE_SIZE, HIDDEN_SIZE),
            nn.Dropout(DROPOUT)
        )
        self.dropout = nn.Dropout(DROPOUT)

    def __call__(self, x, mask=None):
        attn_out = self.attn(self.norm1(x), mask=mask)
        x = x + self.dropout(attn_out)
        ffn_out = self.ffn(self.norm2(x))
        x = x + self.dropout(ffn_out)
        return x


class KiswahiliLLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, HIDDEN_SIZE)
        self.pos_embedding = nn.Embedding(MAX_SEQ_LEN, HIDDEN_SIZE)
        self.layers = [TransformerBlock() for _ in range(NUM_LAYERS)]
        self.norm = nn.LayerNorm(dims=HIDDEN_SIZE, eps=LN_EPS)
        self.output = nn.Linear(HIDDEN_SIZE, VOCAB_SIZE)

    def __call__(self, input_ids):
        seq_len = input_ids.shape[1]
        positions = mx.arange(seq_len)[None, :]

        x = self.embedding(input_ids) + self.pos_embedding(positions)
        mask = nn.MultiHeadAttention.create_causal_mask(seq_len)

        for layer in self.layers:
            x = layer(x, mask=mask)

        x = self.norm(x)
        logits = self.output(x)
        return logits


def load_tokenizer():
    sp = spm.SentencePieceProcessor()
    sp.load("swahili_tokenizer.model")
    return sp


if __name__ == "__main__":
    print("Initializing model..")
    model = KiswahiliLLM()
    mx.eval(model.parameters())

    optimizer = optim.AdamW(learning_rate=3e-4, weight_decay=0.01)
    loss_fn = nn.losses.cross_entropy

    tokenizer = load_tokenizer()
    batch_size = 8
    epochs = 8

    print("Starting training process..")
    for epoch in range(epochs):
        total_loss = 0.0
        progress = tqdm(range(1000), desc=f"Epoch {epoch+1}/{epochs}")

        for step in progress:
            inputs = mx.random.randint(0, VOCAB_SIZE, (batch_size, 128))
            targets = mx.concatenate([inputs[:, 1:], mx.zeros((batch_size, 1), dtype=mx.int32)], axis=1)

            def loss_and_grad(model, x, y):
                logits = model(x)
                return mx.mean(loss_fn(logits.reshape(-1, VOCAB_SIZE), y.reshape(-1)))

            loss, grads = mx.value_and_grad(loss_and_grad)(model, inputs, targets)
            model.update(optimizer.apply_gradients(grads, model))
            mx.eval(model.parameters(), optimizer.state)

            total_loss += loss.item()
            progress.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / 1000
        print(f"Epoch {epoch+1} complete | Average Loss: {avg_loss:.4f}")

        save_path = f"model_epoch_{epoch+1:02d}.npz"
        model.save_weights(save_path)
        print(f"Model saved to {save_path}")

    model.save_weights("swahili_llm_final.npz")
    print("Training finished successfully. Final model saved.")