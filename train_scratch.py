import mlx.core as mx
import mlx.nn as nn
from mlx.optimizers import AdamW
from mlx.utils import tree_flatten
from datasets import load_dataset
from sentencepiece import SentencePieceProcessor
import json
from tqdm import tqdm

# --------------------------
# LOAD CONFIG & TOKENIZER
# --------------------------
with open("model_config.json") as f:
    config = json.load(f)
tokenizer = SentencePieceProcessor(model_file="swahili_tokenizer.model")

# --------------------------
# DEFINE MODEL
# --------------------------
class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = nn.MultiHeadAttention(
            config["hidden_size"],
            config["num_attention_heads"]
        )
        self.mlp = nn.Sequential(
            nn.Linear(config["hidden_size"], config["intermediate_size"]),
            nn.GELU(),
            nn.Linear(config["intermediate_size"], config["hidden_size"])
        )
        self.norm1 = nn.RMSNorm(config["hidden_size"], eps=config["rms_norm_eps"])
        self.norm2 = nn.RMSNorm(config["hidden_size"], eps=config["rms_norm_eps"])

    def __call__(self, x, mask):
        # call: MultiHeadAttention(q, k, v, mask) all same for self-attention
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x), mask)
        x = x + self.mlp(self.norm2(x))
        return x

class SwahiliLLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed = nn.Embedding(config["vocab_size"], config["hidden_size"])
        self.pos_embed = nn.Embedding(config["max_position_embeddings"], config["hidden_size"])
        self.layers = [TransformerBlock(config) for _ in range(config["num_hidden_layers"])]
        self.norm = nn.RMSNorm(config["hidden_size"], eps=config["rms_norm_eps"])
        self.head = nn.Linear(config["hidden_size"], config["vocab_size"])

    def __call__(self, input_ids):
        batch_size, seq_len = input_ids.shape
        pos = mx.arange(seq_len)[None]
        x = self.embed(input_ids) + self.pos_embed(pos)
        # Mask exactly matches current sequence length
        mask = mx.tri(seq_len, seq_len)
        mask = mask.reshape(1, 1, seq_len, seq_len)
        mask = mask.astype(mx.bool_)
        for layer in self.layers:
            x = layer(x, mask)
        x = self.norm(x)
        return self.head(x)

# --------------------------
# PREPARE DATA
# --------------------------
def tokenize_function(examples):
    tokens = [tokenizer.encode(text) for text in examples["text"]]
    return {"input_ids": tokens}

dataset = load_dataset("json", data_files="full_dataset.jsonl", split="train")
dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# --------------------------
# INITIALIZE MODEL
# --------------------------
model = SwahiliLLM(config)
mx.eval(model.parameters())

optimizer = AdamW(learning_rate=3e-4, weight_decay=0.01)

# --------------------------
# TRAINING LOOP
# --------------------------
BATCH_SIZE = 8
SEQ_LEN = 512
EPOCHS = 15

def loss_fn(model, input_ids):
    x = input_ids[:, :-1]
    y = input_ids[:, 1:]
    logits = model(x)
    return nn.losses.cross_entropy(logits, y, reduction="mean")

loss_and_grad = nn.value_and_grad(model, loss_fn)

print("Training from scratch started...")
for epoch in range(EPOCHS):
    total_loss = 0
    for i in tqdm(range(0, len(dataset), BATCH_SIZE)):
        batch = dataset[i:i+BATCH_SIZE]["input_ids"]
        batch_padded = []
        for seq in batch:
            if len(seq) >= SEQ_LEN:
                batch_padded.append(seq[:SEQ_LEN])
            else:
                batch_padded.append(seq + [tokenizer.pad_id()] * (SEQ_LEN - len(seq)))
        batch = mx.array(batch_padded)
        loss, grads = loss_and_grad(model, batch)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(dataset):.4f}")
    model.save_weights(f"model_epoch_{epoch+1}.npz")

model.save_weights("swahili_llm_final.npz")
print("Training complete! Model saved.")