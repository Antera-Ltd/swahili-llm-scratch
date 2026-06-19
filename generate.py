import mlx.core as mx
import json
from sentencepiece import SentencePieceProcessor
from train_scratch import SwahiliLLM

# Load
with open("model_config.json") as f:
    config = json.load(f)
tokenizer = SentencePieceProcessor(model_file="swahili_tokenizer.model")
model = SwahiliLLM(config)
model.load_weights("swahili_llm_final.npz")

# Generate
def generate(prompt, max_tokens=200):
    tokens = mx.array([tokenizer.encode(prompt)])
    for _ in range(max_tokens):
        logits = model(tokens)
        next_token = mx.argmax(logits[:, -1], axis=-1, keepdims=True)
        tokens = mx.concatenate([tokens, next_token], axis=-1)
    return tokenizer.decode(tokens[0].tolist())

print(generate("Eleza jinsi ya kusoma vizuri:"))
print(generate("Explain why education is important in Tanzania:"))
print(generate("Habari yako? I want to know jinsi ya kuanzisha biashara ndogo:"))