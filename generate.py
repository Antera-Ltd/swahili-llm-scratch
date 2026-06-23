"""
Text Generation Script
Author: Shadrackovsky
"""

import mlx.core as mx
import sentencepiece as spm
import json
from train_scratch import KiswahiliLLM

MODEL_PATH = "swahili_llm_final.npz"
TOKENIZER_PATH = "swahili_tokenizer.model"
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.7


def load_model():
    with open("model_config.json", "r", encoding="utf-8") as f:
        config = json.load(f)
    model = KiswahiliLLM()
    model.load_weights(MODEL_PATH)
    mx.eval(model.parameters())
    return model


def load_tokenizer():
    sp = spm.SentencePieceProcessor()
    sp.load(TOKENIZER_PATH)
    return sp


def generate_text(model, tokenizer, prompt):
    tokens = tokenizer.encode(prompt, out_type=int)
    input_ids = mx.array(tokens, dtype=mx.int32)[None, :]

    for _ in range(MAX_NEW_TOKENS):
        logits = model(input_ids)
        logits = logits[:, -1, :] / TEMPERATURE
        next_token = mx.random.categorical(logits, num_samples=1)
        input_ids = mx.concatenate([input_ids, next_token], axis=1)

        if next_token.item() == tokenizer.eos_id():
            break

    output_ids = input_ids[0].tolist()
    return tokenizer.decode(output_ids)


if __name__ == "__main__":
    print("Loading model and tokenizer...")
    model = load_model()
    tokenizer = load_tokenizer()

    print("Model ready. Type your prompt below (type 'exit' to quit):")
    while True:
        prompt = input("> ")
        if prompt.lower() == "exit":
            break
        if not prompt.strip():
            continue
        result = generate_text(model, tokenizer, prompt.strip())
        print(f"\n{result}\n")