"""
Swahili-GPT
Inference Script
Author: Shadrackovsky
"""

import mlx.core as mx
import sentencepiece as spm
import json
import sys
import time
from train_scratch import KiswahiliLLM

COLORS = {
    'header': '\033[95m',
    'blue': '\033[94m',
    'cyan': '\033[96m',
    'green': '\033[92m',
    'yellow': '\033[93m',
    'red': '\033[91m',
    'bold': '\033[1m',
    'dim': '\033[2m',
    'end': '\033[0m'
}

MODEL_PATH = "swahili_llm_final.npz"
TOKENIZER_PATH = "swahili_tokenizer.model"
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.8       # set to Slightly higher = less repetition
TOP_K = 40              # Lower than 50 = more focused choices
# Optional: add TOP_P for bettter control
TOP_P = 0.9

ASCII_ART = r"""
███████╗██╗    ██╗ █████╗ ██╗  ██╗██╗██╗██╗
██╔════╝██║    ██║██╔══██╗██║  ██║██║██║██║
███████╗██║ █╗ ██║███████║███████║██║██║██║
╚════██║██║███╗██║██╔══██║██╔══██║██║██║██║
███████║╚███╔███╔╝██║  ██║██║  ██║██║██║██║
╚══════╝ ╚══╝╚══╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝╚═╝╚═╝
                                             
   ██████╗ ██████╗ ████████╗
  ██╔════╝ ██╔══██╗╚══██╔══╝
  ██║  ███╗██████╔╝   ██║   
  ██║   ██║██╔═══╝    ██║   
  ╚██████╔╝██║        ██║   
   ╚═════╝ ╚═╝        ╚═╝   
"""

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

        # Apply Top-K filtering
        if TOP_K > 0:
            sorted_indices = mx.argsort(logits, axis=-1)
            top_k_indices = sorted_indices[:, -TOP_K:]
            mask = mx.full(logits.shape, -1e9, dtype=logits.dtype)
            for b in range(logits.shape[0]):
                mask[b, top_k_indices[b]] = logits[b, top_k_indices[b]]
            logits = mask

        # Optional Top-P (nucleus) sampling to reduce loops further
        if TOP_P < 1.0:
            sorted_logits = mx.sort(logits, axis=-1)
            sorted_probs = mx.softmax(sorted_logits, axis=-1)
            cumulative = mx.cumsum(sorted_probs, axis=-1)
            mask_p = cumulative > TOP_P
            # Keep at least one token
            mask_p = mx.concatenate([mx.zeros_like(mask_p[:, :1]), mask_p[:, :-1]], axis=-1)
            logits = mx.where(mask_p, mx.array(-1e9, dtype=logits.dtype), logits)

        next_token = mx.random.categorical(logits, num_samples=1)
        input_ids = mx.concatenate([input_ids, next_token], axis=1)

        # Stop at end-of-sequence
        if next_token.item() == tokenizer.eos_id():
            break

    output_ids = input_ids[0].tolist()
    return tokenizer.decode(output_ids)

def typewriter_effect(text, delay=0.001):
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(delay)
    print()

def print_ascii_art():
    print(f"{COLORS['header']}{COLORS['bold']}{ASCII_ART}{COLORS['end']}")

def print_welcome():
    print_ascii_art()

def print_loading():
    print(f"{COLORS['dim']}Loading model and tokenizer...{COLORS['end']}")

def print_ready():
    print(f"{COLORS['green']}✓ Ready, Ask Swahili-GPT anything!{COLORS['end']}")
    print()

def print_prompt():
    return f"{COLORS['blue']}> {COLORS['end']}"

def print_response():
    print(f"{COLORS['green']}Swahili-GPT:{COLORS['end']}")

def print_exit():
    print(f"\n{COLORS['yellow']}Kwaheri!{COLORS['end']}")

def print_error(msg):
    print(f"{COLORS['red']}Error: {msg}{COLORS['end']}")

if __name__ == "__main__":
    print_welcome()
    
    print_loading()
    
    try:
        model = load_model()
        tokenizer = load_tokenizer()
        print_ready()
        
        while True:
            try:
                sys.stdout.write(print_prompt())
                sys.stdout.flush()
                prompt = input()
                
                if prompt.lower() == "exit":
                    print_exit()
                    break
                    
                if not prompt.strip():
                    continue
                
                print_response()
                result = generate_text(model, tokenizer, prompt.strip())
                # Remove unknown token markers
                result = result.replace("⁇", "").strip()
                typewriter_effect(result)
                print()
                
            except KeyboardInterrupt:
                print("\n")
                print_exit()
                break
            except Exception as e:
                print_error(str(e))
                
    except Exception as e:
        print_error(str(e))
        sys.exit(1)