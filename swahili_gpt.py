"""
Swahili-GPT
Inference Script (PyTorch)
Author: Shadrackovsky
"""

import json
import sys
import time
import torch
import torch.nn.functional as F
import sentencepiece as spm
from train_scratch import KiswahiliLLM, DEVICE

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

MODEL_PATH = "swahili_llm_final.pt"
TOKENIZER_PATH = "swahili_tokenizer.model"
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.8       # Slightly higher = less repetition
TOP_K = 40              # Lower than 50 = more focused choices
TOP_P = 0.9             # Nucleus sampling

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
    model = KiswahiliLLM().to(DEVICE)
    state = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model


def load_tokenizer():
    sp = spm.SentencePieceProcessor()
    sp.load(TOKENIZER_PATH)
    return sp


@torch.no_grad()
def generate_text(model, tokenizer, prompt):
    tokens = tokenizer.encode(prompt, out_type=int)
    input_ids = torch.tensor(tokens, dtype=torch.long, device=DEVICE)[None, :]

    for _ in range(MAX_NEW_TOKENS):
        logits = model(input_ids)
        logits = logits[:, -1, :] / TEMPERATURE

        # Top-K filtering
        if TOP_K > 0:
            k = min(TOP_K, logits.shape[-1])
            kth_vals = torch.topk(logits, k, dim=-1).values[:, -1, None]
            logits = torch.where(
                logits < kth_vals, torch.full_like(logits, float("-inf")), logits
            )

        # Top-P (nucleus) filtering
        if TOP_P < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
            cumulative = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            remove = cumulative > TOP_P
            # Keep at least one token
            remove[:, 1:] = remove[:, :-1].clone()
            remove[:, 0] = False
            sorted_logits[remove] = float("-inf")
            logits = sorted_logits.gather(-1, sorted_idx.argsort(-1))

        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        input_ids = torch.cat([input_ids, next_token], dim=1)

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
