"""
Swahili-GPT
Text Generation Script
Author: Shadrackovsky
"""

import mlx.core as mx
import sentencepiece as spm
import json
import sys
import time
from train_scratch import KiswahiliLLM

# Default theme (dark mode)
CURRENT_THEME = 'dark'

# Theme configurations
THEMES = {
    'dark': {
        'header': '\033[95m',
        'blue': '\033[94m',
        'cyan': '\033[96m',
        'green': '\033[92m',
        'yellow': '\033[93m',
        'red': '\033[91m',
        'bold': '\033[1m',
        'dim': '\033[2m',
        'end': '\033[0m'
    },
    'light': {
        'header': '\033[95m',
        'blue': '\033[34m',
        'cyan': '\033[36m',
        'green': '\033[32m',
        'yellow': '\033[33m',
        'red': '\033[31m',
        'bold': '\033[1m',
        'dim': '\033[2m',
        'end': '\033[0m'
    },
    'dark_colorblind': {
        'header': '\033[95m',
        'blue': '\033[94m',
        'cyan': '\033[96m',
        'green': '\033[92m',
        'yellow': '\033[93m',
        'red': '\033[91m',
        'bold': '\033[1m',
        'dim': '\033[2m',
        'end': '\033[0m'
    },
    'light_colorblind': {
        'header': '\033[95m',
        'blue': '\033[34m',
        'cyan': '\033[36m',
        'green': '\033[32m',
        'yellow': '\033[33m',
        'red': '\033[31m',
        'bold': '\033[1m',
        'dim': '\033[2m',
        'end': '\033[0m'
    },
    'dark_ansi': {
        'header': '\033[95m',
        'blue': '\033[94m',
        'cyan': '\033[96m',
        'green': '\033[92m',
        'yellow': '\033[93m',
        'red': '\033[91m',
        'bold': '\033[1m',
        'dim': '\033[2m',
        'end': '\033[0m'
    },
    'light_ansi': {
        'header': '\033[95m',
        'blue': '\033[34m',
        'cyan': '\033[36m',
        'green': '\033[32m',
        'yellow': '\033[33m',
        'red': '\033[31m',
        'bold': '\033[1m',
        'dim': '\033[2m',
        'end': '\033[0m'
    }
}

MODEL_PATH = "swahili_llm_final.npz"
TOKENIZER_PATH = "swahili_tokenizer.model"
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.7
TOP_K = 50  

def get_colors():
    return THEMES.get(CURRENT_THEME, THEMES['dark'])

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

        if TOP_K > 0:
            indices = mx.argpartition(logits, -TOP_K, axis=-1)[:, -TOP_K:]
            mask = mx.full_like(logits, -1e9)
            logits = mask.at[mx.arange(logits.shape[0])[:, None], indices].set(
                logits.at[mx.arange(logits.shape[0])[:, None], indices]
            )

        next_token = mx.random.categorical(logits, num_samples=1)
        input_ids = mx.concatenate([input_ids, next_token], axis=1)

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

def show_theme_menu():
    colors = get_colors()
    print(f"{colors['bold']}Choose the style that looks best with you{colors['end']}")
    print()
    print("To change this later, run /theme")
    print()
    print(f"{colors['blue']}> 1. Dark mode{colors['end']}")
    print("  2. Light mode")
    print("  3. Dark mode (colorblind-friendly)")
    print("  4. Light mode (colorblind-friendly)")
    print("  5. Dark mode (ANSI colors only)")
    print("  6. Light mode (ANSI colors only)")
    print()
    print(f"{colors['dim']}Enter your choice (1-6) or press Enter for default:{colors['end']}")

def set_theme(choice):
    global CURRENT_THEME
    theme_map = {
        '1': 'dark',
        '2': 'light',
        '3': 'dark_colorblind',
        '4': 'light_colorblind',
        '5': 'dark_ansi',
        '6': 'light_ansi'
    }
    if choice in theme_map:
        CURRENT_THEME = theme_map[choice]
        return True
    return False

def get_theme_choice():
    colors = get_colors()
    choice = input(f"{colors['blue']}> {colors['end']}")
    return choice

def print_welcome():
    colors = get_colors()
    print(f"{colors['bold']}Karibu Swahili-GPT {colors['end']}")
    print()
    show_theme_menu()

def print_loading():
    colors = get_colors()
    print(f"{colors['dim']}Loading model and tokenizer...{colors['end']}")

def print_ready():
    colors = get_colors()
    print(f"{colors['green']}✓ Ready{colors['end']}")
    print()

def print_prompt():
    colors = get_colors()
    return f"{colors['blue']}> {colors['end']}"

def print_response():
    colors = get_colors()
    print(f"{colors['green']}Swahili-GPT:{colors['end']}")

def print_exit():
    colors = get_colors()
    print(f"\n{colors['yellow']}Kwaheri!{colors['end']}")

def print_error(msg):
    colors = get_colors()
    print(f"{colors['red']}Error: {msg}{colors['end']}")

def handle_theme_command():
    global CURRENT_THEME
    print()
    show_theme_menu()
    choice = get_theme_choice()
    if choice.strip():
        if set_theme(choice.strip()):
            colors = get_colors()
            print(f"{colors['green']}Theme updated!{colors['end']}")
            print()
        else:
            colors = get_colors()
            print(f"{colors['red']}Invalid choice. Keeping current theme.{colors['end']}")
            print()

if __name__ == "__main__":
    print_welcome()
    theme_choice = get_theme_choice()
    if theme_choice.strip():
        set_theme(theme_choice.strip())
    
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
                
                if prompt.lower() == "/theme":
                    handle_theme_command()
                    continue
                
                if prompt.lower() == "exit":
                    print_exit()
                    break
                    
                if not prompt.strip():
                    continue
                
                print_response()
                result = generate_text(model, tokenizer, prompt.strip())
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