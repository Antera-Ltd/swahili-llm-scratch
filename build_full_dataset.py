"""
Build Full Dataset
Synthetic data: 30%
Swahili Corpus, Crawled:70%
Author: Shadrackovsky
"""

import jsonlines
import os
import random
from tqdm import tqdm
from synthesize_all import (
    generate_swahili_sentence,
    generate_english_sentence,
    generate_kiswaenglish,
    generate_reasoning_text
)


SYNTHETIC_RATIO = 0.3
REAL_RATIO = 0.7
OUTPUT_FILE = "full_dataset.jsonl"
CORPUS_DIR = "./Swahili_Corpus"
CRAWLED_DIR = "./Crawled_Data"


def load_txt_files(folder):
    samples = []
    if not os.path.isdir(folder):
        print(f"Warning: Folder {folder} not found")
        return samples
    for fname in os.listdir(folder):
        if fname.endswith(".txt"):
            path = os.path.join(folder, fname)
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read().strip()
                    chunks = [text[i:i+1000] for i in range(0, len(text), 800)]
                    for chunk in chunks:
                        if len(chunk) > 80:
                            samples.append({"text": chunk})
            except Exception as e:
                print(f"Could not read {fname}: {e}")
    return samples


if __name__ == "__main__":
    print("Generating synthetic data")
    SYNTHETIC_SAMPLES = 60000
    synthetic_data = []

    for _ in tqdm(range(SYNTHETIC_SAMPLES)):
        choice = random.choices(
            ["sw", "en", "mix", "reasoning"], weights=[0.2, 0.2, 0.4, 0.2]
        )[0]
        if choice == "sw":
            text = " ".join([generate_swahili_sentence() for _ in range(random.randint(3,6))])
        elif choice == "en":
            text = " ".join([generate_english_sentence() for _ in range(random.randint(3,6))])
        elif choice == "mix":
            text = " ".join([generate_kiswaenglish() for _ in range(random.randint(3,6))])
        else:
            text = generate_reasoning_text()
        synthetic_data.append({"text": text})

    print("\nLoading Corpus, Crawled data.")
    real_data = []
    real_data.extend(load_txt_files(CORPUS_DIR))
    real_data.extend(load_txt_files(CRAWLED_DIR))

    print(f"Loaded {len(real_data)} real samples")

    target_real = int(len(synthetic_data) * (REAL_RATIO / SYNTHETIC_RATIO))
    if len(real_data) > target_real:
        real_data = random.sample(real_data, target_real)
    elif len(real_data) < target_real:
        print(f"Note: Only {len(real_data)} real samples available, using all")

    final_data = synthetic_data + real_data
    random.shuffle(final_data)

    with jsonlines.open(OUTPUT_FILE, mode="w") as writer:
        writer.write_all(final_data)

    print("\nDataset is ready.")
    print(f"Total samples: {len(final_data)}")
    print(f"Synthetic: {len(synthetic_data)} ({len(synthetic_data)/len(final_data)*100:.1f}%)")
    print(f"Corpus: {len(real_data)} ({len(real_data)/len(final_data)*100:.1f}%)")