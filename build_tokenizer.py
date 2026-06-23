"""
Train Tokenizer
Author: Shadrackovsky
"""

import sentencepiece as spm
import jsonlines


INPUT_DATA = "full_dataset.jsonl"
OUTPUT_PREFIX = "swahili_tokenizer"
VOCAB_SIZE = 3500
CHAR_COVERAGE = 0.9995


def extract_text(input_path, temp_txt="temp_text.txt"):
    with open(temp_txt, "w", encoding="utf-8") as out_f:
        with jsonlines.open(input_path) as reader:
            for item in reader:
                out_f.write(item["text"] + "\n")
    return temp_txt


if __name__ == "__main__":
    print("Extracting text for tokenizer training...")
    text_file = extract_text(INPUT_DATA)

    print(f"Training tokenizer with vocab size: {VOCAB_SIZE}")
    spm.SentencePieceTrainer.train(
        input=text_file,
        model_prefix=OUTPUT_PREFIX,
        vocab_size=VOCAB_SIZE,
        character_coverage=CHAR_COVERAGE,
        model_type="bpe",
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        pad_piece="<pad>",
        unk_piece="<unk>",
        bos_piece="<s>",
        eos_piece="</s>"
    )

    print(f"Tokenizer saved as {OUTPUT_PREFIX}.model / {OUTPUT_PREFIX}.vocab")