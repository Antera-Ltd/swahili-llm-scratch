from sentencepiece import SentencePieceTrainer, SentencePieceProcessor

# Train tokenizer ONLY
SentencePieceTrainer.train(
    input="full_dataset.jsonl",
    model_prefix="swahili_tokenizer",
    vocab_size=32000, 
    character_coverage=1.0,
    model_type="bpe",
    split_by_unicode_script=True,
    allow_whitespace_only_pieces=True
)

print(" Custom tokenizer built from your data")