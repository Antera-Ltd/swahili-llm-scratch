import sentencepiece as spm

spm.SentencePieceTrainer.train(
    input="full_dataset.jsonl",
    model_prefix="swahili_tokenizer",
    model_type="BPE",
    vocab_size=1400,
    character_coverage=1.0,
    split_by_whitespace=True,
    split_by_number=True,
    max_sentence_length=4096,
    pad_id=-1,
    unk_id=0,
    bos_id=1,
    eos_id=2,
    pad_piece="<pad>",
    unk_piece="<unk>",
    bos_piece="<s>",
    eos_piece="</s>"
)

print("Tokenizer trained successfully: swahili_tokenizer.model / .vocab")