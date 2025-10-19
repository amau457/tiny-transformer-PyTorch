from tokenizers import ByteLevelBPETokenizer
import os

tokenizer = ByteLevelBPETokenizer()
tokenizer.train(files=["data_clean.txt"], vocab_size=30_000, min_frequency=2,
                special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"])
out_dir = "tokenizer_bpe"
os.makedirs(out_dir, exist_ok=True)
tokenizer.save_model("tokenizer_bpe")