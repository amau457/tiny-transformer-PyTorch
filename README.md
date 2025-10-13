# tiny-transformer-PyTorch
building a tiny transformer in Pytorch and finetuning it.

Files description:
transformer_causal_charlevel.py -> train and save a causal transformer using character level tokenization
load_data.py -> load a data.txt file as a dataset
transformer_causal_ByteBpe.py -> train and save a causal transformer using Byte level BPE tokenization
train_tokenizer_bpe.py -> train the byte level BPE tokenizer and saves it (needed to be executed once for using transformer_causal_ByteBpe.py)
finetune.py -> coming soon 


objectives:
The idea is to train on a large french dataset a model to generate 'ok' french sentences. 
Then finetune the model on a smaller dataset containing sentences used by a character in order to mimic them.

The dataset files are not included in this repo.
