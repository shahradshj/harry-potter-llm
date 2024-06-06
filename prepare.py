import os
import tiktoken
import numpy as np
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument('--input_file_path', type=str, default='./data/harrypotter/input.txt')
args = argparser.parse_args()
input_file_path = args.input_file_path

with open(input_file_path, 'r', encoding='utf-8') as f:
    data = f.read()
n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

# encode with tiktoken gpt2 bpe
enc = tiktoken.get_encoding("gpt2")
train_ids = enc.encode_ordinary(train_data)
val_ids = enc.encode_ordinary(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(input_file_path), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(input_file_path), 'val.bin'))

# train.bin has 1,285,345 tokens
# val.bin has 153,520 tokens
