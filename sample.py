"""
Sample from a trained model
References:
https://github.com/karpathy/nanoGPT/blob/master/sample.py
"""
import os
import argparse
from contextlib import nullcontext
import torch
import tiktoken
from model import GPTConfig, GPT
import time

argparser = argparse.ArgumentParser()
argparser.add_argument('--init_from', type=str, default='harrypotter.ckpt.pt', help='checkpoint for the model to sample from')
argparser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='checkpoint directory for the model')
argparser.add_argument('--samples_dir', type=str, default='samples', help='directory to save samples')
argparser.add_argument('--num_samples', type=int, default=10, help='number of samples to draw')
argparser.add_argument('--max_new_tokens', type=int, default=500, help='number of tokens generated in each sample')
argparser.add_argument('--temperature', type=float, default=0.8, help='temperature for sampling')
argparser.add_argument('--top_k', type=int, default=200, help='top k for sampling')
argparser.add_argument('--seed', type=int, default=None, help='random seed')
argparser.add_argument('--no_cuda', action='store_true', help='do not use cuda')
argparser.add_argument('--dtype', type=str, default='float32', help='data type for model')
argparser.add_argument('--compile', action='store_true', help='compile the model')
argparser.add_argument('--start', type=str, default='\n', help='starting prompt')

args = argparser.parse_args()

init_from = os.path.join(args.checkpoint_dir, args.init_from)
save_samples_path = os.path.join(args.samples_dir, f"{args.init_from.split('.')[0]}.{time.strftime('%Y.%m.%d.%H.%M.%S')}.txt")
start = args.start
num_samples = args.num_samples
max_new_tokens = args.max_new_tokens
temperature = args.temperature
top_k = args.top_k
seed = torch.randint(0, 1000, (1,)).item() if args.seed is None else args.seed
device = 'cuda' if not args.no_cuda and torch.cuda.is_available() else 'cpu'
dtype = args.dtype
compile = args.compile


torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device == 'cpu' else torch.amp.autocast(device_type=device, dtype=ptdtype)

# model
# init from a model saved in a specific directory
if not os.path.exists(init_from):
    raise FileNotFoundError(f"Can't find model file at {init_from}")
checkpoint = torch.load(init_from, map_location=device)
model = GPT(checkpoint['config'])
state_dict = checkpoint['model']
unwanted_prefix = '_orig_mod.'
for k,v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict)

model.eval()
model.to(device)
if compile:
    model = torch.compile(model) # requires PyTorch 2.0 (optional)

print(f"model config: {model.config}")
print(f"number of iterations: {checkpoint['iter_num']}")
print(f"best validation loss: {checkpoint['best_val_loss']:.4f}")

enc = tiktoken.get_encoding("gpt2")
encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
decode = lambda l: enc.decode(l)

# encode the beginning of the prompt
start_ids = encode(start)
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

os.makedirs(args.samples_dir, exist_ok=True)

# run generation
with open(save_samples_path, 'w') as f:
    with torch.no_grad():
        with ctx:
            for k in range(num_samples):
                print('sample', k)
                y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
                print(decode(y[0].tolist()))
                print('---------------')
                f.write(f"Sample {k}\n")
                f.write(decode(y[0].tolist()))
                f.write('\n---------------\n\n')
