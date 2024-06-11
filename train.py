"""
References:
https://github.com/karpathy/nanoGPT/blob/master/train.py
"""


import os
import argparse
import time
import math
from contextlib import nullcontext

import numpy as np
import torch

from model import GPTConfig, GPT

argparser = argparse.ArgumentParser(description="Parameters for training the GPT model")
argparser.add_argument('--checkpoints_dir', type=str, default='checkpoints', help='Directory to save model checkpoints')
argparser.add_argument('--log_interval', type=int, default=10, help='How often to log training info')
argparser.add_argument('--eval_interval', type=int, default=250, help='How often to evaluate the model')
argparser.add_argument('--eval_iters', type=int, default=200, help='How many iters to use for evaluation')
argparser.add_argument('--eval_only', action='store_true', help='If True, script exits after the first evaluation')
argparser.add_argument('--always_save_checkpoint', action='store_true', help='If True, always save a checkpoint after each evaluation')
argparser.add_argument('--init_from', type=str, default='',
                       help='A cheakpoint to initialize the model from or a GPT2 model (gpt2, gpt2-medium, gpt2-large, gpt2-xl) to download and initialize from')
argparser.add_argument('--dataset', type=str, default='harrypotter', help='Dataset to use for training')
argparser.add_argument('--batch_size', type=int, default=4, help='Batch size for training')
argparser.add_argument('--block_size', type=int, default=1024, help='Block size for training')
argparser.add_argument('--n_layer', type=int, default=6, help='Number of layers in the model')
argparser.add_argument('--n_head', type=int, default=6, help='Number of heads in the model')
argparser.add_argument('--n_embd', type=int, default=384, help='Embedding size in the model')
argparser.add_argument('--dropout', type=float, default=0, help='Dropout rate in the model')
argparser.add_argument('--bias', action='store_true', help='If True, use bias inside LayerNorm and Linear layers')
argparser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate for the model')
argparser.add_argument('--max_iters', type=int, default=5000, help='Total number of training iterations')
argparser.add_argument('--weight_decay', type=float, default=1e-1, help='Weight decay for the optimizer')
argparser.add_argument('--beta1', type=float, default=0.9, help='Beta1 for the optimizer')
argparser.add_argument('--beta2', type=float, default=0.95, help='Beta2 for the optimizer')
argparser.add_argument('--grad_clip', type=float, default=1.0, help='Clip gradients at this value')
argparser.add_argument('--no_decay_lr', action='store_false', help='If True, would not decay the learning rate')
argparser.add_argument('--warmup_iters', type=int, default=100, help='Number of warmup iterations')
argparser.add_argument('--lr_decay_iters', type=int, default=5000, help='Number of learning rate decay iterations')
argparser.add_argument('--min_lr', type=float, default=1e-5, help='Minimum learning rate')
argparser.add_argument('--backend', type=str, default='nccl', help='Backend for DDP')
argparser.add_argument('--no_cuda', action='store_true', help='If True, use CPU for training')
argparser.add_argument('--dtype', type=str, default='bfloat16', help='Data type to use for training')
argparser.add_argument('--compile', action='store_true', help='If True, use PyTorch 2.0 to compile the model')
argparser.add_argument('--flops_promised', type=float, default=15e12, help='Promised flops for the model')
args = argparser.parse_args()



checkpoints_dir = args.checkpoints_dir
log_interval = args.log_interval
eval_interval = args.eval_interval
eval_iters = args.eval_iters
eval_only = args.eval_only
always_save_checkpoint = args.always_save_checkpoint
init_from = args.init_from
dataset = args.dataset
batch_size = args.batch_size
block_size = args.block_size
n_layer = args.n_layer
n_head = args.n_head
n_embd = args.n_embd
dropout = args.dropout
bias = args.bias
learning_rate = args.learning_rate
max_iters = args.max_iters
weight_decay = args.weight_decay
beta1 = args.beta1
beta2 = args.beta2
grad_clip = args.grad_clip
no_decay_lr = args.no_decay_lr
warmup_iters = args.warmup_iters
lr_decay_iters = args.lr_decay_iters
min_lr = args.min_lr
backend = args.backend
no_cuda = args.no_cuda
dtype = args.dtype
compile = args.compile
flops_promised = args.flops_promised


os.makedirs(checkpoints_dir, exist_ok=True)

torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn

device = 'cpu' if no_cuda else 'cuda' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device == 'cpu' else torch.amp.autocast(device_type=device, dtype=ptdtype)

data_dir = os.path.join('data', dataset)
if not os.path.exists(data_dir):
    raise FileNotFoundError(f"Data directory {data_dir} not found. Please create the dataset first.")

def get_batch(split):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# init these up here, can override if init_from='checkpointname' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9



# model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=50304, dropout=dropout) # start with model_args from command line
if init_from == '':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    model = GPT.from_pretrained(init_from, dropout)
    # read off the created config params, so we can store them into checkpoint correctly
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)
else:
    checkpoint_path = os.path.join(checkpoints_dir, init_from)
    print(f"Resuming training from {checkpoint_path}")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file {checkpoint_path} not found")
    # resume training from a checkpoint.
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    checkpoint_model_config = vars(checkpoint['config'])
    print(f"checkpoint model config: {checkpoint_model_config}")
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_config[k]
    # create the model
    model = GPT(checkpoint['config'])
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']

print(f"model config {model.config}")
tokens_per_iter = batch_size * model.config.block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size # so that the checkpoint will have the right value
model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device)
if init_from.endswith('.ckpt.pt'):
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None # free up memory

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# training loop
X, Y = get_batch('train') # fetch the very first batch
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model
running_mfu = -1.0
while True:

    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if no_decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, learning rate {lr:.4f}")
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': model.config,
                }
                save_path = os.path.join(checkpoints_dir, f'{dataset}.ckpt.pt')
                print(f"saving checkpoint to {save_path}")
                torch.save(checkpoint, save_path)
    if iter_num == 0 and eval_only:
        break

    # forward backward update, and using the GradScaler if data type is float16
    with ctx:
        logits, loss = model(X, Y)
    # immediately async prefetch next batch while model is doing the forward pass on the GPU
    X, Y = get_batch('train')
    # backward pass, with gradient scaling if training in fp16
    scaler.scale(loss).backward()
    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0:
        # get loss as float. note: this is a CPU-GPU sync point
        lossf = loss.item()
        if local_iter_num >= 5: # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(batch_size, dt, flops_promised)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break

