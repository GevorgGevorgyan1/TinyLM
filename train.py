"""Train the GPT on a prepared corpus.

    python train.py                      # DATASET below
    python train.py --dataset tinystories

Data and checkpoints are namespaced per dataset, so runs on different corpora
never collide: data/<dataset>/ in, out/<dataset>/ out.
"""

import argparse
import json
import math
import os
import time
from contextlib import nullcontext

import numpy as np
import torch

from model import GPT, GPTConfig

# --- data / io -------------------------------------------------------------
DATASET = 'fineweb-edu'          # must match a `python prepare.py <name>` run
EVAL_INTERVAL = 500
EVAL_ITERS = 100
LOG_INTERVAL = 10
ALWAYS_SAVE = False              # save every eval, not just on val improvement
RESUME = False                   # continue from out/<dataset>/ckpt.pt

# --- logging ---------------------------------------------------------------
TENSORBOARD = True               # tensorboard --logdir out/<dataset>/tb
RUN_NAME = ''                    # '' -> timestamp; reuse a name to extend that run's curves

# --- optimisation ----------------------------------------------------------
BLOCK_SIZE = 1024                # context length; ignored on resume (ckpt wins)
BATCH_SIZE = 16                  # sequences per micro-step
GRAD_ACCUM_STEPS = 16            # micro-steps before an optimiser step
MAX_ITERS = 20000                # 262k tokens/iter -> ~5.2B tokens, ~0.55 epochs
LEARNING_RATE = 6e-4             # fresh run, so a real cosine decay rather than a flat floor
MIN_LR = 6e-5
WARMUP_ITERS = 500
LR_DECAY_ITERS = MAX_ITERS       # cosine floor lands at the end of training
WEIGHT_DECAY = 0.1
BETAS = (0.9, 0.95)
GRAD_CLIP = 1.0

ap = argparse.ArgumentParser()
ap.add_argument('--dataset', default=DATASET, help='name under data/')
ap.add_argument('--resume', action='store_true', default=RESUME)
_args = ap.parse_args()
DATASET, RESUME = _args.dataset, _args.resume
DATA_DIR = os.path.join('data', DATASET)
OUT_DIR = os.path.join('out', DATASET)

# --- system ----------------------------------------------------------------
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DTYPE = ('bfloat16' if DEVICE == 'cuda' and torch.cuda.is_bf16_supported()
         else 'float16' if DEVICE == 'cuda' else 'float32')
print(f'using device={DEVICE} dtype={DTYPE} ')

COMPILE = True
FLOPS_PROMISED = 71e12           # RTX 3090 bf16 dense, fp32 accumulate (GA102).


device_type = 'cuda' if DEVICE.startswith('cuda') else 'cpu'
torch.manual_seed(1337)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
ctx = (nullcontext() if device_type == 'cpu' else
       torch.amp.autocast(device_type='cuda', dtype=getattr(torch, DTYPE)))

with open(os.path.join(DATA_DIR, 'meta.json')) as f:
    meta = json.load(f)

# Resolve the config before anything reads block_size — on resume it must come
# from the checkpoint, not from whatever the defaults happen to say now.
iter_num = 0
best_val_loss = float('inf')
ckpt = None
if RESUME:
    ckpt = torch.load(os.path.join(OUT_DIR, 'ckpt.pt'), map_location=DEVICE)
    cfg = GPTConfig(**ckpt['config'])
    cfg.dropout = 0.1
    iter_num = ckpt['iter_num']
    best_val_loss = ckpt['best_val_loss']
else:
    cfg = GPTConfig(block_size=BLOCK_SIZE)

tokens_per_iter = GRAD_ACCUM_STEPS * BATCH_SIZE * cfg.block_size
os.makedirs(OUT_DIR, exist_ok=True)
print(f'dataset: {DATASET} | {meta["splits"]["train"]["tokens"]:,} train tokens '
      f'across {len(meta["splits"]["train"]["shards"])} shards')
print(f'tokens/iter: {tokens_per_iter:,} | total: {tokens_per_iter * MAX_ITERS / 1e9:.2f}B '
      f'({tokens_per_iter * MAX_ITERS / meta["splits"]["train"]["tokens"]:.2f} epochs)')


# --- metrics ---------------------------------------------------------------
writer = None
if TENSORBOARD:
    from torch.utils.tensorboard import SummaryWriter

    log_dir = os.path.join(OUT_DIR, 'tb', RUN_NAME or time.strftime('%Y%m%d-%H%M%S'))
    writer = SummaryWriter(log_dir, flush_secs=30)
    hparams = {
        'batch_size': BATCH_SIZE, 'grad_accum_steps': GRAD_ACCUM_STEPS,
        'tokens_per_iter': tokens_per_iter,
        'max_iters': MAX_ITERS, 'learning_rate': LEARNING_RATE, 'min_lr': MIN_LR,
        'warmup_iters': WARMUP_ITERS, 'lr_decay_iters': LR_DECAY_ITERS,
        'weight_decay': WEIGHT_DECAY, 'grad_clip': GRAD_CLIP, 'dtype': DTYPE,
        **cfg.__dict__,
    }
    # A markdown table in the TEXT tab — add_hparams wants a metric to rank runs
    # by, which we do not have until the run is over.
    writer.add_text('config', '\n'.join(
        ['|param|value|', '|---|---|'] + [f'|{k}|{v}|' for k, v in hparams.items()]))
    print(f'tensorboard --logdir {os.path.join(OUT_DIR, "tb")}')


# --- data ------------------------------------------------------------------
# Weight shard choice by token count so a short tail shard is not over-represented.
SHARDS = {s: meta['splits'][s]['shards'] for s in ('train', 'val')}
SHARD_P = {s: np.array([sh['tokens'] for sh in SHARDS[s]], dtype=np.float64)
              / sum(sh['tokens'] for sh in SHARDS[s]) for s in SHARDS}
rng = np.random.default_rng(1337)


def get_batch(split):
    # One shard per micro-batch — the alternative is BATCH_SIZE memmaps per step,
    # and at 100M tokens a shard is far larger than anything a batch correlates over.
    shards = SHARDS[split]
    shard = shards[0] if len(shards) == 1 else shards[rng.choice(len(shards), p=SHARD_P[split])]
    # Reopened every call: keeping a live memmap across a long run leaks the
    # page cache into RSS. This is cheap, the OS keeps the pages warm.
    data = np.memmap(os.path.join(DATA_DIR, shard['file']),
                     dtype=np.dtype(meta['dtype']), mode='r')
    ix = torch.randint(len(data) - cfg.block_size - 1, (BATCH_SIZE,))
    x  = torch.stack([torch.from_numpy(data[i:i + cfg.block_size].astype(np.int64)) for i in ix])
    y  = torch.stack([torch.from_numpy(data[i + 1:i + 1 + cfg.block_size].astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        x = x.pin_memory().to(DEVICE, non_blocking=True)
        y = y.pin_memory().to(DEVICE, non_blocking=True)
    else:
        x, y = x.to(DEVICE), y.to(DEVICE)
    return x, y


# --- model -----------------------------------------------------------------
model = GPT(cfg)
if RESUME:
    model.load_state_dict(ckpt['model'])
model.to(DEVICE)
print(f'params: {model.num_params():,}')

scaler = torch.amp.GradScaler(enabled=(DTYPE == 'float16'))
optimizer = model.configure_optimizers(WEIGHT_DECAY, LEARNING_RATE, BETAS, device_type)
if RESUME:
    optimizer.load_state_dict(ckpt['optimizer'])
    ckpt = None

# Kept unwrapped: compile renames parameters with an _orig_mod. prefix, and the
# checkpoint should not carry it. cfg and estimate_mfu also live here.
raw_model = model
if COMPILE:
    model = torch.compile(model)


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ('train', 'val'):
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            x, y = get_batch(split)
            with ctx:
                _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def get_lr(it):
    if it < WARMUP_ITERS:                      # linear warmup
        return LEARNING_RATE * (it + 1) / (WARMUP_ITERS + 1)
    if it > LR_DECAY_ITERS:                    # constant floor after decay
        return MIN_LR
    ratio = (it - WARMUP_ITERS) / (LR_DECAY_ITERS - WARMUP_ITERS)
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return MIN_LR + coeff * (LEARNING_RATE - MIN_LR)


# --- training loop ---------------------------------------------------------
x, y = get_batch('train')
t0 = time.time()

while iter_num <= MAX_ITERS:
    lr = get_lr(iter_num)
    for group in optimizer.param_groups:
        group['lr'] = lr

    if iter_num % EVAL_INTERVAL == 0:
        losses = estimate_loss()
        print(f'step {iter_num}: train {losses["train"]:.4f} val {losses["val"]:.4f}')
        if writer:
            # Averaged over EVAL_ITERS batches, so these are the curves to trust.
            writer.add_scalar('loss/train', losses['train'], iter_num)
            writer.add_scalar('loss/val', losses['val'], iter_num)
            writer.add_scalar('loss/gap', losses['val'] - losses['train'], iter_num)
        if losses['val'] < best_val_loss or ALWAYS_SAVE:
            best_val_loss = min(losses['val'], best_val_loss)
            if iter_num > 0:
                torch.save({
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'config': raw_model.cfg.__dict__,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'meta': meta,
                }, os.path.join(OUT_DIR, 'ckpt.pt'))

    for _ in range(GRAD_ACCUM_STEPS):
        with ctx:
            _, loss = model(x, y)
            loss = loss / GRAD_ACCUM_STEPS
        # Prefetch the next batch while the backward pass runs.
        x, y = get_batch('train')
        scaler.scale(loss).backward()

    grad_norm = None
    if GRAD_CLIP != 0.0:
        scaler.unscale_(optimizer)
        # Pre-clip norm — the diagnostic worth watching. A rising or spiky trace
        # means the clip is the only thing holding the run together.
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    t1 = time.time()
    dt, t0 = t1 - t0, t1
    if iter_num % LOG_INTERVAL == 0:
        mfu = raw_model.estimate_mfu(tokens_per_iter, dt, FLOPS_PROMISED)
        batch_loss = loss.item() * GRAD_ACCUM_STEPS
        print(f'iter {iter_num}: loss {batch_loss:.4f} '
              f'| {dt * 1000:.0f}ms | mfu {mfu * 100:.1f}%')
        if writer:
            # Last micro-batch only, so it is noisy next to loss/train — but it
            # is the one trace that updates between evals.
            writer.add_scalar('loss/train_batch', batch_loss, iter_num)
            writer.add_scalar('lr', lr, iter_num)
            writer.add_scalar('perf/tokens', iter_num * tokens_per_iter, iter_num)
            if grad_norm is not None:
                writer.add_scalar('grad_norm', grad_norm.item(), iter_num)
            if iter_num > 0:
                # Iteration 0 carries the compile and warmup cost; logging it
                # would set the y-axis for the whole run.
                writer.add_scalar('perf/ms_per_iter', dt * 1000, iter_num)
                writer.add_scalar('perf/mfu', mfu, iter_num)
    iter_num += 1

if writer:
    writer.close()
