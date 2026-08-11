"""Supervised fine-tuning of a pretrained checkpoint on a chat corpus.

    python prepare_sft.py --max-examples 50000
    python sft.py
    python sft.py --init out/fineweb-edu/ckpt.pt --dataset smol-smoltalk

Three things separate this from train.py:

  * loss is masked to assistant turns — the model is graded on what it should
    say, not on reciting the question back;
  * batches start at conversation boundaries rather than at random offsets;
  * the two chat delimiters need embedding rows, which the pretrained model has
    spare because vocab_size was padded 50257 -> 50304.
"""

import argparse
import json
import math
import os
import time
from contextlib import nullcontext

import numpy as np
import torch

from chat import IM_END, IM_START
from model import GPT, GPTConfig

# --- data / io -------------------------------------------------------------
DATASET = 'smol-smoltalk'        # must match a `python prepare_sft.py <name>` run
INIT_FROM = os.path.join('out', 'fineweb-edu', 'ckpt.pt')
EVAL_INTERVAL = 100
EVAL_ITERS = 50
LOG_INTERVAL = 10
ALWAYS_SAVE = False

# --- logging ---------------------------------------------------------------
TENSORBOARD = True
RUN_NAME = ''

# --- optimisation ----------------------------------------------------------
EPOCHS = 2.0
BATCH_SIZE = 16                  # sequences per micro-step
GRAD_ACCUM_STEPS = 4             # 65,536 tokens/iter — a quarter of pretraining, because
                                 # SFT wants many more steps over far fewer tokens
LEARNING_RATE = 1e-4             # ~1/6 of the pretrain peak. Higher than the usual SFT
                                 # rule of thumb on purpose: two embedding rows start from
                                 # noise and the shift from web text to dialogue is large.
                                 # Drop to 5e-5 if val loss turns up early.
MIN_LR = 1e-5
WARMUP_ITERS = 100
WEIGHT_DECAY = 0.1
BETAS = (0.9, 0.95)
GRAD_CLIP = 1.0
DROPOUT = 0.1                    # a real multi-epoch pass over a small corpus, unlike pretraining

ap = argparse.ArgumentParser()
ap.add_argument('--dataset', default=DATASET, help='name under data/')
ap.add_argument('--init', default=INIT_FROM, help='pretrained checkpoint to start from')
ap.add_argument('--epochs', type=float, default=EPOCHS)
ap.add_argument('--lr', type=float, default=LEARNING_RATE)
_args = ap.parse_args()
DATASET, INIT_FROM, EPOCHS, LEARNING_RATE = _args.dataset, _args.init, _args.epochs, _args.lr
DATA_DIR = os.path.join('data', DATASET)
OUT_DIR = os.path.join('out', DATASET + '-sft')

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
assert meta.get('chat'), f'{DATA_DIR} was not built by prepare_sft.py'

if not os.path.exists(INIT_FROM):
    raise SystemExit(f'no checkpoint at {INIT_FROM} — pretrain with train.py first')
ckpt = torch.load(INIT_FROM, map_location=DEVICE, weights_only=False)

# The architecture is whatever was pretrained; only dropout changes.
cfg = GPTConfig(**ckpt['config'])
cfg.dropout = DROPOUT
assert cfg.vocab_size > IM_END, (
    f'vocab_size {cfg.vocab_size} has no room for the chat delimiters at '
    f'{IM_START}/{IM_END}')
if meta['max_len'] > cfg.block_size:
    print(f'warning: corpus allows {meta["max_len"]}-token conversations but block_size '
          f'is {cfg.block_size} — long ones will be cut by the window')

tokens_per_iter = GRAD_ACCUM_STEPS * BATCH_SIZE * cfg.block_size
train_tokens = meta['splits']['train']['tokens']
MAX_ITERS = max(1, int(EPOCHS * train_tokens / tokens_per_iter))
LR_DECAY_ITERS = MAX_ITERS

os.makedirs(OUT_DIR, exist_ok=True)
print(f'dataset: {DATASET} | {meta["splits"]["train"]["examples"]:,} conversations, '
      f'{train_tokens:,} tokens '
      f'({meta["splits"]["train"]["supervised_tokens"] / train_tokens:.1%} supervised)')
print(f'init: {INIT_FROM} @ iter {ckpt["iter_num"]:,}, val loss {ckpt["best_val_loss"]:.4f}')
print(f'tokens/iter: {tokens_per_iter:,} | {MAX_ITERS:,} iters for {EPOCHS:g} epochs')


# --- metrics ---------------------------------------------------------------
writer = None
if TENSORBOARD:
    from torch.utils.tensorboard import SummaryWriter

    log_dir = os.path.join(OUT_DIR, 'tb', RUN_NAME or time.strftime('%Y%m%d-%H%M%S'))
    writer = SummaryWriter(log_dir, flush_secs=30)
    hparams = {
        'init_from': INIT_FROM, 'dataset': DATASET, 'epochs': EPOCHS,
        'batch_size': BATCH_SIZE, 'grad_accum_steps': GRAD_ACCUM_STEPS,
        'tokens_per_iter': tokens_per_iter, 'max_iters': MAX_ITERS,
        'learning_rate': LEARNING_RATE, 'min_lr': MIN_LR, 'warmup_iters': WARMUP_ITERS,
        'weight_decay': WEIGHT_DECAY, 'grad_clip': GRAD_CLIP, 'dtype': DTYPE,
        **cfg.__dict__,
    }
    writer.add_text('config', '\n'.join(
        ['|param|value|', '|---|---|'] + [f'|{k}|{v}|' for k, v in hparams.items()]))
    print(f'tensorboard --logdir {os.path.join(OUT_DIR, "tb")}')


# --- data ------------------------------------------------------------------
SHARDS = {s: meta['splits'][s]['shards'] for s in ('train', 'val')}
SHARD_P = {s: np.array([sh['tokens'] for sh in SHARDS[s]], dtype=np.float64)
              / sum(sh['tokens'] for sh in SHARDS[s]) for s in SHARDS}
rng = np.random.default_rng(1337)


def _stem(split, j):
    return os.path.join(DATA_DIR, SHARDS[split][j]['file'].removesuffix('.bin'))


def _load_starts(split, j):
    """Conversation offsets that leave a full window of tokens after them."""
    starts = np.fromfile(_stem(split, j) + '.idx.bin', dtype=np.uint64).astype(np.int64)
    room = SHARDS[split][j]['tokens'] - cfg.block_size - 1
    starts = starts[starts <= room]
    if len(starts) == 0:
        raise SystemExit(
            f'{split} shard {j} holds fewer than block_size+1 tokens — rerun '
            f'prepare_sft.py with a larger --max-examples/--val-examples')
    return starts


STARTS = {s: [_load_starts(s, j) for j in range(len(SHARDS[s]))] for s in SHARDS}


def get_batch(split):
    shards = SHARDS[split]
    j = 0 if len(shards) == 1 else rng.choice(len(shards), p=SHARD_P[split])
    stem = _stem(split, j)
    # Reopened every call, as in train.py: a live memmap across a long run leaks
    # the page cache into RSS, and the OS keeps the pages warm anyway.
    data = np.memmap(stem + '.bin', dtype=np.dtype(meta['dtype']), mode='r')
    mask = np.memmap(stem + '.mask.bin', dtype=np.uint8, mode='r')

    # Windows open on a <|im_start|>, never mid-turn. Conversations are packed,
    # so a window usually spans several — each carrying its own mask, so the
    # supervision stays correct even where the boundaries fall.
    T = cfg.block_size
    ix = STARTS[split][j][rng.integers(len(STARTS[split][j]), size=BATCH_SIZE)]
    x = torch.stack([torch.from_numpy(data[i:i + T].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + T].astype(np.int64)) for i in ix])
    m = torch.stack([torch.from_numpy(mask[i + 1:i + 1 + T].astype(np.bool_)) for i in ix])
    y = y.masked_fill(~m, -100)   # prompt and header positions do not train

    if device_type == 'cuda':
        x = x.pin_memory().to(DEVICE, non_blocking=True)
        y = y.pin_memory().to(DEVICE, non_blocking=True)
    else:
        x, y = x.to(DEVICE), y.to(DEVICE)
    return x, y


# --- model -----------------------------------------------------------------
model = GPT(cfg)
state = {k.removeprefix('_orig_mod.'): v for k, v in ckpt['model'].items()}
model.load_state_dict(state)

# The delimiter rows exist but were never trained on: tied to lm_head, they have
# spent the whole pretraining run being pushed down by the softmax denominator.
# Reset them to the init distribution so they start neutral rather than repelled.
with torch.no_grad():
    for tok in (IM_START, IM_END):
        model.wte.weight[tok].normal_(mean=0.0, std=0.02)

model.to(DEVICE)
print(f'params: {model.num_params():,}')

scaler = torch.amp.GradScaler(enabled=(DTYPE == 'float16'))
optimizer = model.configure_optimizers(WEIGHT_DECAY, LEARNING_RATE, BETAS, device_type)
ckpt = None

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
    if it < WARMUP_ITERS:
        return LEARNING_RATE * (it + 1) / (WARMUP_ITERS + 1)
    if it > LR_DECAY_ITERS:
        return MIN_LR
    ratio = (it - WARMUP_ITERS) / (LR_DECAY_ITERS - WARMUP_ITERS)
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return MIN_LR + coeff * (LEARNING_RATE - MIN_LR)


def save(iter_num, best_val_loss):
    torch.save({
        'model': raw_model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'config': raw_model.cfg.__dict__,
        'iter_num': iter_num,
        'best_val_loss': best_val_loss,
        # Carries chat=True and the delimiter ids, which is how sample.py knows
        # to apply the template instead of treating the prompt as raw text.
        'meta': meta,
    }, os.path.join(OUT_DIR, 'ckpt.pt'))


# --- training loop ---------------------------------------------------------
iter_num = 0
best_val_loss = float('inf')
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
            writer.add_scalar('loss/train', losses['train'], iter_num)
            writer.add_scalar('loss/val', losses['val'], iter_num)
            writer.add_scalar('loss/gap', losses['val'] - losses['train'], iter_num)
        if losses['val'] < best_val_loss or ALWAYS_SAVE:
            best_val_loss = min(losses['val'], best_val_loss)
            if iter_num > 0:
                save(iter_num, best_val_loss)

    for _ in range(GRAD_ACCUM_STEPS):
        with ctx:
            _, loss = model(x, y)
            # Each micro-batch holds a different number of supervised tokens, so
            # this weights micro-batches equally rather than tokens. The bias is
            # small at this batch size and the alternative costs a second pass to
            # count targets before scaling.
            loss = loss / GRAD_ACCUM_STEPS
        x, y = get_batch('train')
        scaler.scale(loss).backward()

    grad_norm = None
    if GRAD_CLIP != 0.0:
        scaler.unscale_(optimizer)
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
            writer.add_scalar('loss/train_batch', batch_loss, iter_num)
            writer.add_scalar('lr', lr, iter_num)
            writer.add_scalar('perf/tokens', iter_num * tokens_per_iter, iter_num)
            if grad_norm is not None:
                writer.add_scalar('grad_norm', grad_norm.item(), iter_num)
            if iter_num > 0:
                writer.add_scalar('perf/ms_per_iter', dt * 1000, iter_num)
                writer.add_scalar('perf/mfu', mfu, iter_num)
    iter_num += 1

if best_val_loss == float('inf'):
    save(iter_num, best_val_loss)     # nothing ever evaluated better; keep the last state
if writer:
    writer.close()
print(f'done — python sample.py --ckpt {os.path.join(OUT_DIR, "ckpt.pt")} '
      f'--prompt "What is a transformer?"')
