"""Evaluate a checkpoint on HellaSwag (4-way commonsense completion).

    python hellaswag.py                                  # out/fineweb-edu/ckpt.pt
    python hellaswag.py --ckpt out/tinystories/ckpt.pt --limit 500

Each example is a context plus four candidate endings. We score every candidate
by the model's loss over the *ending tokens only* and pick the cheapest. Two
metrics, following the standard convention:

    acc      — argmin of summed loss, so it favours short endings
    acc_norm — argmin of per-token loss, the number normally reported

Random guessing is 25%. GPT-2 124M scores ~29.5% acc_norm, so anything smaller
sits close to chance and is only useful as a relative baseline.
"""

import argparse
import json
import os
import urllib.request
from contextlib import nullcontext

import tiktoken
import torch
import torch.nn.functional as F

from model import GPT, GPTConfig

URL = "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl"
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "hellaswag")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--ckpt', default=os.path.join('out', 'fineweb-edu', 'ckpt.pt'))
    p.add_argument('--batch', type=int, default=16, help='examples collated at once (x4 rows)')
    # Peak memory is the float32 logits — rows x tokens x 50304 x 4 bytes, which
    # is ~2GB for 64 rows of the longest examples. Scoring in row-chunks bounds
    # that regardless of --batch.
    p.add_argument('--chunk', type=int, default=16, help='rows per forward pass')
    p.add_argument('--limit', type=int, default=0, help='stop after N examples; 0 = all')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--dtype', default=None, choices=['float32', 'bfloat16', 'float16'])
    return p.parse_args()


def download():
    """Fetch the validation split once; it is ~47MB of jsonl."""
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, "hellaswag_val.jsonl")
    if not os.path.exists(path):
        print(f"downloading {URL}")
        urllib.request.urlretrieve(URL, path)
    with open(path) as f:
        return [json.loads(line) for line in f]


def render(example, enc):
    """One example -> four token rows and a mask marking the ending region."""
    ctx_tokens = enc.encode(example["ctx"])
    rows, masks = [], []
    for end in example["endings"]:
        # Leading space matters: GPT-2 BPE encodes word-initial tokens with it,
        # and without it every ending starts on an off-distribution token.
        end_tokens = enc.encode(" " + end)
        rows.append(ctx_tokens + end_tokens)
        masks.append([0] * len(ctx_tokens) + [1] * len(end_tokens))
    return rows, masks, int(example["label"])


def collate(rows, masks, device):
    n, width = len(rows), max(len(r) for r in rows)
    tokens = torch.zeros(n, width, dtype=torch.long)
    mask = torch.zeros(n, width, dtype=torch.long)
    for i, (r, m) in enumerate(zip(rows, masks)):
        tokens[i, :len(r)] = torch.tensor(r)
        mask[i, :len(m)] = torch.tensor(m)
    # Padding sits at the end of each row and attention is causal, so real
    # tokens never attend to it; the mask keeps it out of the loss.
    return tokens.to(device), mask.to(device)


@torch.no_grad()
def score(model, tokens, mask, chunk):
    """Per-row (summed, per-token) loss over the masked ending region."""
    sums = []
    for i in range(0, tokens.size(0), chunk):
        rows, row_mask = tokens[i:i + chunk], mask[i:i + chunk, 1:]
        # targets is passed only to take forward()'s full-sequence branch — with
        # targets=None the model returns logits for the last position alone. The
        # scalar loss it computes alongside is discarded for the masked one below.
        logits, _ = model(rows, rows)
        losses = F.cross_entropy(
            logits[:, :-1, :].reshape(-1, logits.size(-1)).float(),
            rows[:, 1:].reshape(-1), reduction='none',
        ).view(rows.size(0), -1) * row_mask
        sums.append(losses.sum(dim=1))
    summed = torch.cat(sums)
    return summed, summed / mask[:, 1:].sum(dim=1)


def main():
    args = parse_args()
    if not os.path.exists(args.ckpt):
        raise SystemExit(f'no checkpoint at {args.ckpt} — run train.py first')

    device_type = 'cuda' if args.device.startswith('cuda') else 'cpu'
    dtype = args.dtype or ('bfloat16' if device_type == 'cuda'
                           and torch.cuda.is_bf16_supported() else 'float32')
    ctx = (nullcontext() if device_type == 'cpu' or dtype == 'float32' else
           torch.amp.autocast(device_type=device_type, dtype=getattr(torch, dtype)))

    ckpt = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    model = GPT(GPTConfig(**ckpt['config']))
    model.load_state_dict({k.removeprefix('_orig_mod.'): v for k, v in ckpt['model'].items()})
    model.eval().to(args.device)
    # From the checkpoint, not hardcoded: scoring must use the training encoding.
    enc = tiktoken.get_encoding(ckpt['meta']['encoding'])
    block = model.cfg.block_size
    print(f'{args.ckpt}: iter {ckpt["iter_num"]:,} | val loss {ckpt["best_val_loss"]:.4f} '
          f'| {model.num_params() / 1e6:.2f}M params | block_size {block}')

    examples = download()
    if args.limit:
        examples = examples[:args.limit]

    # source -> [n, correct, correct_norm, gold_loss, all_loss, margin]. The file
    # is ordered activitynet-then-wikihow and the two behave very differently, so
    # a single number hides most of what the eval is telling you.
    stats, skipped, n = {}, 0, 0
    pending = []
    for example in examples + [None]:                 # None flushes the last batch
        if example is not None:
            rows, masks, label = render(example, enc)
            if max(len(r) for r in rows) > block:
                skipped += 1                          # cannot score what will not fit
                continue
            pending.append((rows, masks, label, example['source_id'].split('~')[0]))
        if pending and (len(pending) >= args.batch or example is None):
            flat_rows = [r for rows, _, _, _ in pending for r in rows]
            flat_masks = [m for _, masks, _, _ in pending for m in masks]
            tokens, mask = collate(flat_rows, flat_masks, args.device)
            with ctx:
                summed, per_token = score(model, tokens, mask, args.chunk)
            summed = summed.view(-1, 4)
            per_token = per_token.view(-1, 4)
            for i, (_, _, label, source) in enumerate(pending):
                n += 1
                acc = stats.setdefault(source, [0, 0, 0, 0.0, 0.0, 0.0])
                pt = per_token[i]
                gold = pt[label].item()
                acc[0] += 1
                acc[1] += int(summed[i].argmin().item() == label)
                acc[2] += int(pt.argmin().item() == label)
                acc[3] += gold                        # per-token NLL of the true ending
                acc[4] += pt.mean().item()            # ...averaged over all four
                # Positive means the model finds the true ending cheaper than the
                # distractors — a continuous signal that survives at chance accuracy.
                acc[5] += ((pt.sum().item() - gold) / 3) - gold
            pending = []
            tot = [sum(v[j] for v in stats.values()) for j in range(6)]
            print(f'\r  {n:,}/{len(examples):,} | acc {tot[1] / n:.4f} '
                  f'| acc_norm {tot[2] / n:.4f} | loss {tot[3] / n:.4f}', end='', flush=True)

    def report(name, s):
        cnt = s[0]
        print(f'  {name:12s} {cnt:6,}  acc {s[1] / cnt:.4f}  acc_norm {s[2] / cnt:.4f}  '
              f'loss {s[3] / cnt:.4f}  all {s[4] / cnt:.4f}  margin {s[5] / cnt:+.4f}')

    print(f'\n\nHellaSwag: {n:,} examples'
          + (f' ({skipped} skipped, longer than block_size)' if skipped else ''))
    for source in sorted(stats):
        report(source, stats[source])
    report('OVERALL', [sum(v[j] for v in stats.values()) for j in range(6)])
    print('\n  loss   = per-token NLL of the correct ending (compare to val loss)\n'
          '  all    = same, averaged over all four candidates\n'
          '  margin = mean wrong-ending loss minus correct-ending loss; >0 favours the truth\n'
          '  random baseline is 0.2500 acc_norm; GPT-2 124M scores ~0.2950')


if __name__ == '__main__':
    main()
