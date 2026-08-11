"""Tokenize a chat SFT corpus into token / mask / index shards.

    python prepare_sft.py                            # all 460k conversations
    python prepare_sft.py --max-examples 50000       # a slice, to check the pipeline first

Writes data/<dataset>/:

    {split}_NNNNNN.bin       uint16 token ids, conversations concatenated
    {split}_NNNNNN.mask.bin  uint8, 1 where the token is a supervised target
    {split}_NNNNNN.idx.bin   uint64 start offset of each conversation in the shard
    meta.json                manifest

The index is what keeps SFT honest: sft.py samples windows that begin at a
conversation boundary, so a batch never opens midway through an assistant turn
with the prompt that produced it out of view.
"""

import argparse
import json
import os
import sys

import numpy as np
from datasets import load_dataset

from chat import IM_END, IM_START, enc, render_fit

DATASETS = {
    'smol-smoltalk': {
        'path': 'HuggingFaceTB/smol-smoltalk',   # 460k conversations, for models <1B
        'name': None,
        'train_split': 'train',
        'val_split': 'test',
    },
}

MESSAGES_COLUMN = 'messages'
SHARD_TOKENS = 100_000_000       # ~200MB of tokens + ~100MB of mask per shard
VAL_EXAMPLES = 5_000             # the test split is 24k; this is plenty to track a curve
ROOT = os.path.dirname(os.path.abspath(__file__))


class ShardWriter:
    """Buffers whole conversations, rolling to a new shard past SHARD_TOKENS.

    A shard must end where a conversation ends, or its index file would point
    at offsets the token file does not contain.
    """

    def __init__(self, out_dir, split):
        self.out_dir, self.split = out_dir, split
        self.shards, self.total, self.supervised = [], 0, 0
        self._reset()

    def _reset(self):
        self.tokens, self.masks, self.starts, self.n = [], [], [], 0

    def add(self, tokens, mask):
        self.starts.append(self.n)
        self.tokens.append(np.asarray(tokens, dtype=np.uint16))
        self.masks.append(np.asarray(mask, dtype=np.uint8))
        self.n += len(tokens)
        self.supervised += int(sum(mask))
        if self.n >= SHARD_TOKENS:
            self.flush()

    def flush(self):
        if not self.starts:
            return
        stem = f'{self.split}_{len(self.shards):06d}'
        np.concatenate(self.tokens).tofile(os.path.join(self.out_dir, stem + '.bin'))
        np.concatenate(self.masks).tofile(os.path.join(self.out_dir, stem + '.mask.bin'))
        np.asarray(self.starts, dtype=np.uint64).tofile(
            os.path.join(self.out_dir, stem + '.idx.bin'))
        self.shards.append({
            'file': stem + '.bin',
            'tokens': self.n,
            'examples': len(self.starts),
        })
        self.total += self.n
        self._reset()

    def close(self):
        self.flush()
        return {
            'tokens': self.total,
            'supervised_tokens': self.supervised,
            'examples': sum(s['examples'] for s in self.shards),
            'shards': self.shards,
        }


def write_split(cfg, split_key, out_dir, split, max_len, max_examples=None):
    """Render every conversation in `split_key` into shards. Returns a manifest."""
    dset = load_dataset(cfg['path'], name=cfg['name'], split=split_key, streaming=True)
    writer = ShardWriter(out_dir, split)
    seen = dropped = truncated = 0

    for row in dset:
        seen += 1
        messages = row[MESSAGES_COLUMN]
        tokens, mask = render_fit(messages, max_len)
        if tokens is None:
            # Not even the opening exchange fits the window.
            dropped += 1
        else:
            # One <|im_start|> per turn kept, so this counts turns dropped off
            # the end rather than guessing from token totals.
            if tokens.count(IM_START) < len(messages):
                truncated += 1
            writer.add(tokens, mask)

        if seen % 2000 == 0:
            sys.stdout.write(f'\r  {seen:,} read | {writer.total + writer.n:,} tokens '
                             f'| {len(writer.shards)} shards | {dropped:,} dropped')
            sys.stdout.flush()
        if max_examples and seen - dropped >= max_examples:
            break

    sys.stdout.write('\n')
    info = writer.close()
    kept = info['examples']
    print(f'  kept {kept:,}/{seen:,} | {truncated:,} cut to a turn boundary at {max_len} '
          f'tokens, {dropped:,} dropped whole')
    if kept:
        print(f'  {info["tokens"]:,} tokens, {info["supervised_tokens"] / info["tokens"]:.1%} '
              f'supervised, {info["tokens"] / kept:.0f} tokens/conversation')
    return info


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('dataset', nargs='?', default='smol-smoltalk', choices=sorted(DATASETS))
    ap.add_argument('--config', help='override the HF config name')
    ap.add_argument('--max-len', type=int, default=1024,
                    help="drop conversations longer than this; match the model's block_size")
    ap.add_argument('--max-examples', type=int, default=None,
                    help='stop after roughly this many kept training conversations')
    ap.add_argument('--val-examples', type=int, default=VAL_EXAMPLES)
    args = ap.parse_args()

    cfg = dict(DATASETS[args.dataset])
    if args.config:
        cfg['name'] = args.config

    out_dir = os.path.join(ROOT, 'data', args.dataset)
    os.makedirs(out_dir, exist_ok=True)
    print(f'{args.dataset}: {cfg["path"]}'
          + (f' [{cfg["name"]}]' if cfg['name'] else '')
          + f' -> {out_dir} (max_len {args.max_len})')

    meta = {
        'dataset': args.dataset,
        'source': cfg['path'],
        'config': cfg['name'],
        'encoding': enc.name,
        'dtype': 'uint16',
        'vocab_size': enc.n_vocab,
        'im_start': IM_START,
        'im_end': IM_END,
        'eot_token': IM_END,      # what generation stops on
        'chat': True,             # sample.py switches templating on this
        'max_len': args.max_len,
        'splits': {},
    }

    print('val:')
    meta['splits']['val'] = write_split(cfg, cfg['val_split'], out_dir, 'val',
                                        args.max_len, args.val_examples)
    print('train:')
    meta['splits']['train'] = write_split(cfg, cfg['train_split'], out_dir, 'train',
                                          args.max_len, args.max_examples)

    with open(os.path.join(out_dir, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    print(f'\nwrote {out_dir}/meta.json — now: python sft.py --dataset {args.dataset}')


if __name__ == '__main__':
    main()
    # Same finalisation race as prepare.py: the streaming reader's background
    # threads are still in flight when the interpreter tears down and abort with
    # a GIL assertion, after everything is safely on disk. Skip finalisation so a
    # finished run exits 0 rather than dumping core.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
