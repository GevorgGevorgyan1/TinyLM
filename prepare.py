"""Tokenize a text dataset into flat token shards on disk.

    python prepare.py tinystories
    python prepare.py fineweb-edu
    python prepare.py fineweb-edu --max-tokens 1e9    # quick slice

Streams documents in batches through tiktoken's multithreaded Rust encoder and
appends straight to fixed-size shards, so peak RAM is one batch regardless of
dataset size. Produces data/<dataset>/{train,val}_NNNNNN.bin plus a meta.json
describing them. An interrupted run resumes at the last completed shard.
"""

import argparse
import json
import os
import sys
import time
from itertools import chain

import numpy as np
import tiktoken
from datasets import load_dataset

# Everything that differs between corpora; the pipeline below is dataset-agnostic.
DATASETS = {
    "tinystories": {
        "path": "roneneldan/TinyStories",
        "name": None,
        "val_split": "validation",   # ships its own held-out split
        "val_tokens": None,
        "streaming": False,          # 2GB of text, arrow random access is free
    },
    "fineweb-edu": {
        "path": "HuggingFaceFW/fineweb-edu",
        "name": "sample-10BT",       # ~9.7B gpt2 tokens; also sample-100BT, sample-350BT
        "val_split": None,           # train-only, so carve a val slice off the front
        "val_tokens": 10_000_000,
        "streaming": True,           # ~45GB of text, never lands on disk as arrow
    },
}

ENCODING = "gpt2"                    # ~4% worse than cl100k here, but half the vocab
BATCH_SIZE = 8192                    # documents handed to the encoder at once
SHARD_TOKENS = 100_000_000           # ~200MB per shard at uint16
TEXT_COLUMN = "text"
NUM_THREADS = os.cpu_count() or 8
ROOT = os.path.dirname(os.path.abspath(__file__))

enc = tiktoken.get_encoding(ENCODING)
EOT = enc.eot_token
# Widen automatically rather than silently wrapping if the encoding ever changes.
DTYPE = np.uint16 if enc.n_vocab <= np.iinfo(np.uint16).max else np.uint32


class ShardWriter:
    """Appends token arrays to numbered shards, rolling over at SHARD_TOKENS.

    Completed shards are recorded in a progress file, so an interrupted run
    restarts at the last shard boundary rather than at the top of the corpus —
    which matters once a pass takes hours.
    """

    def __init__(self, out_dir, split):
        self.out_dir, self.split = out_dir, split
        self.state_path = os.path.join(out_dir, f".{split}.progress.json")
        self.shards, self.tokens, self.docs = [], 0, 0
        self.f, self.shard_tokens = None, 0

    def resume(self, skip_docs=0):
        """Reload progress, discarding any shard a crash left half-written."""
        self.docs = skip_docs
        if not os.path.exists(self.state_path):
            return False
        with open(self.state_path) as fh:
            state = json.load(fh)
        self.shards, self.tokens, self.docs = state["shards"], state["tokens"], state["docs"]
        # Anything numbered past the last recorded shard is a partial write.
        stale = os.path.join(self.out_dir, self._name(len(self.shards)))
        if os.path.exists(stale):
            os.remove(stale)
        return True

    def _name(self, i):
        return f"{self.split}_{i:06d}.bin"

    def write(self, tokens, n_docs):
        if self.f is None:
            path = os.path.join(self.out_dir, self._name(len(self.shards)))
            self.f = open(path, "wb", buffering=4 << 20)
            self.shard_tokens = 0
        self.f.write(tokens)  # buffer protocol, no intermediate bytes copy
        self.shard_tokens += len(tokens)
        self.tokens += len(tokens)
        self.docs += n_docs
        # Roll on batch boundaries only. A shard must end where a document ends,
        # or the resume doc offset would skip tokens that were never written.
        if self.shard_tokens >= SHARD_TOKENS:
            self._roll()

    def _roll(self):
        self.f.close()
        self.f = None
        self.shards.append({
            "file": self._name(len(self.shards)),
            "tokens": self.shard_tokens,
        })
        # Checkpoint only on a closed, fully flushed shard — that is the entire
        # point of having a boundary.
        tmp = self.state_path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump({"shards": self.shards, "tokens": self.tokens, "docs": self.docs}, fh)
        os.replace(tmp, self.state_path)

    def close(self):
        if self.f is not None and self.shard_tokens > 0:
            self._roll()
        return {
            "tokens": self.tokens,
            "bytes": sum(os.path.getsize(os.path.join(self.out_dir, s["file"]))
                         for s in self.shards),
            "shards": self.shards,
        }


def iter_batches(cfg, split_key, skip_docs=0):
    """Yield lists of raw document strings, BATCH_SIZE at a time."""
    if cfg["streaming"]:
        dset = load_dataset(cfg["path"], name=cfg["name"], split=split_key, streaming=True)
        if skip_docs:
            print(f"  skipping {skip_docs:,} already-consumed docs")
            dset = dset.skip(skip_docs)
        batch = []
        for row in dset:
            batch.append(row[TEXT_COLUMN])
            if len(batch) == BATCH_SIZE:
                yield batch
                batch = []
        if batch:
            yield batch
    else:
        dset = load_dataset(cfg["path"], name=cfg["name"], split=split_key)
        for start in range(skip_docs, len(dset), BATCH_SIZE):
            yield dset[start:start + BATCH_SIZE][TEXT_COLUMN]


def write_split(cfg, split_key, out_dir, split, max_tokens=None, skip_docs=0):
    """Encode `split_key` into shards under out_dir. Returns a manifest dict."""
    writer = ShardWriter(out_dir, split)
    if writer.resume(skip_docs):
        print(f"  resuming: {len(writer.shards)} shards, {writer.tokens:,} tokens, "
              f"{writer.docs:,} docs consumed")

    t0, base = time.time(), writer.tokens
    for texts in iter_batches(cfg, split_key, writer.docs):
        # Encodes the batch across a Rust thread pool with the GIL released,
        # which is the entire reason this runs in one process.
        docs = enc.encode_batch(
            texts,
            num_threads=NUM_THREADS,
            allowed_special={"<|endoftext|>"},
            disallowed_special=(),
        )
        for doc in docs:
            doc.append(EOT)

        count = sum(map(len, docs))
        flat = np.fromiter(chain.from_iterable(docs), dtype=DTYPE, count=count)
        if max_tokens is not None:
            # One batch is ~8M tokens on fineweb-edu, so stopping only on batch
            # boundaries would blow straight past a small cap. Trim inside the
            # batch instead. The doc count still reports every document the
            # stream consumed, so a carved-out val slice and train cannot overlap.
            flat = flat[:max(max_tokens - writer.tokens, 0)]
        if len(flat):
            writer.write(flat, len(docs))

        rate = (writer.tokens - base) / max(time.time() - t0, 1e-9) / 1e6
        sys.stdout.write(f"\r  {writer.docs:,} docs | {writer.tokens:,} tokens "
                         f"| {len(writer.shards)} shards | {rate:.2f}M tok/s")
        sys.stdout.flush()

        if max_tokens is not None and writer.tokens >= max_tokens:
            break

    sys.stdout.write("\n")
    return writer.close(), writer.docs


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", choices=sorted(DATASETS), help="corpus to tokenize")
    ap.add_argument("--config", help="override the HF config name, e.g. sample-100BT")
    ap.add_argument("--max-tokens", type=float, default=None,
                    help="stop after roughly this many train tokens")
    ap.add_argument("--val-tokens", type=float, default=None,
                    help="override the carved-out val slice size")
    args = ap.parse_args()

    cfg = dict(DATASETS[args.dataset])
    if args.config:
        cfg["name"] = args.config
    if args.val_tokens:
        cfg["val_split"], cfg["val_tokens"] = None, int(args.val_tokens)
    max_tokens = int(args.max_tokens) if args.max_tokens else None

    out_dir = os.path.join(ROOT, "data", args.dataset)
    os.makedirs(out_dir, exist_ok=True)
    print(f"{args.dataset}: {cfg['path']}"
          + (f" [{cfg['name']}]" if cfg["name"] else "")
          + f" -> {out_dir} ({NUM_THREADS} encoder threads)")

    meta = {
        "dataset": args.dataset,
        "source": cfg["path"],
        "config": cfg["name"],
        "encoding": ENCODING,
        "dtype": np.dtype(DTYPE).name,
        "vocab_size": enc.n_vocab,
        "eot_token": EOT,
        "splits": {},
    }

    # Val first when it has to be carved off the front of train: the doc count it
    # consumes becomes train's starting offset, so the two never overlap.
    train_skip = 0
    print("val:")
    if cfg["val_split"]:
        meta["splits"]["val"], _ = write_split(cfg, cfg["val_split"], out_dir, "val")
    else:
        meta["splits"]["val"], train_skip = write_split(
            cfg, "train", out_dir, "val", max_tokens=cfg["val_tokens"])

    print("train:")
    meta["splits"]["train"], _ = write_split(
        cfg, "train", out_dir, "train", max_tokens=max_tokens, skip_docs=train_skip)

    for split, info in meta["splits"].items():
        print(f"  {split}: {info['tokens']:,} tokens, {info['bytes'] / 1e9:.2f} GB, "
              f"{len(info['shards'])} shards")

    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # Load a shard for training with (native byte order, zero-copy):
    #   tokens = np.memmap("data/<dataset>/train_000000.bin", dtype=np.uint16, mode="r")


if __name__ == "__main__":
    main()
    # tiktoken's Rust pool and the streaming reader's background threads race
    # during interpreter finalisation and abort with a GIL assertion — after all
    # the work is safely on disk. Skip finalisation so a finished run exits 0
    # rather than dumping core at the end of a multi-hour job.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
