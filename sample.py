"""Sample from a trained GPT checkpoint.

    python sample.py --prompt "Once upon a time"
    python sample.py --prompt "Lily found a key" --num_samples 5 --temperature 0.7
    python sample.py                                  # unconditional, from <|endoftext|>

An SFT checkpoint from sft.py is detected from its metadata: the prompt is then
wrapped in the chat template and generation stops at <|im_end|>.

    python sample.py --ckpt out/smol-smoltalk-sft/ckpt.pt --prompt "Why is the sky blue?"
"""

import argparse
import os
from contextlib import nullcontext

import tiktoken
import torch

import chat
from model import GPT, GPTConfig


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--ckpt', default=os.path.join('out', 'fineweb-edu', 'ckpt.pt'),
                   help='out/<dataset>/ckpt.pt')
    p.add_argument('--prompt', default='',
                   help='priming text, or @path/to/file.txt to read from disk. '
                        'Empty means start a fresh document from the EOT token. '
                        'On a chat checkpoint this is the user turn.')
    p.add_argument('--system', default=None,
                   help='system turn, chat checkpoints only')
    p.add_argument('--num_samples', type=int, default=3)
    p.add_argument('--max_new_tokens', type=int, default=300)
    p.add_argument('--temperature', type=float, default=0.8,
                   help='<1 sharpens the distribution, >1 flattens it, '
                        '0 is greedy (deterministic)')
    p.add_argument('--top_k', type=int, default=200,
                   help='sample only from the k likeliest tokens; 0 disables')
    p.add_argument('--seed', type=int, default=1337)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--dtype', default=None, choices=['float32', 'bfloat16', 'float16'],
                   help='default: bfloat16 on a capable GPU, else float32')
    p.add_argument('--compile', action='store_true',
                   help='worth it only when generating a lot; the compile itself '
                        'costs more than a few short samples')
    return p.parse_args()


def main():
    args = parse_args()
    if args.temperature < 0:
        raise SystemExit('--temperature must be >= 0 (0 means greedy)')
    if args.temperature == 0 and args.num_samples > 1:
        # Greedy is a function of the prompt alone, so every row would be
        # byte-identical. Collapse rather than print the same story N times.
        print(f'temperature 0 is deterministic — collapsing '
              f'{args.num_samples} samples to 1\n')
        args.num_samples = 1
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device_type = 'cuda' if args.device.startswith('cuda') else 'cpu'
    dtype = args.dtype or ('bfloat16' if device_type == 'cuda'
                           and torch.cuda.is_bf16_supported() else 'float32')
    ctx = (nullcontext() if device_type == 'cpu' or dtype == 'float32' else
           torch.amp.autocast(device_type=device_type, dtype=getattr(torch, dtype)))

    # --- model -------------------------------------------------------------
    if not os.path.exists(args.ckpt):
        raise SystemExit(f'no checkpoint at {args.ckpt} — run train.py first')
    ckpt = torch.load(args.ckpt, map_location=args.device, weights_only=False)

    model = GPT(GPTConfig(**ckpt['config']))
    # torch.compile wraps parameter names in an _orig_mod. prefix. train.py saves
    # the uncompiled module so this is a no-op there, but checkpoints from a
    # compiled save would otherwise fail to load.
    state = {k.removeprefix('_orig_mod.'): v for k, v in ckpt['model'].items()}
    model.load_state_dict(state)
    model.eval()
    model.to(args.device)
    if args.compile:
        model = torch.compile(model)

    print(f'{args.ckpt}: iter {ckpt["iter_num"]:,} | val loss '
          f'{ckpt["best_val_loss"]:.4f} | {model.num_params() / 1e6:.2f}M params')
    print(f'device={args.device} dtype={dtype}\n')

    # --- tokenizer ---------------------------------------------------------
    # From the checkpoint, not hardcoded: a model trained on a different
    # encoding must not be decoded with this one.
    meta = ckpt['meta']
    chat_mode = meta.get('chat', False)
    if chat_mode:
        # GPT-2 BPE plus the two turn delimiters, and generation ends a turn
        # rather than a document.
        enc, eot = chat.enc, chat.IM_END
    else:
        enc = tiktoken.get_encoding(meta['encoding'])
        eot = meta['eot_token']

    prompt = args.prompt
    if prompt.startswith('@'):
        with open(prompt[1:]) as f:
            prompt = f.read()

    if chat_mode:
        if not prompt:
            raise SystemExit('a chat checkpoint needs --prompt: there is no '
                             'unconditional mode once every turn has a speaker')
        # Ends on the assistant header, so the first token sampled is the first
        # token of the reply.
        start_ids = chat.render_prompt(prompt, system=args.system)
        start_ids = start_ids[-model.cfg.block_size:]
    elif prompt:
        start_ids = enc.encode(prompt, allowed_special={'<|endoftext|>'})
        # The tail is what the model conditions on, so keep room to generate.
        start_ids = start_ids[-model.cfg.block_size:]
    else:
        # Every document in the training stream is followed by EOT, so EOT is
        # what the model has learned to read as "a new story begins here".
        start_ids = [eot]

    idx = torch.tensor(start_ids, dtype=torch.long, device=args.device)
    idx = idx.unsqueeze(0).expand(args.num_samples, -1)  # one batched forward pass

    # --- generate ----------------------------------------------------------
    with ctx:
        out = model.generate(
            idx,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k or None,
            eot_token=eot,
        )

    for i, row in enumerate(out.tolist()):
        completion = row[len(start_ids):]
        if eot in completion:
            completion = completion[:completion.index(eot)]
        # vocab_size is padded to 50304 for the matmul, so ids past the real
        # vocab are decodable by nothing — drop them rather than crash tiktoken.
        completion = [t for t in completion if t < enc.n_vocab]
        print(f'--- sample {i + 1}/{args.num_samples} '
              f'({len(completion)} tokens) ---')
        if chat_mode:
            if args.system:
                print(f'system:    {args.system}')
            print(f'user:      {prompt}')
            print(f'assistant: {enc.decode(completion)}')
        else:
            print(prompt + enc.decode(completion))
        print()


if __name__ == '__main__':
    main()
