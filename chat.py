"""ChatML-style conversation rendering on top of GPT-2 BPE.

The pretrained model has vocab_size 50304 — GPT-2's 50257 padded up to a
multiple of 64 — so ids 50257..50303 already exist as rows in the embedding
matrix and have never been used. Two of them become the turn delimiters, which
is why chat tokens cost no embedding resize and no checkpoint surgery.

A conversation renders as:

    <|im_start|>user\\n{content}<|im_end|>\\n<|im_start|>assistant\\n{reply}<|im_end|>\\n

alongside a parallel mask marking which positions are supervised targets.
"""

import tiktoken

BASE_ENCODING = 'gpt2'
IM_START_STR = '<|im_start|>'
IM_END_STR = '<|im_end|>'
IM_START = 50257
IM_END = 50258

_base = tiktoken.get_encoding(BASE_ENCODING)
assert _base.n_vocab == IM_START, (
    f'{BASE_ENCODING} has n_vocab {_base.n_vocab}, so {IM_START} is not the first '
    f'free id — the delimiters would collide with real tokens')

# Same merges and split pattern as gpt2, two extra special tokens. Every id
# below 50257 keeps its meaning, so the pretrained embedding table transfers
# across unchanged.
enc = tiktoken.Encoding(
    name='gpt2-chatml',
    pat_str=_base._pat_str,
    mergeable_ranks=_base._mergeable_ranks,
    special_tokens={**_base._special_tokens, IM_START_STR: IM_START, IM_END_STR: IM_END},
)

NEWLINE = enc.encode('\n')


def _text(s):
    """Encode message content as ordinary text.

    allowed_special is empty and disallowed_special is empty, so a literal
    '<|im_end|>' typed inside a user turn encodes as its constituent bytes
    rather than raising — or, worse, forging a turn boundary.
    """
    return enc.encode(s, allowed_special=set(), disallowed_special=())


def render_turns(messages, train_roles=('assistant',)):
    """Yield (tokens, mask) one message at a time, in order.

    mask[i] == 1 means position i is a supervised target. The role header is
    never supervised — the model is not being taught to predict whose turn it
    is — but the closing <|im_end|> is, since that is the only way it learns to
    stop.
    """
    for msg in messages:
        header = [IM_START] + _text(f"{msg['role']}\n")
        body = _text(msg['content']) + [IM_END]
        supervise = int(msg['role'] in train_roles)
        yield (header + body + NEWLINE,
               [0] * len(header) + [supervise] * len(body) + [0] * len(NEWLINE))


def render(messages, train_roles=('assistant',)):
    """Whole conversation -> (token ids, loss mask), two lists of equal length."""
    tokens, mask = [], []
    for seg_tokens, seg_mask in render_turns(messages, train_roles):
        tokens += seg_tokens
        mask += seg_mask
    return tokens, mask


def render_fit(messages, max_len, train_roles=('assistant',)):
    """The longest prefix of whole turns that fits in max_len.

    Cutting a conversation at an arbitrary token would strand an assistant turn
    mid-sentence and teach the model to stop early, but cutting at a turn
    boundary just yields a shorter conversation, which is a perfectly good
    training example. Most over-length conversations are multi-turn, so this
    keeps the great majority of what a hard length filter would throw away.

    Returns (tokens, mask), or (None, None) when not even the opening exchange
    fits.
    """
    segments, total = [], 0
    for segment in render_turns(messages, train_roles):
        if total + len(segment[0]) > max_len:
            break
        segments.append(segment)
        total += len(segment[0])

    # A trailing unsupervised turn is context for a reply that did not fit: it
    # costs window and teaches nothing, so cut back to the last supervised turn.
    while segments and not any(segments[-1][1]):
        segments.pop()
    if not segments:
        return None, None

    tokens = [t for seg in segments for t in seg[0]]
    mask = [m for seg in segments for m in seg[1]]
    return tokens, mask


def render_prompt(user, system=None):
    """Token ids for a single user turn, ending in the assistant header.

    The next token the model samples is the first token of its reply.
    """
    messages = ([{'role': 'system', 'content': system}] if system else [])
    messages.append({'role': 'user', 'content': user})
    tokens, _ = render(messages)
    return tokens + [IM_START] + _text('assistant\n')
