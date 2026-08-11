# TinyLM

This is an educational repo for getting hands-on with implementing, training, and evaluating
language models.

A 50.9M-parameter GPT trained from scratch — tokenizer to sampling — in ~1,000 lines of
readable PyTorch. No `transformers`, no Lightning, no config framework. Two corpora are
set up out of the box: **TinyStories**, which a model this size can actually learn to
write well, and **FineWeb-Edu**, where the same model gets to fluent English and stops
short of anything you would call knowledge.

Everything here runs on one consumer GPU. Both runs trained on a single RTX 3090 at a
sustained **71% MFU** — about 52 TFLOP/s of its 71 TFLOP/s bf16 peak.

## Results

![Training and validation cross-entropy for both runs, log scale](assets/loss-curves.png)

| | TinyStories | FineWeb-Edu |
|---|---|---|
| context length | 512 | 1024 |
| tokens per optimizer step | 131,072 | 262,144 |
| iterations | 8,000 | 20,000 |
| tokens seen | 1.05B (2.21 epochs) | 5.24B (0.55 epochs) |
| best val loss | **1.3186** @ iter 7,500 | **3.3579** @ iter 20,000 |
| train/val gap at the end | 0.038 | 0.022 |
| MFU | 71.0% | 71.3% |
| wall clock | ~1.9h | ~10.3h |

Neither run overfits: the train/val gap stays under 0.04 nats throughout, so both were
still improving when they stopped. Per-eval numbers are in
[assets/metrics-tinystories.csv](assets/metrics-tinystories.csv) and
[assets/metrics-fineweb-edu.csv](assets/metrics-fineweb-edu.csv).


### HellaSwag

The FineWeb-Edu checkpoint, zero-shot over all 10,042 validation examples, split by the
two sources HellaSwag draws from:

| | n | acc | acc_norm | correct-ending loss | margin |
|---|---|---|---|---|---|
| activitynet | 3,243 | 0.3235 | **0.3290** | 3.6615 | **+0.2134** |
| wikihow | 6,799 | 0.2484 | 0.2455 | 3.2759 | −0.0120 |
| **overall** | 10,042 | 0.2727 | 0.2725 | 3.4004 | +0.0608 |
| random baseline | | 0.2500 | 0.2500 | | 0 |

### Training dynamics

![Learning rate schedule, pre-clip gradient norm, and MFU over the FineWeb-Edu run](assets/dynamics-fineweb-edu.png)

Cosine decay from 6e-4 to 6e-5 after 500 warmup iterations. The pre-clip gradient norm is
the trace worth watching: it settles to ~0.45 and stays flat, comfortably under the 1.0
clip, so the clip is not what is holding the run together. MFU sits at 71% for the whole
run with no thermal or memory-pressure drift.

The TinyStories run behaves the same way at half the context length — same schedule shape,
gradient norm settling near 0.31, and the same 71% utilisation:

![Learning rate schedule, pre-clip gradient norm, and MFU over the TinyStories run](assets/dynamics-tinystories.png)

## Samples

**TinyStories**, unconditional, temperature 0.8:

> Once upon a time, there was a little girl named Lily. She had a magical lamp that made
> all her wishes come true. One night, when Lily was looking up at her lamp, she noticed
> it was starting to glow.
>
> Lily was surprised and asked her mom, "Mommy, what is happening?"
>
> Her mom replied, "I don't know, sweetie. It's unknown."
>
> Lily was confused and asked, "What do you mean, mommy?"
>
> Her mom explained, "When things are unknown, just like you and your lamp. They can't be
> found because they have lost forever."

Prompted with *"Once upon a time, there was a little robot who"*:

> …wanted to design something special. He found a big bag of buttons and he thought that
> it would be perfect.
>
> The robot said to the box, "I will make something special with these buttons and it will
> be perfect!" The box replied, "What would it be?" The robot was so excited and he said,
> "I will make a cool surprise!"
>
> The robot worked hard and soon he had designed a beautiful toy robot. The robot's eyes
> were bright and he was so proud of his work.

Coherent characters, consistent tense, and a story that resolves. That is what a 50M model
gets you when the corpus is narrow enough.

**FineWeb-Edu**, same settings — the contrast is the point:

> When you visit the site, you can be sure to make sure that you are answering all your
> questions at the same time. When you go to the site, you can be sure that you are
> answering your questions. […] Why do we need to visit the site? Why can I visit the site
> more often? Why do we need more time?

Grammatical and locally fluent, but it loops and has nothing to say. At 3.36 val loss on
open web text the model has learned English syntax and no world model to speak of.

## Architecture

A decoder-only transformer, pre-norm, in [model.py](model.py):

| | |
|---|---|
| parameters | 50,930,176 |
| layers / heads / d_model | 8 / 8 / 512 |
| feed-forward dim | 2048 |
| context length | 1024 (512 for the TinyStories run) |
| vocab | 50,304 (GPT-2 BPE, padded to a multiple of 64) |
| normalization | pre-norm LayerNorm, no bias |
| positional encoding | learned |
| attention | `F.scaled_dot_product_attention` (FlashAttention) |
| weight tying | input embedding ↔ output head (saves 12.9M params) |

The parameter count excludes the position embedding and counts the tied
embedding once, so it is identical across both context lengths.

Trained in bfloat16 autocast under `torch.compile`, batch 16 × 1024 tokens ×
16 gradient-accumulation steps = **262,144 tokens per optimizer step**. AdamW,
β = (0.9, 0.95), weight decay 0.1 on matmul parameters only — 1-D gains, biases
and embeddings are excluded — and gradient clip 1.0.

## Usage

With [uv](https://github.com/astral-sh/uv):

```bash
uv venv
uv pip install -r requirements.txt
source .venv/bin/activate          # or prefix each command with `uv run`
```

Or with the standard library:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

**Tokenize a corpus** into `data/<dataset>/` — flat `uint16` token shards plus a
`meta.json` manifest. Streams FineWeb-Edu rather than landing 45GB of arrow on disk, and
resumes at the last shard boundary if interrupted:

```bash
python prepare.py tinystories          # ~475M tokens, a few minutes
python prepare.py fineweb-edu          # ~9.7B tokens (sample-10BT), a few hours
```

**Train.** Data and checkpoints are namespaced per dataset, so runs on different corpora
never collide:

```bash
python train.py --dataset tinystories
python train.py --dataset fineweb-edu --resume
tensorboard --logdir out/fineweb-edu/tb
```

Hyperparameters are module-level constants at the top of [train.py](train.py) — edit them
there. On resume the architecture comes from the checkpoint, not from the constants.

**Sample:**

```bash
python sample.py --ckpt out/tinystories/ckpt.pt \
                 --prompt "Once upon a time" --num_samples 3 --temperature 0.8
```

**Supervised fine-tuning** on [smol-smoltalk](https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk)
— the SmolTalk variant built for models under 1B, with the advanced math and function-calling
subsets removed:

```bash
python prepare_sft.py --max-examples 50000     # a slice first; drop the flag for all 460k
python sft.py                                  # starts from out/fineweb-edu/ckpt.pt
python sample.py --ckpt out/smol-smoltalk-sft/ckpt.pt --prompt "Why is the sky blue?"
```

Three things make this different from pretraining:

*Chat tokens cost nothing.* `vocab_size` is GPT-2's 50257 padded to 50304, so ids 50257
and 50258 are rows the embedding matrix already has and has never used. They become
`<|im_start|>` and `<|im_end|>` with no resize and no checkpoint surgery. Tied to `lm_head`,
those rows spent pretraining being pushed down by the softmax denominator, so `sft.py`
resets them to the init distribution before training.

*Loss is masked to assistant turns.* [prepare_sft.py](prepare_sft.py) writes a `uint8` mask
beside every token shard, and [sft.py](sft.py) turns unmasked positions into `-100` for
`ignore_index`. The model is graded on what it should say, not on reciting the question back.

*Batches start at conversation boundaries.* A third shard file holds the `uint64` offset of
every conversation, so a window never opens midway through an assistant turn with the prompt
that produced it out of view. Conversations too long for `block_size` are cut back to the last
complete turn rather than dropped — at 1024 tokens that keeps 97.5% of them instead of 57%.

**Evaluate on HellaSwag:**

```bash
python hellaswag.py --ckpt out/fineweb-edu/ckpt.pt
python hellaswag.py --chunk 8          # lower if 24GB is not enough
```

Downloads the 47MB validation split on first run. Peak memory is the float32 logits, so
`--chunk` bounds it independently of `--batch`.

**Regenerate the figures and CSVs in `assets/`** from the TensorBoard event files:

```bash
python plot.py
```

## Repository layout

| file | |
|---|---|
| [model.py](model.py) | GPT definition, MFU estimator, optimizer grouping |
| [prepare.py](prepare.py) | corpus → resumable `uint16` token shards + manifest |
| [train.py](train.py) | training loop, cosine schedule, TensorBoard, checkpointing |
| [chat.py](chat.py) | ChatML rendering on GPT-2 BPE, conversation → tokens + loss mask |
| [prepare_sft.py](prepare_sft.py) | chat corpus → token / mask / conversation-index shards |
| [sft.py](sft.py) | masked-loss fine-tuning from a pretrained checkpoint |
| [sample.py](sample.py) | autoregressive generation with temperature / top-k |
| [hellaswag.py](hellaswag.py) | zero-shot HellaSwag by length-normalized likelihood |
| [plot.py](plot.py) | TensorBoard events → `assets/` figures and CSVs |