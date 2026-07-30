"""Render the README figures and metric tables from the TensorBoard event files.

    python plot.py                       # all datasets found under out/
    python plot.py --dataset fineweb-edu

Reads out/<dataset>/tb/, writes assets/. The event files are the only complete
record of a run — the terminal logs were scrollback and lost their early
iterations — so everything published is derived from them rather than re-typed.

One PNG per figure, on a white surface, embedded in the README directly.
"""

import argparse
import csv
import glob
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from model import GPT, GPTConfig

OUT_ROOT = 'out'
ASSETS = 'assets'

FLOPS_PROMISED = 71e12   # RTX 3090 bf16 dense, fp32 accumulate (GA102)

# Slots 1 and 2 of the categorical palette. Validated as a pair against this
# surface: lightness band, chroma, CVD separation (worst adjacent dE 24.7
# protan) and >=3:1 contrast all pass.
THEME = dict(
    surface='#ffffff', ink='#0b0b0b', secondary='#52514e', muted='#898781',
    grid='#e1e0d9', axis='#c3c2b7', train='#2a78d6', val='#eb6834')

TITLE = {'fineweb-edu': 'FineWeb-Edu', 'tinystories': 'TinyStories'}


def event_files(dataset):
    paths = (glob.glob(os.path.join(OUT_ROOT, dataset, 'tb', '*', '*'))
             + glob.glob(os.path.join(OUT_ROOT, dataset, 'tb', 'events*')))
    return sorted(p for p in paths if not os.path.isdir(p))


def load(dataset):
    """Merge every event file for a dataset into {tag: [(step, value), ...]}.

    A run that was started and killed leaves a stub event file beside the real
    one; merging by step keeps whichever wrote a given step last instead of
    making the caller guess which file is the good one.
    """
    merged = {}
    for path in event_files(dataset):
        acc = EventAccumulator(path, size_guidance={'scalars': 0})
        acc.Reload()
        for tag in acc.Tags()['scalars']:
            merged.setdefault(tag, {}).update({s.step: s.value for s in acc.Scalars(tag)})
    return {tag: sorted(steps.items()) for tag, steps in merged.items()}


def load_config(dataset):
    """Recover the run's hyperparameters from the |param|value| table train.py
    writes into the TEXT tab, so a figure never has to be told what it is plotting."""
    conf = {}
    for path in event_files(dataset):
        acc = EventAccumulator(path, size_guidance={'tensors': 0})
        acc.Reload()
        for tag in acc.Tags()['tensors']:
            for event in acc.Tensors(tag):
                for line in event.tensor_proto.string_val[0].decode().splitlines():
                    cells = [c.strip() for c in line.strip('|').split('|')]
                    if len(cells) == 2 and cells[0] not in ('param', '---'):
                        conf[cells[0]] = cells[1]
    return conf


def mfu(series, conf):
    """
    Recompute utilisation from measured ms/iter rather than trusting perf/mfu.
    """
    need = ('vocab_size', 'block_size', 'n_layer', 'n_head', 'n_embd', 'd_ff')
    if not all(k in conf for k in need) or 'tokens_per_iter' not in conf:
        return series.get('perf/mfu', [])
    cfg = GPTConfig(**{k: int(conf[k]) for k in need})
    n = GPT(cfg).num_params(non_embedding=False)
    flops_per_token = (6 * n + 12 * cfg.n_layer * cfg.n_head
                       * (cfg.n_embd // cfg.n_head) * cfg.block_size)
    tokens_per_iter = int(conf['tokens_per_iter'])
    return [(step, flops_per_token * tokens_per_iter / (ms / 1000) / FLOPS_PROMISED)
            for step, ms in series.get('perf/ms_per_iter', [])]


def xy(series, tag):
    pts = series.get(tag, [])
    return [p[0] for p in pts], [p[1] for p in pts]


def median(points, skip_steps=()):
    """Steady-state value of a trace. Eval iterations are excluded by the caller:
    their timing is dominated by estimate_loss, not by the training step."""
    values = sorted(v for step, v in points if step not in skip_steps)
    return values[len(values) // 2] if values else 0.0


def ema(values, alpha=0.05):
    """Smoothing for the per-iteration traces, which are far too noisy to read raw."""
    out, acc = [], None
    for v in values:
        acc = v if acc is None else alpha * v + (1 - alpha) * acc
        out.append(acc)
    return out


def style(ax, t, xlabel, ylabel, title=None):
    ax.set_facecolor(t['surface'])
    ax.grid(True, color=t['grid'], linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for edge in ('top', 'right'):
        ax.spines[edge].set_visible(False)
    for edge in ('bottom', 'left'):
        ax.spines[edge].set_color(t['axis'])
        ax.spines[edge].set_linewidth(1.0)
    ax.tick_params(colors=t['muted'], labelsize=9, length=0)
    ax.set_xlabel(xlabel, color=t['secondary'], fontsize=9.5)
    ax.set_ylabel(ylabel, color=t['secondary'], fontsize=9.5)
    if title:
        ax.set_title(title, color=t['ink'], fontsize=11.5, fontweight='bold',
                     loc='left', pad=10)


LOG_TICKS = (1, 1.2, 1.5, 2, 3, 4, 5, 7, 10, 15, 20)


def log_yaxis(ax, values):
    """Loss falls by ~8 nats in the first few hundred iterations. On a linear axis
    that start owns the whole plot and the remaining 19,500 iterations — the part
    anyone actually wants to read — collapse onto the bottom gridline."""
    ax.set_yscale('log')
    lo, hi = min(values), max(values)
    ticks = [v for v in LOG_TICKS if lo <= v <= hi]
    # Extend past the floor so the tail of the curve — where the run actually
    # ends up — has a labelled gridline under it rather than dangling in blank space.
    below = [v for v in LOG_TICKS if v < lo]
    if below:
        ticks.insert(0, below[-1])
    ax.set_ylim(ticks[0] if below else lo * 0.97, hi * 1.05)
    ax.set_yticks(ticks)
    ax.set_yticklabels([f'{v:g}' for v in ticks])
    ax.minorticks_off()


def figure(name, build, size):
    t = THEME
    fig = plt.figure(figsize=size, facecolor=t['surface'])
    build(fig, t)
    path = os.path.join(ASSETS, f'{name}.png')
    fig.savefig(path, dpi=160, facecolor=t['surface'], bbox_inches='tight', pad_inches=0.3)
    plt.close(fig)
    print(f'  wrote {path}')


def loss_figure(runs):
    """One panel per run: the corpora differ in difficulty, so they never share a
    y-axis — small multiples rather than two scales on one chart."""
    def build(fig, t):
        axes = fig.subplots(1, len(runs))
        axes = [axes] if len(runs) == 1 else list(axes)
        for ax, (dataset, series) in zip(axes, runs):
            span = []
            for tag, key, name in (('loss/train', 'train', 'train'),
                                   ('loss/val', 'val', 'val')):
                x, y = xy(series, tag)
                span += y
                # The final value rides in the legend rather than as an end-of-line
                # annotation: train and val sit ~0.02 apart, so the two labels
                # would print on top of each other.
                ax.plot(x, y, color=t[key], linewidth=2.0, zorder=3,
                        marker='o', markersize=3.5, markevery=max(1, len(x) // 12),
                        label=f'{name}   {y[-1]:.4f}' if y else name)
            final = dict(zip(*xy(series, 'loss/val')))
            last = max(final) if final else 0
            log_yaxis(ax, span)
            style(ax, t, 'iteration', 'cross-entropy loss',
                  f'{TITLE.get(dataset, dataset)}  ·  {last:,} iters  ·  '
                  f'val {final.get(last, float("nan")):.4f}')
            ax.margins(x=0.04)
            leg = ax.legend(frameon=False, fontsize=9.5, loc='upper right',
                            handlelength=1.6, labelspacing=0.6)
            for text in leg.get_texts():
                text.set_color(t['secondary'])
        fig.tight_layout(w_pad=3.5)
    figure('loss-curves', build, (6.2 * len(runs), 4.2))


def dynamics_figure(dataset, series):
    """LR schedule, pre-clip gradient norm and MFU — the three traces that say
    whether the run was healthy rather than how good the model got."""
    panels = [
        ('lr', 'learning rate', 'train', False),
        ('grad_norm', 'grad norm (pre-clip)', 'val', True),
        ('mfu', 'model FLOPs utilisation (excl. eval steps)', 'train', True),
    ]
    panels = [p for p in panels if p[0] in series]
    # An eval iteration runs estimate_loss before the timer stops, so it clocks
    # ~11.5s against a normal 1.86s and lands at ~11% MFU. Those points measure
    # the instrumentation, not the training step, and on the raw trace they draw
    # a vertical bar every 500 iterations straight through the plot.
    eval_steps = {s for s, _ in series.get('loss/val', [])}

    def build(fig, t):
        axes = fig.subplots(1, len(panels))
        axes = [axes] if len(panels) == 1 else list(axes)
        for ax, (tag, title, key, smooth) in zip(axes, panels):
            x, y = xy(series, tag)
            if tag == 'mfu':
                x, y = map(list, zip(*[(a, b) for a, b in zip(x, y)
                                       if a not in eval_steps]) or ((), ()))
            if smooth:
                # Raw trace kept underneath: the spread is the point, the EMA
                # only makes the trend legible on top of it.
                ax.plot(x, y, color=t[key], linewidth=0.7, alpha=0.25, zorder=2)
                ax.plot(x, ema(y), color=t[key], linewidth=2.0, zorder=3)
            else:
                ax.plot(x, y, color=t[key], linewidth=2.0, zorder=3)
            if tag == 'mfu':
                ax.set_ylim(0, 1)
                ax.yaxis.set_major_formatter(lambda v, _: f'{v * 100:.0f}%')
            style(ax, t, 'iteration', '', title)
        fig.tight_layout(w_pad=3.0)
    figure(f'dynamics-{dataset}', build, (4.6 * len(panels), 3.6))


def write_csv(dataset, series):
    """The eval-step table: the numbers behind the curves, in a form a reader can
    diff and re-plot without TensorBoard."""
    steps = sorted({s for tag in ('loss/train', 'loss/val') for s, _ in series.get(tag, [])})
    train, val = dict(series.get('loss/train', [])), dict(series.get('loss/val', []))
    path = os.path.join(ASSETS, f'metrics-{dataset}.csv')
    with open(path, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['iter', 'train_loss', 'val_loss', 'gap'])
        for s in steps:
            if s in train and s in val:
                w.writerow([s, f'{train[s]:.4f}', f'{val[s]:.4f}', f'{val[s] - train[s]:.4f}'])
    print(f'  wrote {path} ({len(steps)} eval steps)')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset', action='append', help='repeatable; default is all of out/')
    args = ap.parse_args()

    datasets = args.dataset or sorted(
        d for d in os.listdir(OUT_ROOT)
        if os.path.isdir(os.path.join(OUT_ROOT, d, 'tb')))
    os.makedirs(ASSETS, exist_ok=True)

    runs = []
    for dataset in datasets:
        series = load(dataset)
        if 'loss/val' not in series:
            print(f'{dataset}: no loss/val scalars, skipping')
            continue
        series['mfu'] = mfu(series, load_config(dataset))
        stored, fixed = series.get('perf/mfu', []), series['mfu']
        evals = {s for s, _ in series['loss/val']}
        if stored and fixed and abs(median(stored, evals) - median(fixed, evals)) > 0.02:
            print(f'{dataset}: stored perf/mfu {median(stored, evals) * 100:.1f}% is wrong '
                  f'(bad peak-FLOPs constant); recomputed {median(fixed, evals) * 100:.1f}%')
        print(f'{dataset}: {len(series)} tags, {len(series["loss/val"])} evals')
        write_csv(dataset, series)
        dynamics_figure(dataset, series)
        runs.append((dataset, series))

    if runs:
        loss_figure(runs)


if __name__ == '__main__':
    main()
