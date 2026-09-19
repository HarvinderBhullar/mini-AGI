# mini-AGI

mini-AGI - is a **continual learning** byte-level language model that assembles its own architecture, trains from scratch on a single 8 GB VRAM GPU, and keeps learning from everything it reads.
It stores its weights as ordinary files on disk and pages them onto the card as it needs them, so the parameter count is bounded by free disk space rather than by VRAM. It grows new capacity while training when it runs short, prunes what nothing asks for, and reads through exactly the same code path it serves on. Targeted at a PC or laptop with at least an 8 GB VRAM GPU on the board.

**NOTE: as of now this is a small toy-level model.** Do not expect a frontier level capabilities. This is rather a small experiment to show, that continual learning from the single stream of data without catastrophic forgetting is possible. Furthermore it is possible on a modest hardware. Which means that almost everyone could train their own version of the model (or simply continue training this one) exactly as they see it fit. And the capabilities would be bounded by the actual hardware, scale and quality of the data available and the amount of time one willing to spend on training the model.

![dashboard](assets/dashboard.png)
Here is how min-run dashboard looks like. The model is pointed to the corpus to constantly read and learn from.

## Motivation

Every language model you can actually own today is a model somebody else trained and then froze. You can fine-tune around the edges of it, but you cannot train one from scratch on your own hardware, and you cannot keep training it on what you do day to day - the moment you try, it forgets what it knew before. The result is that a personal model is always somebody else's model with a thin layer of you on top, and it stops learning the day it ships.

**mini-AGI** model has small enough GPU footprint that it is possible to train end-to-end on one consumer card, and it is built so that training never has to stop. It reads a stream of characters one chunk at a time, takes a gradient step on each, and the same path serves generation. There is no separate fine-tuning regime and no frozen base: reading and being trained are the same event.

Three constraints shape everything else in the design:

- **It has to fit on 8 GB.** Not with quantisation - training needs gradients and optimiser state, which is roughly three times the weights again. So the weights live on disk and only the working set is resident.
- **It has to not forget.** A model that learns continually and overwrites itself is worse than one that does not learn at all.
- **It has to be able to read anything.** The alphabet is the 256 byte values, so there is no tokenizer to fit and no data type that needs a new vocabulary.

The model is genuinely yours: trained on your hardware, on your data, that keeps learning from every conversation you have with it, and that nobody else can take it away or switch it off. 

<!--**Watch this video for a detailed explanation**
[here will be the link to the video when its ready]()-->

## How the architecture works

Characters (bytes) does not pass through a fixed stack of layers as it would be in a traditional LLM. Instead, it passes through **two dense prelude blocks** and then through **one recurrent block applied up to 24 times**, each application choosing its own experts from a shared pool. The latent state between applications is never decoded - it is merged with the embedded input by an adapter each time round, so the loop cannot drift away from the text it is reading.

Three distinct blocks, up to 26 block-applications per character.

- **Adaptive depth.** A halting head scores every character at every row, and the character stops as soon as another row would not change the answer. Easy characters take one row, hard ones take many. This is the PonderNet recipe: while training, every depth is computed and weighted by its halting probability, so the halting head learns through those weights.
- **Routing per block-application, not per character.** Each of the 26 applications picks its own top-8 experts, so one character touches far more of the pool than "top-8" suggests, and the same expert can be selected several times at different depths. What varies is *which* eight at each point.
- **No expert is assigned a subject.** There are no labels anywhere. Soft top-k routing distributes capability across the pool by itself, and a character can combine fragments from several experts. The cost is that capabilities share parameters and so *can* interfere.

Positions are rotary and carry no learned parameters, which is why the context window can be extended by continued training rather than by re-initialising anything.

## How paging works

Every expert is a file on disk holding its weights and its Adam moments. Above disk sit two caches and a working set:

| | key | what it is |
|---|---|---|
| disk | — | every expert the model has; bounded by free space |
| RAM | `ram_cache` | recently wanted experts, least-recently-used evicted |
| VRAM | `resident` | the working set - what a character may route through |

Before every chunk the model is asked what the text about to be read wants, and the answer becomes the working set. Demand is scored on the hidden states the call sites actually routed on while reading the previous chunk - an embedding carries no context, so scoring on raw embeddings would have every subject asking for the same experts.

Two rules the project holds to:

- **Adam's moments travel with the expert.** They belong to the expert, not to the slot of VRAM it happened to occupy. Leaving them behind would hand one expert's momentum to whatever took its place, and training would carry on looking healthy while every swapped expert inherited a stranger's history.
- **An expert already on the card stays in the slot it is in.** Demand comes back sorted, so the order churns while the set itself barely moves. Matching by identity rather than by position is what keeps the number of loads equal to how much the set really changed.

Because the choice is made from the previous chunk, it cannot see the text it is about to predict. What keeps the working set from churning on noise is hysteresis - a candidate has to beat a resident by `margin` to displace it, and a newcomer is safe for `dwell_chars` of reading.

## How growth and pruning work

The pool grows when it is short of capacity and shrinks when parts of it stop being asked for.

New experts are added on speculation, at a small gate so they change almost nothing, and kept only if something goes on asking for them. A new expert is built by **recombination** - whole hidden units taken from several existing experts - because a clone of one parent is not novel enough to be worth routing to, and a random expert computes nothing worth routing to. What works is novelty assembled from trained parts.

Growth is refused unless every brake agrees:

- **room** - disk and VRAM can take it
- **used** - the capacity already added is being asked for
- **earning** - the previous cohort survived its trial
- **fits** - not too many experts are already inside their trial
- **honest** - train and held-out have not separated

**Dead means unaddressed.** Both the growth brake and the pruner read how long it has been since anything asked for an expert, and never its gate. This is the single most useful finding in the repository: the gate is not merely uninformative here, it is anti-predictive. The smallest gates belong to the *busiest* experts - one that behaves as a sink, chosen constantly and contributing little per character, reads as dead on a gate test, while a high-gate expert nothing has wanted in hundreds of thousands of segments reads as alive.

A new expert is safe for a full survival window no matter what, so it cannot be judged before it has had a chance to be chosen. When the model grows an expert a new file appears; when it prunes one, that file is deleted. 

## How continual learning works

Training on a single stream, one subject at a time, is the classic recipe for catastrophic forgetting. Reading half a million characters of chess at the experts' own learning rate takes the other seven subjects from 1.12 to 3.73 nats.

**The trunk learning rate is the mechanism.** The trunk - embeddings, attention, routers, the halting head - is the part every character passes through, and it carries 97.6% of the squared gradient norm. Running it at 0.1x the experts' rate takes forgetting from +2.2300 to +0.0067 nats, which is 99.84% of progress retained against chance.

| configuration | unread subjects | retained vs chance |
|---|---|---|
| working set frozen, trunk LR = expert LR | +2.5871 | 42.88% |
| swapping, trunk LR = expert LR | +2.2300 | 50.68% |
| **swapping, trunk at 0.1x - what the run uses** | **+0.0067** | **99.84%** |
| *control: all seven subjects read* | *-0.0077* | *-* |

![Forgetting under three configurations](assets/mitigations.png)

**This is the measurement the whole design rests on.** The model reads 524,000 characters of chess and nothing else, at batch 1, and the y-axis on the left is what happened to the **seven subjects it did not read** - zero means nothing was forgotten, up means worse. Three lines, one variable each. Two of them climb to +2.2 and +2.6 nats, which is the model losing most of what it knew. The third, at a trunk learning rate one tenth of the experts', never leaves the floor: **+0.0067 nats after half a million characters of a single subject**.

The grey dashed line is the control - the same probe with all seven subjects read, where forgetting is impossible by construction. 

The right panel converts the same three arms into progress retained against chance. The gap between 50.68% and 99.84% is one number in a config file.

Two readings matter here, and the second one corrects this project's own earlier account:

- **The expert pool is not what prevents forgetting.** Freezing the working set - removing the one property that makes the pool a pool - costs only 0.3571 nats, 13.8% of the effect. In that arm 93 of 136 experts received no gradient at all and the model still collapsed. Preserving most of the weights is not sufficient.
- **The damage is displacement, not destruction.** Damage the model badly and then read everything again: three quarters of it comes back in 131,000 characters, against the ~50M characters it took to learn those subjects the first time. Knowledge that had to be relearned does not come back 380x faster. "Catastrophic" describes how it looks at the bottom of the curve, not what happened to the weights.

![Every subject during a massed read, and how much of the pool was touched](assets/probe_massed.png)

**What that same read looks like from the inside.** This is the working configuration - trunk at 0.1x - during the identical 524,000-character chess probe. On the left, every subject plotted against where it started. Chess, the subject actually being read, improves by 0.013 nats. The shaded band is the range across the seven subjects that are *not* being read, and it stays within ±0.02 nats for the whole probe: **learning one thing did not cost anything measurable anywhere else**. That is the claim in the first paragraph of this README, drawn rather than asserted.

The right panel is why that is possible at all. Over the whole probe only **54 of 136 experts received any gradient** - 60% of the model was structurally untouched, because routing never selected it. This is the pool doing exactly what a pool is for: confining an update to the part of the model that the text actually addressed.

**The learning rate is not scheduled.** A cosine schedule asserts that the run ends, which for a model that reads continually is false. Instead a controller watches held-out loss and moves the rate in both directions: clear improvement buys a little more, no evidence eases it down, and a confirmed jump in held-out steps it back up. 

## Reading your own files

This is the shortest path to a model that knows something you care about.

```bash
python3 train.py read ~/notes                     # a dry read - nothing kept
python3 train.py read src/ docs/ --passes 3 --save
```

Point it at files or directories. There is nothing to prepare: the alphabet is the 256 byte values, so a file is already written in the only vocabulary the model has. Directories are walked, binaries are skipped by sampling their contents rather than trusting the extension, and each file is read from its beginning to its end because a document has an order.

It is the same path training uses: same chunking, same cache, same gradient step.

| flag | |
|---|---|
| `--passes N` | read the whole set N times |
| `--save` | keep what it learned; without it `weights/` is untouched |
| `--lr` | default 5e-5, below a training run: reading should adjust the model, not overwrite it |
| `--mix ""` | skip the before/after scoring |

Two defaults worth knowing. **Nothing is saved without `--save`**, so a read is a dry run until you decide otherwise. And it scores the held-out mixture before and after, then says plainly if reading your files cost the model ground elsewhere - the forgetting question measured per-read rather than assumed away.

## Benchmarks

The numbers below are for tracking purposes and move as the run continues. Held-out loss is reported with its standard error, and the size of the evaluation is what sets that error - a difference smaller than it is the instrument rather than a result.

There is a second variance underneath these figures. The same configuration run twice lands about 0.014 apart, because the expert dispatch is not deterministic on CUDA. **Treat about 0.03 as the threshold for a real difference**, not the error bar printed beside one score.

**Where the model is** (236.8M characters read, 143 experts):

| | nats/char | bits/byte |
|---|---|---|
| **held-out, all eight subjects** | **0.8827** ± 0.0360 | **1.2735** |
| train | 0.7355 | 1.0611 |

**Held-out loss per subject:**

| Subject | nats/char | bits/byte |
|---|---|---|
| `chess` | 0.563 | 0.812 |
| `stories` | 0.664 | 0.958 |
| `arithmetic` | 0.675 | 0.974 |
| `code` | 0.779 | 1.124 |
| `reasoning` | 0.865 | 1.248 |
| `chat` | 0.889 | 1.283 |
| `chat_hermes` | 1.257 | 1.813 |
| `wikipedia` | 1.369 | 1.975 |

### Data Scaling

![Data scaling against published byte-level and subword models](assets/scaling.png)

Every point on this chart is a model with a **published bits-per-byte** - the only loss unit that survives a change of tokenizer, which is why a byte-level model can be put beside GPT-3 at all. 

Three held-out sets are involved - PG19, Pile-CC and this project's own mixture - so the vertical positions are not strictly comparable across colours. MambaByte-353M is the closest like-for-like, same parameter class and essentially the same FLOPs per byte, and it read **127x more data than this model has**. Transformer-320M read 339x more. 

**The results so far are promising.** The red line is the fitted power law, `L ∝ D^-0.221` with R² 0.94 over every point past the warmup - a clean, healthy exponent, between Kaplan's 0.095 and Chinchilla's 0.28, and it has held for more than a decade of data. Later windows of the run are steeper still, which is why the projection carries a band out to α = 0.35 rather than a single line.

Read straight off that trend, and remembering that the target is this model's own mixture rather than PG19:

| held-out | bytes needed | days at ~780 char/s |
|---|---|---|
| 1.00 BPB | 0.91B | ~13 |
| 0.93 BPB | 1.26B | ~19 |
| 0.80 BPB | 2.50B | ~37 |

Those are weeks of reading on one laptop GPU, not years, and all of them sit inside a single pass of the 7.87B-character corpus.

The right panel shows which subjects are still moving. Code, chat, stories and reasoning are the steep ones; wikipedia and chat_hermes carry the most loss and have the shallowest slopes, which is the honest counterweight - the expensive domains are not the fastest ones.

## Running it

1. Make sure you have a CUDA-capable GPU with at least 8 GB of VRAM, and Python 3.10 or newer. The reference machine is an RTX 3070 Laptop GPU with 8 GB.
2. Clone the repository:
    ```bash
    git clone <repository-url>
    cd mini-AGI
    ```
3. Install the dependencies:
    ```bash
    pip install torch numpy pyyaml matplotlib      # the model, and its graphs
    pip install flask                              # serve.py
    pip install tokenizers chess zstandard         # building corpora
    pip install scipy                              # a few of the analysis tools
    ```
    PyTorch has to match your CUDA version - see [the PyTorch install page](https://pytorch.org/get-started/locally/). The reference environment is torch 2.6.0+cu124 with numpy 1.24.4. Only the first line is needed to train.
4. Build a corpus, or point the model at your own files. Each `corpora` command writes `train.bin`, `val.bin` and `meta.json` into a `data_*` directory:
    ```bash
    python3 -m corpora code                 # Python from the local filesystem
    python3 -m corpora arithmetic
    python3 -m corpora chat
    python3 -m corpora chess
    ```
5. Create a model. Everything about its shape comes from `config.yaml`:
    ```bash
    python3 tools/init_weights.py
    ```
6. Start reading:
    ```bash
    python3 train.py read data/train --save --weights-dir weights \
        --sample-every 10 --sample-log runs/samples.txt
    ```
7. Serve it:
    ```bash
    python3 serve.py --port 8080            # then open http://127.0.0.1:8080
    ```

The run writes a sample log, redraws its graphs as it goes, and checkpoints every few minutes. It is meant to be left alone for days.

### Everything else

```bash
python3 -m minagi.store weights                    # what the model is right now
python3 tools/export_history.py                    # samples.txt -> JSONL + CSV
python3 tools/plot_dashboard.py                    # the nine-panel run dashboard

python3 -m benchmarks report                       # all benchmarks
python3 -m benchmarks code --ckpt weights
python3 -m benchmarks arithmetic --ckpt weights
python3 -m benchmarks chess --ckpt weights

python3 -m experiments capacity --mode adapt       # what to spend VRAM on
python3 -m experiments forgetting                  # streaming vs mixed batches
python3 -m experiments trunk_lr                    # how slowly the trunk should learn

python3 train.py stream --steps 140000 --lr 2e-4   # the packed-corpus path
python3 train.py batch --use-pool --steps 2000     # the fixed-window regime
```

Every tool takes `--ckpt weights` - the directory is the model, and there are no `.pt` files to keep track of.

## Initialization

A fresh model starts small and grows into its shape. The context window begins at `model.context_start` and extends one character at a time, but only when the model is still getting something out of the far end of the window it already has. The expert pool begins at `pool.experts` and grows from there.

This means the first hours of a run look nothing like the rest of it. Loss falls fast, the pool churns, the window is short, and the learning-rate controller has not gathered enough evaluations to act. None of that is a problem to fix.

If a run diverges, it repairs itself: when held-out exceeds the best by more than `--revert-factor` (default 1.5x) the run reloads `weights/`, halves the learning rate, pulls the context back and continues. After `--max-reverts` it stops rather than thrash.

## Layout

```
minagi/          the model. no command lines here.
  config.py        reading config.yaml, which building and training both use
  precision.py     what the model computes in, and how moments are stored
  tokenizer.py     bytes in, bytes out - 256 values plus structural markers
  model.py         the transformer: RMSNorm, rotary positions, SwiGLU, flash attention
  decode.py        how a character is chosen, without a random number generator
  ingest.py        turning a pile of files into something to read
  pool.py          the expert pool, and the rules by which it grows and shrinks
  paged.py         the same pool spread over disk, RAM and VRAM
  recur.py         latent recurrence with adaptive depth
  stream.py        reading a corpus behind a KV cache, one chunk at a time
  corpus.py        the fixed-window view, for the batch trainer and benchmarks
  store.py         the weights directory, which IS the model
  optim.py         optimisers for the trunk, and the gradient noise scale
  plasticity.py    the learning rate, governed by held-out loss
  schedule.py      the fixed cosine schedule, for the batch trainer
  live.py          serving a model that is being trained underneath
  report.py        the model reading statistics off its own weights
  notify.py        telling someone when a long run needs attention

train.py         read | stream | batch | ponder-probe
serve.py         local web UI
config.yaml      the settings worth changing
benchmarks/      python3 -m benchmarks <code|arithmetic|chess>
corpora/         python3 -m corpora <code|arithmetic|chat|chess>
experiments/     python3 -m experiments <forgetting|trunk_lr|freeze|conflict|replay>
tools/           plotting, run history, probes, animations
weights/         one file per expert. this directory is the model.
runs/            sample logs, measurements, and the figures the project cites
```

### The weights directory is the model

```
weights/
  manifest.json     what exists, its shape, and where it came from
  core.npz          embeddings, attention, norms, adapter, halting head
  routers.npz       the gate, the segment router, one row per expert per site
  optim.npz         Adam moments for the trunk and the routers
  experts/          one file per expert: w1, w3, w2 and its own Adam moments
    e00000.npz ...
```

Training resumes from it - weights, Adam moments and step count - and advances it whenever a run improves on what is there, so a session run only to check something still contributes if it finds anything. The directory holds the **best** state the model has reached, not the most recent one. Writes are atomic: every file is written to a `.tmp` and renamed, so an interrupted save cannot leave a half-written weight behind.

## The model

Byte level - vocabulary 265: the 256 byte values plus 9 structural markers (`<think>…</think>` scratchpad, `<user>/<bot>` turns, `<g>` for games, and end-of-text). Context 4,096.

| | |
|---|---|
| body | RMSNorm, RoPE, SwiGLU, flash attention via `scaled_dot_product_attention` |
| depth | 3 distinct blocks, up to 26 block-applications per character |
| recurrence | one weight-shared block applied up to 24 times; the latent is never decoded |
| halting | PonderNet - each character halts independently, so hard ones get more depth |
| routing | top-8 experts per block-application, chosen per character |
| paging | 32 experts resident on the card; the rest live on disk |

The parameter count moves, because the pool grows and prunes itself while training. `python3 -m minagi.store weights` prints what it is now. At the time of writing:

```
core        8.27M   embeddings, attention, norms, adapter, halting head
routers     0.15M   one row per expert per call site, plus depth embeddings
experts   459.3M    146 x 3.15M each  (3 x 512 x 2048)
--------------------
total     467.7M
```

**VRAM is set by the working set, not by the pool.** Only 32 experts are resident at a time - about 101M parameters of the 468M - which is why the pool can keep growing on an 8 GB card. Per byte the model costs about 2.4 GFLOPs to train, which puts it in the same compute class as a dense 400M byte-level transformer.

## AI usage

This project was assisted by "Claude Opus 5" model. The model did implemented most of code of this project, verified and debugged it when it was necessary. The model was searching for published papers related to the problems that the project were trying to solve, build tests and experiments, and help with brainstorming the complex problems that arose along the way. The animations, graphs and other media you see here are all done by Claude as well form the real data traces. While I myself provided main ideas, steering, intuition, rejections when thing went in a wrong direction, code monitoring and verification, as well as decisions and strong opinions of how everything should be wired together and work in principle. Documentation was written in tandem.  

## Acknowledgments

[PyTorch](https://pytorch.org/) does the arithmetic, [NumPy](https://numpy.org/) holds the weights on disk, and [Matplotlib](https://matplotlib.org/) draws every graph.

The parts the model is built out of:

[Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538) - Shazeer et al., 2017. The entire expert pool, and the load-balancing auxiliary loss.  
[Switch Transformers](https://arxiv.org/abs/2101.03961) - Fedus et al., 2021. The capacity-based batched dispatch, which is what lets the pool run as three matrix multiplies.  
[PonderNet: Learning to Ponder](https://arxiv.org/abs/2107.05407) - Banino et al., 2021. The adaptive depth mechanism.  
[RoFormer: Rotary Position Embedding](https://arxiv.org/abs/2104.09864) - Su et al., 2021. Why the context window can grow by continued training.  
[GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) - Shazeer, 2020. SwiGLU.  
[Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467) - Zhang & Sennrich, 2019.  
[FlashAttention](https://arxiv.org/abs/2205.14135) - Dao et al., 2022. Reached through PyTorch's `scaled_dot_product_attention`.  
[Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101) - Loshchilov & Hutter, 2017. AdamW.  
[Training Deep Nets with Sublinear Memory Cost](https://arxiv.org/abs/1604.06174) - Chen et al., 2016. Gradient checkpointing, which on 8 GB is not optional.

[ZeRO-Offload](https://arxiv.org/abs/2101.06840) - Ren et al., 2021, and [ZeRO-Infinity](https://arxiv.org/abs/2104.07857) - Rajbhandari et al., 2021. Training a model larger than the card it sits on is not a new capability.  
[Dynamic Mixture of Experts Against Severe Distribution Shifts](https://arxiv.org/abs/2511.18987) - Kim et al., 2025. Adds experts to a live MoE, and reports the failure this project spent a week fixing.

[Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556) - Hoffmann et al., 2022. Chinchilla, and the ratio any efficiency claim has to be tested against.  
[The Pile](https://arxiv.org/abs/2101.00027) - Gao et al., 2020. Bits per UTF-8 byte, chosen there for invariance to tokenisation.  
[Transformer-XL](https://arxiv.org/abs/1901.02860) - Dai et al., 2019, and [Compressive Transformers](https://arxiv.org/abs/1911.05507) - Rae et al., 2019. The character-level benchmarks to aim at.  
[An Empirical Model of Large-Batch Training](https://arxiv.org/abs/1812.06162) - McCandlish et al., 2018. The gradient noise scale.  
[The AdEMAMix Optimizer](https://arxiv.org/abs/2409.03137) - Pagliardini et al., 2024. Implemented for the trunk and available, though at the paper's settings it hurt this model and it is not the default.

The corpus: [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories), [OpenHermes-2.5](https://huggingface.co/datasets/teknium/OpenHermes-2.5), [OpenThoughts-114k](https://huggingface.co/datasets/open-thoughts/OpenThoughts-114k) and [the Lichess open database](https://database.lichess.org/). Wikipedia and the source-code portion come from public dumps and public repositories.

## Citation

If you use this project in your research or work, please cite it as:

```bibtex
@software{Borsky_mini_AGI_2026,
  author = {Borsky, Alexey},
  month = {9},
  title = {{mini-AGI: A Continually Learning Byte-Level Language Model}},
  url = {https://github.com/volotat/mini-AGI},
  version = {1.0.0},
  year = {2026}
}
```
