"""Latent recurrence with adaptive depth.

The same blocks are applied repeatedly to a hidden state that is never decoded,
so a small number of distinct blocks becomes a much larger number of block
applications. Each character decides for itself how many passes it needs
(PonderNet-style halting) and stops when another pass would not change the
answer.

    n_prelude + max_steps * (n_recur + n_coda)   block applications
    n_prelude + n_recur + n_coda                 distinct blocks

`RecurConfig.n_layer_effective` is the authority on that arithmetic; anything
that quotes it should read it rather than recompute it.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import Config, Block, RMSNorm, build_rope
from .decode import pick_next
from .precision import amp


@dataclass
class RecurConfig(Config):
    # a shared, self-growing expert pool with no assigned specialities
    use_pool: bool = False
    pool_experts: int = 64
    pool_d_ff: int = 192          # hidden units inside one expert
    pool_depth: int = 1           # SwiGLU blocks stacked inside one expert
    pool_top_k: int = 4
    # the largest share of a batch one expert may take, as a
    # multiple of its fair share; overflow is dropped. 0 = no bound
    pool_capacity_factor: float = 1.5
    pool_max: int = 1024
    pool_aux: float = 0.01
    n_prelude: int = 1        # blocks before the loop
    n_recur: int = 2          # blocks inside the loop (weight-shared)
    n_coda: int = 1           # blocks after the loop, run per step
    max_steps: int = 4          # ceiling at inference
    min_steps: int = 1
    train_steps_mean: float = 0.0   # 0 = always run max_steps while training
    bptt_window: int = 4            # backprop through the last N passes only
    ponder_beta: float = 0.01
    halt_prior: float = 0.4   # geometric prior on depth; mean ~ 1/halt_prior
    halt_thresh: float = 0.9  # inference: halt once cumulative exceeds this

    @property
    def n_layer_effective(self):
        return self.n_prelude + self.max_steps * (self.n_recur + self.n_coda)


class RecurCoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.prelude = nn.ModuleList([Block(cfg) for _ in range(cfg.n_prelude)])
        self.recur = nn.ModuleList([Block(cfg) for _ in range(cfg.n_recur)])
        self.coda = nn.ModuleList([Block(cfg) for _ in range(cfg.n_coda)])

        self.pool = None
        if cfg.use_pool:
            from .pool import SharedPool, PooledMLP
            self.pool = SharedPool(cfg.d_model, cfg.pool_experts,
                                   cfg.pool_d_ff, cfg.pool_max,
                                   depth=getattr(cfg, "pool_depth", 1))
            # every recurrent block routes into the SAME pool, so a fragment
            # learned at one depth or one pass is reachable from all of them
            site = 0
            for blk in list(self.recur) + list(self.coda):
                blk.mlp = PooledMLP(
                    self.pool, cfg.d_model, cfg.pool_top_k, site,
                    capacity_factor=getattr(
                        cfg, 'pool_capacity_factor', 1.5))
                site += 1
        # merges the running latent state with the original embedded input, so
        # the loop cannot drift away from what it is actually reading
        self.adapter = nn.Linear(2 * cfg.d_model, cfg.d_model, bias=False)
        self.ln_f = RMSNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.halt = nn.Linear(cfg.d_model, 1)
        if cfg.tie_embeddings:
            self.head.weight = self.tok_emb.weight

        cos, sin = build_rope(cfg.block, cfg.d_model // cfg.n_head,
                              cfg.rope_theta, torch.device("cpu"))
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init)
        if self.pool is not None:
            from .pool import PooledMLP
            with torch.no_grad():
                self.pool.gate.fill_(1.0)
                for m in self.modules():
                    if isinstance(m, PooledMLP):
                        m.depth_emb.zero_()
        depth = cfg.n_prelude + cfg.n_recur + cfg.n_coda
        for name, p in self.named_parameters():
            if name.endswith("proj.weight") or name.endswith("w2.weight"):
                nn.init.normal_(p, 0.0, 0.02 / math.sqrt(2 * depth))
        with torch.no_grad():
            # [I | I]: the adapter starts as the sum of the latent and the
            # embedded input, so the first pass sees the text itself and every
            # later pass accumulates on top of it. The input half is what keeps
            # the loop anchored - without it the latent, which starts at zero,
            # would circulate without the text ever entering.
            self.adapter.weight.zero_()
            eye = torch.eye(cfg.d_model)
            self.adapter.weight[:, :cfg.d_model].copy_(eye)
            self.adapter.weight[:, cfg.d_model:].copy_(eye)
            # start biased toward pondering rather than halting instantly
            self.halt.bias.fill_(-2.0)
            self.halt.weight.mul_(0.01)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, 0.0, 0.02)

    def n_params(self, non_embedding=False):
        n = sum(p.numel() for p in self.parameters())
        return n - self.tok_emb.weight.numel() if non_embedding else n

    def pool_aux(self):
        from .pool import PooledMLP
        t = [m.aux for m in self.modules() if isinstance(m, PooledMLP)]
        return torch.stack(t).mean() if t else torch.zeros(
            (), device=self.tok_emb.weight.device)

    def pool_dropped(self, reset=True):
        """
        Share of token-expert assignments the capacity bound discarded.

        Returns (share, dropped, routed) over the interval since the last
        call, which is what the progress line wants - a cumulative count over
        a week of reading tells you nothing about now.

        A dropped assignment costs a token one of its top_k experts. It is not
        free and it is not fatal; what matters is whether it is rare. Rising
        here means the router is concentrating faster than capacity_factor
        allows, and the experts it is most confident about are the ones losing
        their tokens.
        """
        from .pool import PooledMLP
        d = r = 0
        for m in self.modules():
            if isinstance(m, PooledMLP):
                d += int(getattr(m, "dropped", 0))
                r += int(getattr(m, "routed", 0))
                if reset:
                    m.dropped = 0
                    m.routed = 0
        return (d / r if r else 0.0), d, r

    def n_slots(self):
        c = self.cfg
        return c.n_prelude + c.max_steps * (c.n_recur + c.n_coda)

    def want_experts(self, idx):
        """
        Ask the pool for what this text wants, and make it resident.

        Called before the forward that reads it, so the experts that compute
        the forward are the ones the backward updates. Swapping inside a step
        would hand expert A's gradient to whatever occupied its slot by the
        time backward ran - and with checkpointing, the forward is recomputed
        during backward, so it would not even be consistent with itself.
        """
        p = getattr(self, "pool", None)
        if p is None or not hasattr(p, "demand"):
            return 0
        from .pool import PooledMLP
        sites = [m for m in self.modules() if isinstance(m, PooledMLP)]
        if not sites:
            return 0
        with torch.no_grad():
            x = self.tok_emb(idx)
            want = p.demand(x, sites, sites[0].top_k)
            moved = p.swap_to(p.choose_by_demand(want))
        p.arm_observation()      # collect the states this chunk routes on
        return moved

    def begin_segment(self):
        """
        Choose the working set for the stretch of text about to be read.

        Only a paged pool has anything to do here. The choice is scored on the
        segment just finished, so it cannot see what it is about to predict,
        and it is made once for the whole model rather than per token - which
        is what lets the set be small enough to be worth paging.
        """
        p = getattr(self, "pool", None)
        if p is None or not hasattr(p, "swap_to"):
            return 0
        return p.swap_to(p.choose())

    def end_segment(self, h):
        """Remember what this segment looked like, for the next choice."""
        p = getattr(self, "pool", None)
        if p is not None and hasattr(p, "observe"):
            p.observe(h)

    def empty_caches(self):
        return [{"k": None, "v": None} for _ in range(self.n_slots())]

    def sample_depth(self):
        """
        How many passes to run for this batch.

        PonderNet computes every pass up to the ceiling and weights them by the
        halting distribution, so cost scales with the ceiling whether or not a
        token needed the depth. While reading, the depth is drawn per batch
        from a Poisson with mean `train_steps_mean` and clamped to
        [min_steps, max_steps]: the average cost stays near the mean while the
        model still sees deep passes often enough to learn to use them.

        Inference runs the full ceiling and lets halting decide per character.
        """
        c = self.cfg
        if not self.training or c.train_steps_mean <= 0:
            return c.max_steps
        n = int(torch.poisson(torch.tensor(float(c.train_steps_mean))).item()) + 1
        return max(c.min_steps, min(c.max_steps, n))

    def forward(self, idx, targets=None, caches=None, pos_offset=0,
                collect=False):
        cfg = self.cfg
        B, T = idx.shape
        x = self.tok_emb(idx)
        if pos_offset + T > self.rope_cos.shape[0]:
            # Reading past the rotary tables gives an empty slice and a
            # shape error four frames deeper, which says nothing useful
            raise ValueError(
                f"reading at position {pos_offset + T:,} but the rotary "
                f"tables were built to {self.rope_cos.shape[0]:,}. The window "
                f"may not exceed the ceiling the model was built with "
                f"(model.context_end in config.yaml).")
        cos = self.rope_cos[pos_offset:pos_offset + T]
        sin = self.rope_sin[pos_offset:pos_offset + T]

        def slot(i):
            return None if caches is None else caches[i]

        ci = 0
        for blk in self.prelude:
            x = blk(x, cos, sin, slot(ci))
            ci += 1

        h = torch.zeros_like(x)
        cum = torch.ones(B, T, 1, device=x.device, dtype=torch.float32)
        loss_terms, p_terms = [], []
        halted_logits = None
        halted = torch.zeros(B, T, 1, device=x.device, dtype=torch.bool)
        steps_used = torch.zeros(B, T, device=x.device)
        per_step = []

        n_steps = self.sample_depth()
        for n in range(n_steps):
            # truncated backprop: only the last few passes carry gradient, so
            # memory does not grow with depth
            if (targets is not None and cfg.bptt_window > 0
                    and n < n_steps - cfg.bptt_window):
                h = h.detach()
            h = self.adapter(torch.cat([h, x], dim=-1))
            for blk in self.recur:
                h = blk(h, cos, sin, slot(ci))
                ci += 1
            y = h
            for blk in self.coda:
                y = blk(y, cos, sin, slot(ci))
                ci += 1
            yf = self.ln_f(y)
            logits_n = self.head(yf)

            lam = torch.sigmoid(self.halt(yf).float())          # [B,T,1]
            if n == n_steps - 1:
                lam = torch.ones_like(lam)                      # must stop
            elif n < cfg.min_steps - 1:
                lam = torch.zeros_like(lam)                     # must continue
            p_n = cum * lam
            cum = cum * (1.0 - lam)

            if targets is not None:
                ce = F.cross_entropy(
                    logits_n.reshape(-1, logits_n.size(-1)).float(),
                    targets.reshape(-1), reduction="none").view(B, T)
                loss_terms.append(p_n.squeeze(-1) * ce)
                p_terms.append(p_n.squeeze(-1))
                # the halting-weighted mixture of every depth's logits, so
                # what is returned is what the model would actually emit
                term = logits_n * p_n.to(logits_n.dtype)
                halted_logits = (term if halted_logits is None
                                 else halted_logits + term)
            else:
                # Each token halts on its own schedule: the first step whose
                # cumulative halting mass crosses the threshold is the one
                # whose logits that token keeps. The forced lam=1 on the last
                # step guarantees every token halts somewhere.
                if halted_logits is None:
                    halted_logits = logits_n.clone()
                    steps_used = torch.ones(B, T, device=x.device)
                newly = (~halted) & ((1.0 - cum) >= cfg.halt_thresh)
                if bool(newly.any()):
                    halted_logits = torch.where(newly, logits_n, halted_logits)
                    steps_used = torch.where(
                        newly.squeeze(-1),
                        torch.full_like(steps_used, float(n + 1)), steps_used)
                halted = halted | newly
            if collect:
                per_step.append({"step": n + 1,
                                 "halt_p": float(p_n.mean()),
                                 "cum": float((1 - cum).mean())})

        # what this stretch of text looked like, for the next segment's choice
        self.end_segment(x)

        if targets is None:
            out = {"steps": steps_used} if collect else None
            if collect:
                out["per_step"] = per_step
            return halted_logits, out

        # PonderNet: expected loss under the halting distribution, plus a KL
        # pull toward a geometric prior so it does not simply always run long.
        P = torch.stack(p_terms, 0)                              # [N,B,T]
        L = torch.stack(loss_terms, 0)
        loss = L.sum(0).mean()
        prior = torch.tensor(
            [cfg.halt_prior * (1 - cfg.halt_prior) ** n
             for n in range(len(p_terms))], device=x.device)
        prior = (prior / prior.sum()).view(-1, 1, 1)
        kl = (P.clamp_min(1e-8) * (P.clamp_min(1e-8).log() - prior.log())).sum(0)
        loss = loss + cfg.ponder_beta * kl.mean()
        steps = (P * torch.arange(1, len(p_terms) + 1, device=x.device)
                 .view(-1, 1, 1)).sum(0)
        self.last_steps = float(steps.mean())
        # logits are the halting-weighted mixture, so top-1 accuracy measured
        # downstream reflects what the model would actually have emitted
        return halted_logits, loss

    @torch.no_grad()
    def peek_experts(self, idx, free=True, window=None):
        """
        Choose the working set from THIS text, by reading it once first.

        demand() scores the states the call sites routed on while reading the
        previous chunk, and that is the right evidence in the middle of a
        passage: text is locally coherent, and what the last few hundred
        characters needed is a fair guess at what the next few hundred will.

        At a boundary it is not evidence at all. The first chunk of a new
        passage, and a prompt, have no previous chunk of their own - so the
        buffer still holds the states of whatever was read last, which is a
        different subject entirely. demand() then answers a question nobody
        asked: it returns the same working set for a chess game and a Python
        file, because the text it is scoring is neither of them. Measured: the
        same buffer with two different prompts gave byte-identical working
        sets.

        So read the chunk once and score on its own states - but read it FROM
        A FIXED STARTING SET, because the read alone is not enough.

        The peek routes through the pool at every recurrence step, so the
        states it collects depend on two things: the text, and whichever
        experts happened to be resident when it began. The second is the
        previous subject, and it does not wash out - measured over seven
        passages, peeking from wherever the last one left off gave 4.8
        different working sets for one prompt, and re-reading up to three
        times still gave 3.3.

        Pinning the starting set to a constant removes that input. The states
        are then a function of the text alone, and the same passage chooses
        the same experts whatever preceded it - 1.0 distinct working sets
        across the same seven passages, and one answer per arithmetic prompt
        instead of two. A boundary is meant to be a blank slate; this is what
        makes it one.

        The constant is the top of the gate rather than an arbitrary set.
        Any fixed set gives the invariance, since all that matters is that it
        never changes. Taking the experts that have earned the most means the
        peek also reads with competent ones, so the states it hands to
        demand() are worth scoring: on the nine sample prompts this reached
        62 of the pool against 43 for the alternative that gets determinism
        by throwing evidence away.

        The set moves as the gate moves, so it is recomputed rather than
        cached. That is a topk over the pool and costs nothing. It means the
        choice is a function of the text and the current weights - which is
        the intent: a boundary should not depend on what was READ before it.

        Costs one forward over `window` characters and two swaps - one into
        the fixed set, one into the chosen one. Paid once per passage in
        training, one chunk in 32,768 characters, and once per reply while
        serving. Every chunk after the first keeps want_experts, which is
        free. The forward is under no_grad, and reading carries 2,048
        characters WITH gradients every chunk, so a chunk without them once
        per passage is not a cost worth trading evidence for.

        BOUNDED, and not only for the cost. `window` defaults to the chunk
        reading uses, so a passage boundary scores the whole chunk it is
        about to read - all of it, not a slice whose size came from
        somewhere else. A prompt is not bounded that way: it is however long
        the conversation has got, and forwarding that in one pass
        materialises every position across every block application at once,
        which is the thing serve.py's chunked prefill exists to avoid, and
        past cfg.block it is not a forward at all but a rotary-table error.

        The tail is what the window takes when it binds. For a prompt that
        is the right end - the last characters are the message being
        answered. For a chunk it never binds, because the window is the
        chunk.
        """
        if window is None:
            # the chunk reading uses. Hardcoding it here put the number in
            # two places with nothing tying them together, and the literal
            # that got written was train.py's fallback rather than the
            # setting in effect - so the peek scored a quarter of the chunk
            # it was choosing for.
            from .config import get, load
            window = int(get(load(), "training.chunk", 512))
        look = idx[:, -min(window, self.cfg.block):]
        p = getattr(self, "pool", None)
        if p is None or not hasattr(p, "demand") or not hasattr(p, "swap_to"):
            # No pool, or one with every expert resident: there is no working
            # set to choose and nothing to peek from. Ask anyway. A caller
            # asks the pool with the text before generating, and that holds
            # whichever pool is underneath - the ask is simply a no-op here.
            return self.want_experts(look)
        p.swap_to(self.canonical_experts())
        p.arm_observation()          # drop what the last text left behind
        self(look, caches=self.empty_caches(), pos_offset=0)
        return self.choose_for(look, free=free)

    @torch.no_grad()
    def canonical_experts(self):
        """
        The fixed set a boundary reads from: the most-earned experts.

        Fixed is the requirement - the peek's states must not depend on what
        was resident before it. Most-earned is the preference, so that the
        text is read by experts that contribute rather than by whichever ones
        an arbitrary rule named.
        """
        p = self.pool
        n = getattr(p, "_n", 0)
        k = min(len(p.slots), n)
        g = p.gate.detach().abs()[:n]
        return torch.topk(g, k).indices.tolist()

    @torch.no_grad()
    def choose_for(self, idx, free=False):
        """
        Put the experts this text wants on the card.

        A paged model loads with an EMPTY card - every slot -1, every expert
        weight zero - so anything that generates without calling this runs on
        the trunk alone and the pool contributes nothing at all. That is not a
        degraded model, it is a different and much smaller one.

        `free` releases the hysteresis that keeps the working set steady while
        reading a continuous stream. A prompt is the opposite: a deliberate
        change of subject, and the model should be free to re-choose at once.
        """
        p = getattr(self, "pool", None)
        if p is None or not hasattr(p, "demand"):
            return 0
        keep = (getattr(p, "dwell", None), getattr(p, "margin", None))
        if free and keep[0] is not None:
            p.dwell, p.margin = 0, 0.0
        try:
            return self.want_experts(idx)
        finally:
            if free and keep[0] is not None:
                p.dwell, p.margin = keep

    def generate(self, idx, max_new_tokens, temperature=0.0, top_k=0,
                 top_p=1.0, collect=False, rep_penalty=1.0,
                 no_repeat_ngram=0, reselect_every=None,
                 adapt_strength=2.5, adapt_decay=0.88):
        # None means "whatever config.yaml says". Hardcoding it here put the
        # same literal in three files with nothing tying them together, while
        # its sibling - how often READING re-chooses - sat in the settings.
        if reselect_every is None:
            from .config import get, load
            reselect_every = get(load(), "pool.reselect_chars", 64)
        self.eval()
        cfg = self.cfg
        caches = self.empty_caches()
        out = idx[:, -cfg.block:]
        cur, offset = out, 0
        steps_log = []
        # the prompt decides which experts answer it
        self.choose_for(out, free=True)
        for _i in range(max_new_tokens):
            # and the answer decides again as it develops: what the text wants
            # after a hundred characters is not what the prompt alone asked for
            if reselect_every and _i and _i % reselect_every == 0:
                self.choose_for(cur)
            if offset + cur.shape[1] > cfg.block:
                caches = self.empty_caches()
                cur = out[:, -cfg.block // 2:]
                offset = 0
            with amp(idx.device):
                logits, extra = self(cur, caches=caches, pos_offset=offset,
                                     collect=collect)
            offset += cur.shape[1]
            if collect and extra is not None:
                steps_log.append(float(extra["steps"][0, -1]))
            nxt = pick_next(logits[:, -1, :].float(), out, temperature, top_k,
                            top_p, rep_penalty, no_repeat_ngram,
                            adapt_strength=adapt_strength,
                            adapt_decay=adapt_decay)
            out = torch.cat([out, nxt], dim=1)
            cur = nxt
        return (out, steps_log) if collect else out


def load_recur(path, device, read_only=False):
    """
    Load a model from the weights directory, or from a .pt checkpoint.

    The directory is the model, so `weights` is the normal thing to pass. A
    .pt path still works because the film's captures and older invocations use
    one, and there is no reason to break them.
    """
    import os
    if os.path.isdir(path) and os.path.exists(os.path.join(path, "manifest.json")):
        return _load_dir(path, device, read_only=read_only)
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = RecurConfig(**ck["cfg"])
    m = RecurCoder(cfg).to(device)
    # cfg.pool_experts is the size the pool was BUILT at; growth and pruning
    # move it, and the checkpoint holds wherever it ended up. Resize before
    # loading, or a strict load raises and a lenient one silently drops every
    # expert past the built size.
    if cfg.use_pool:
        n_ck = max((int(k.split(".")[2]) for k in ck["model"]
                    if k.startswith("pool.experts.")), default=-1) + 1
        if n_ck and n_ck != m.pool.n_experts():
            have = m.pool.n_experts()
            if n_ck > have:
                m.pool.add_experts(n_ck - have, device=device)
            else:
                m.pool.experts = nn.ModuleList(list(m.pool.experts)[:n_ck])
                m.pool.gate = nn.Parameter(m.pool.gate.data[:n_ck].clone())
                for b in ("use", "age", "born", "gate_seen"):
                    if hasattr(m.pool, b):
                        setattr(m.pool, b, getattr(m.pool, b)[:n_ck].clone())
                m.pool.invalidate()
    m.load_state_dict(ck["model"])
    m.eval()
    return m, ck


# ----------------------------------------------------------------------------
# training
# ----------------------------------------------------------------------------



def load_any(ckpt_path, device, read_only=True):
    """
    Load whatever is at this path: a weights DIRECTORY or an older .pt file.

    The directory is the normal case now, and it is what every benchmark
    should be pointed at. A paged model arrives with an EMPTY card - every
    slot -1 and every expert weight zero - so anything that generates without
    first asking for experts runs on the trunk alone, about a fortieth of the
    model. RecurCoder.generate now asks; nothing here needs to.
    """
    import os
    if os.path.isdir(ckpt_path):
        return load_recur(ckpt_path, device, read_only=read_only)
    head = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if head.get("arch") == "recur":
        return load_recur(ckpt_path, device)
    return load_model(ckpt_path, device)

def load_model(ckpt_path, device):
    """
    Load whatever kind of checkpoint this is.

    Everything current is a RecurCoder, but benchmarks and the UI are given
    paths by hand and should not have to know that.
    """
    return load_recur(ckpt_path, device)


def _load_dir(path, device, paged=None, read_only=False):
    """
    Rebuild a model from a weights directory.

    A directory written by the paged trainer holds one file per expert, and
    materialising all of them costs the whole pool in RAM - which grows every
    time the model does. So a manifest marked `paged` is loaded through the
    paging path by default: only the working set becomes tensors, and the cost
    stops depending on how large the pool has become. Pass paged=False to
    force every expert into memory, which is what a tool that needs to touch
    all of them at once must do.
    """
    import json
    import os
    import sys
    from . import store as weights_store
    with open(os.path.join(path, "manifest.json")) as f:
        man = json.load(f)
    if paged is None:
        paged = bool(man.get("paged"))
    if paged:
        # build_paged lives in train.py; a caller in another directory (the
        # film's captures run from video/) needs the repo root on the path
        _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if _root not in sys.path:
            sys.path.insert(0, _root)
        import train as _train
        m, cfg, pool, man2 = _train.build_paged(path, device,
                                                read_only=read_only)
        return m, {"cfg": cfg.__dict__, "step": man2.get("step"),
                   "val": man2.get("val")}
    cfg = RecurConfig(**{k: v for k, v in (man.get("cfg") or {}).items()
                         if k in RecurConfig.__dataclass_fields__})
    want = int(man["n_experts"])
    # The routers are sized by pool_max, so a checkpoint that grew past the
    # ceiling recorded in its own cfg cannot be rebuilt from it: the router
    # rows come back one short and load_state_dict refuses the whole model.
    # train.py's read path already widens the ceiling to what the pool
    # actually holds; do the same here so a weights directory loads whatever
    # it contains.
    if cfg.use_pool and want > cfg.pool_max:
        cfg.pool_max = want
    m = RecurCoder(cfg).to(device)
    if cfg.use_pool and want != m.pool.n_experts():
        have = m.pool.n_experts()
        if want > have:
            m.pool.add_experts(want - have, device=device)
        else:
            m.pool.experts = nn.ModuleList(list(m.pool.experts)[:want])
            m.pool.gate = nn.Parameter(m.pool.gate.data[:want].clone())
            for b in ("use", "age", "born", "gate_seen"):
                setattr(m.pool, b, getattr(m.pool, b)[:want].clone())
            m.pool.invalidate()
    weights_store.load(m, path, device=device)
    m.eval()
    return m, {"cfg": man.get("cfg"), "step": man.get("step"),
               "val": man.get("val"), "arch": "recur"}
