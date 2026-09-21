"""
Write a fresh weights directory: one file per expert, nothing trained yet.

The directory IS the model, so starting from scratch means writing the
directory rather than initialising an object and hoping something saves it
later. Every expert file holds its weights beside a pair of zeroed Adam
moments, which is what the paged pool expects to find when it loads one.

This used to be a script in tools/, which meant a checkout without tools/
could not create a model at all, and the documented first step was a file
that might not be there. Training calls `create` itself now when it is
pointed at a directory that does not exist, so there is no first step.

Every default comes from config.yaml. Nothing here decides model shape.
"""

import os
import shutil

import torch


def create(out, seed=0, verbose=True, force=False, **over):
    """
    Build `out` from config.yaml, returning the config that was written.

    Keyword overrides take the same names the settings do - experts,
    resident, d_ff, depth, d_model, trunk_d_ff, n_head, block, max_steps,
    top_k - and a None is ignored, so a caller can forward unset CLI
    arguments without special-casing each one.
    """
    from dataclasses import asdict

    from minagi import store
    from minagi.config import load, get
    from minagi.recur import RecurConfig, RecurCoder

    c = load()
    over = {k: v for k, v in over.items() if v is not None}

    def pick(name, *path, default=None):
        if name in over:
            return over[name]
        for p in path:
            v = get(c, p, None)
            if v is not None:
                return v
        return default

    experts    = int(pick("experts", "pool.experts", default=32))
    resident   = int(pick("resident", "pool.resident", default=32))
    d_ff       = int(pick("d_ff", "pool.width", "pool.d_ff", default=1024))
    depth      = int(pick("depth", "pool.depth", default=1))
    d_model    = int(pick("d_model", "model.d_model", default=512))
    trunk_d_ff = int(pick("trunk_d_ff", "model.d_ff", default=1408))
    n_head     = int(pick("n_head", "model.n_head", default=8))
    block      = int(pick("block", "model.context_end", "model.context",
                          default=16384))
    max_steps  = int(pick("max_steps", "model.max_steps", default=6))
    top_k      = int(pick("top_k", "pool.top_k", default=8))

    if os.path.exists(out):
        if not force:
            raise FileExistsError(f"{out} exists; pass force=True to replace it")
        shutil.rmtree(out)

    cfg = RecurConfig(vocab_size=265, d_model=d_model, n_head=n_head,
                      d_ff=trunk_d_ff,
                      n_prelude=get(c, "model.n_prelude", 2),
                      n_recur=get(c, "model.n_recur", 3),
                      n_coda=get(c, "model.n_coda", 1),
                      max_steps=max_steps, block=block, use_pool=True,
                      pool_experts=experts, pool_d_ff=d_ff,
                      pool_depth=depth, pool_top_k=top_k,
                      # Router rows belong to EXPERTS, not to VRAM slots, so
                      # this is one row per expert in the pool - not the
                      # working set. Sizing it to `resident` writes a router
                      # too small to address most of the pool, and the model
                      # then refuses to load. Growth adds rows to both this
                      # and the segment router.
                      pool_max=experts)
    torch.manual_seed(seed)
    m = RecurCoder(cfg)
    d = asdict(cfg)
    d["pool_resident"] = resident
    store.save(m, out, step=-1, val=None, opt=None, cfg=d, verbose=verbose)

    if verbose:
        per = depth * 3 * d_model * d_ff
        print(f"\n  {experts} experts x {per:,} parameters "
              f"({d_ff} hidden units, depth {depth})")
        print(f"  pool {experts * per / 1e6:.2f}M, resident {resident} = "
              f"{resident * per / 1e6:.2f}M in VRAM")
        print(f"  total {m.n_params() / 1e6:.2f}M")
    return d
