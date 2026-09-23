"""
Allocator settings, which have to be in the environment before torch is.

A MODULE OF ITS OWN, and that is the whole point of it. These are read by each
backend's caching allocator when it first initialises, and setting them
afterwards is a silent no-op rather than an error - the run simply behaves as
though they had never been set. `minagi.device` is where every other hardware
question is answered, but importing it pulls in torch, so the one setting that
must precede torch cannot live there. This file imports nothing but `os`.

Both entry points call `configure()` as their first statement. Everything is
`setdefault`, so a value already exported in a shell still wins.
"""

import os


def configure():
    """
    CUDA: EXPANDABLE SEGMENTS. Without them the caching allocator keeps free
    blocks in fixed segments, and a run that alternates between a few large
    short-lived tensors - which is exactly what a checkpointed expert dispatch
    does, three batched matmuls recomputed in the backward - strands memory it
    cannot hand back. Measured on an 8,192 window: the backward asked for 384
    MiB with 394 MiB free and 861 MiB reserved but unallocated. There was
    plenty of memory; there was no contiguous piece of it.

    MPS: THE CPU FALLBACK. Metal does not implement every operator torch has,
    and without this an unimplemented one stops the run instead of running
    slowly. This project hits none of them on its current path, so the
    fallback should never fire - it is here so a future edit degrades rather
    than dies three days into a read.

    MPS: THE HIGH-WATERMARK RATIO IS LEFT ALONE, deliberately. Raising it lets
    Metal push the working set into swap, and on a reader that runs for weeks
    a run that silently starts paging to an SSD is far worse than an honest
    allocation failure - the pool's growth brake reads memory pressure to
    decide whether to add experts, and swapping tells it there is room when
    there is not.
    """
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
