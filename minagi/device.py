"""
Which accelerator this process runs on, and what it is able to do.

One module answers every hardware question, because the alternative - each
call site testing `device.type == "cuda"` for itself - is how a codebase
acquires a second, silent definition of "is there a GPU here". There were
eleven such tests before this file existed, and on Apple silicon every one of
them answered "no", so the model ran in fp32 on the CPU path while sitting on
a Metal device that could have done the work.

THE THREE BACKENDS, and what actually differs between them:

    cuda    everything. Fused AdamW, pinned host memory with asynchronous
            copies, a peak-allocation counter, a device memory total.
    mps     Apple silicon through Metal. Computes and autocasts, but has no
            fused optimiser, no pinned memory (pinning is a CUDA concept - the
            memory is already shared with the GPU), and no peak counter; see
            `max_memory_allocated` for what is used instead.
    cpu     no autocast worth having, no memory brake.

UNIFIED MEMORY CHANGES WHAT THE PAGING MEANS, and it is worth being explicit
because the rest of this project is built around the opposite assumption. On a
discrete card, an expert in VRAM and the same expert in RAM are two copies
across a PCIe bus, and `paged.py` exists to keep the VRAM side small. On Apple
silicon there is one pool of memory and the copy is a memcpy, not a transfer.
The paging still works and still bounds what the Metal allocator holds, but
the tier it is defending is cheaper than the one it was designed against, so
`pool.resident` can go up a lot on a machine with room. Nothing here changes
that setting; this note is so the number is understood rather than inherited.

WHAT IS NOT PORTED, deliberately:

    torch.cuda.graphs  never used by this project.
    fused AdamW        `foreach` is the fast path everywhere else, and
                       `supports_fused_adam` is what the call sites ask.
    NCCL / distributed never used; single process by design.
"""

import os

import torch


# ---------------------------------------------------------------------------
# what is here
# ---------------------------------------------------------------------------

def mps_available():
    """Metal is present, built into this torch, and usable."""
    b = getattr(torch.backends, "mps", None)
    try:
        return bool(b is not None and b.is_built() and b.is_available())
    except Exception:                                     # noqa: BLE001
        return False


def pick(name=None):
    """
    A torch.device, either the one asked for or the best one present.

    Order is cuda, mps, cpu - fastest first. An explicit name always wins,
    including `--device cpu` to stay off an accelerator on purpose.
    """
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if mps_available():
        return torch.device("mps")
    return torch.device("cpu")


def kind(dev):
    """'cuda' | 'mps' | 'cpu' from a device, a string, or a module."""
    if dev is None:
        return "cpu"
    if hasattr(dev, "type"):
        return dev.type
    if hasattr(dev, "parameters"):                        # an nn.Module
        try:
            return next(dev.parameters()).device.type
        except StopIteration:
            return "cpu"
    return str(dev).split(":")[0]


def is_accel(dev):
    return kind(dev) in ("cuda", "mps")


def label(dev):
    """A one-line name for a log: 'cuda (NVIDIA ...)' / 'mps (Apple ...)'."""
    k = kind(dev)
    if k == "cuda":
        try:
            return f"cuda ({torch.cuda.get_device_name(0)})"
        except Exception:                                 # noqa: BLE001
            return "cuda"
    if k == "mps":
        return f"mps ({_chip()})"
    return "cpu"


def _chip():
    try:
        import subprocess
        return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                              capture_output=True, text=True,
                              timeout=2).stdout.strip() or "Apple silicon"
    except Exception:                                     # noqa: BLE001
        return "Apple silicon"


# ---------------------------------------------------------------------------
# capabilities
# ---------------------------------------------------------------------------

def supports_fused_adam(dev):
    """
    Whether `fused=True` is accepted by torch.optim on this device.

    ASKED OF THE INSTALLED TORCH, not assumed from the backend, because the
    answer changed under this project rather than being a property of the
    hardware. MPS gained a fused AdamW during the 2.x series; before that the
    flag raised, and passing it hopefully would stop a run at its first step.
    Measured here on torch 2.14 / M5 Pro, fused and foreach produce updates
    that agree to the last bit, so this is a speed switch and never a
    numerical one - which is what makes probing for it safe.
    """
    k = kind(dev)
    if k == "cpu":
        return False
    if k == "cuda":
        return True
    if k not in _PROBED:
        _PROBED[k] = _probe_fused(k)
    return _PROBED[k]


def _probe_fused(k):
    """Build the smallest fused optimiser that would prove it, and step it."""
    try:
        p = torch.nn.Parameter(torch.zeros(2, device=k))
        p.grad = torch.zeros(2, device=k)
        torch.optim.AdamW([p], lr=1e-4, fused=True).step()
        return True
    except Exception:                                     # noqa: BLE001
        return False


def supports_pinning(dev):
    """
    Whether host tensors can be page-locked for asynchronous copies.

    Probed for the same reason as fused Adam. Pinning is a CUDA idea - it lets
    a DMA engine read host memory without the driver staging a copy - and the
    expectation was that unified memory would have no use for it. The
    installed torch pins on MPS anyway, so the transfer path is left to say so
    for itself rather than being told.

    `non_blocking` must travel with this answer and not without it: passing
    non_blocking with unpinned source memory is a silently synchronous copy,
    which is slower than the honest one and looks like it worked.
    """
    k = kind(dev)
    if k == "cpu":
        return False
    if k == "cuda":
        return True
    key = ("pin", k)
    if key not in _PROBED:
        _PROBED[key] = _probe_pinning()
    return _PROBED[key]


def _probe_pinning():
    try:
        return bool(torch.zeros(2).pin_memory().is_pinned())
    except Exception:                                     # noqa: BLE001
        return False


def supports_autocast(dev, dtype=None):
    """
    Whether an autocast region on this device will actually do anything.

    Probed once and remembered, because the answer depends on the torch build
    and the macOS version rather than on anything this project controls: bf16
    autocast on MPS needs macOS 14 or newer, and older torch does not register
    an MPS autocast backend at all. A build that cannot do it must fall back
    to fp32 rather than raise halfway through a run.
    """
    k = kind(dev)
    if k == "cpu":
        return False
    if k == "cuda":
        return True
    if k != "mps":
        return False
    dt = dtype or torch.bfloat16
    key = ("mps", str(dt))
    if key not in _PROBED:
        _PROBED[key] = _probe_mps_autocast(dt)
    return _PROBED[key]


_PROBED = {}


def _probe_mps_autocast(dt):
    """Run the smallest matmul that would prove it, and see what comes back."""
    try:
        with torch.autocast("mps", dtype=dt):
            a = torch.zeros(2, 2, device="mps")
            out = a @ a
        return out.dtype == dt
    except Exception:                                     # noqa: BLE001
        return False


# ---------------------------------------------------------------------------
# memory
# ---------------------------------------------------------------------------
#
# The growth brake in pool.AutoGrow reads `mem_frac` and refuses to add
# experts when the card is nearly spent, so these have to mean the same thing
# on both backends or the brake is either permanently on or permanently off.

_MPS_PEAK = {"bytes": 0}


def max_memory_allocated(dev):
    """
    Peak bytes this process has held on the device. 0 where unknowable.

    CUDA keeps this itself. MPS has no peak counter - only
    `current_allocated_memory()` - so the high-water mark is accumulated here
    from every call. That makes it a peak over the SAMPLES taken rather than a
    true peak, which is honest enough for the two things that read it: a
    growth brake asked once per growth interval, and a number printed in a log
    line. It is not a profiler, and a spike between two calls is invisible to
    it.
    """
    k = kind(dev)
    if k == "cuda":
        return int(torch.cuda.max_memory_allocated())
    if k == "mps":
        try:
            now = int(torch.mps.current_allocated_memory())
        except Exception:                                 # noqa: BLE001
            return 0
        if now > _MPS_PEAK["bytes"]:
            _MPS_PEAK["bytes"] = now
        return _MPS_PEAK["bytes"]
    return 0


def reset_peak(dev):
    k = kind(dev)
    if k == "cuda":
        torch.cuda.reset_peak_memory_stats()
    elif k == "mps":
        _MPS_PEAK["bytes"] = 0


def total_memory(dev):
    """
    Bytes the device may use, as the brake's denominator. 0 where unknowable.

    On CUDA that is the card. On Apple silicon it is NOT the machine's RAM:
    Metal will not let one process hold all of unified memory, and
    `recommended_max_memory()` is the working-set size it will actually give -
    typically around three quarters of the total. Measuring against physical
    RAM instead would put the brake at a fraction it can never reach, so the
    pool would grow until the allocator started failing.
    """
    k = kind(dev)
    if k == "cuda":
        try:
            return int(torch.cuda.get_device_properties(0).total_memory)
        except Exception:                                 # noqa: BLE001
            return 0
    if k == "mps":
        try:
            rec = int(torch.mps.recommended_max_memory())
            if rec > 0:
                return rec
        except Exception:                                 # noqa: BLE001
            pass
        return _physical_memory()
    return 0


def _physical_memory():
    """hw.memsize, for a torch too old to have recommended_max_memory."""
    try:
        import subprocess
        out = subprocess.run(["sysctl", "-n", "hw.memsize"],
                             capture_output=True, text=True, timeout=2).stdout
        # the same three-quarters Metal would have recommended
        return int(int(out.strip()) * 0.75)
    except Exception:                                     # noqa: BLE001
        return 0


def mem_frac(dev):
    """
    Peak fraction of the device this process needs. 0.0 without an accelerator.

    Deliberately NOT mem_get_info(): that reports the caching allocator's
    reserved pool, which stays near 100% once a run is warm whether or not
    there is real room, so a brake reading it would refuse growth forever.
    Peak *allocated* is the honest number - it is what has to fit.
    """
    total = total_memory(dev)
    if total <= 0:
        return 0.0
    return max_memory_allocated(dev) / total


def empty_cache(dev=None):
    """Hand cached blocks back to the driver. A no-op on CPU."""
    k = kind(dev) if dev is not None else None
    if k in (None, "cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
    if k in (None, "mps") and mps_available():
        try:
            torch.mps.empty_cache()
        except Exception:                                 # noqa: BLE001
            pass


def synchronize(dev):
    """Wait for queued device work. Needed before timing anything."""
    k = kind(dev)
    if k == "cuda":
        torch.cuda.synchronize()
    elif k == "mps":
        try:
            torch.mps.synchronize()
        except Exception:                                 # noqa: BLE001
            pass


# ---------------------------------------------------------------------------
# running out of it
# ---------------------------------------------------------------------------

def oom_errors():
    """
    Exception classes to catch around anything that might not fit.

    CUDA raises a dedicated `OutOfMemoryError`. MPS raises a plain
    `RuntimeError` whose message is the only thing distinguishing it from any
    other failure, so RuntimeError has to be in the tuple and `is_oom` has to
    be asked inside the handler before it is swallowed. Catching RuntimeError
    and assuming it was memory would turn every bug into "the GPU is full".
    """
    errs = []
    for holder in (torch, getattr(torch, "cuda", None)):
        e = getattr(holder, "OutOfMemoryError", None)
        if isinstance(e, type) and issubclass(e, BaseException):
            errs.append(e)
    errs.append(RuntimeError)
    return tuple(dict.fromkeys(errs))


def is_oom(exc):
    """Whether a caught exception really was the device running out."""
    for holder in (torch, getattr(torch, "cuda", None)):
        e = getattr(holder, "OutOfMemoryError", None)
        if isinstance(e, type) and isinstance(exc, e):
            return True
    msg = str(exc).lower()
    return ("out of memory" in msg or "insufficient memory" in msg
            or "can't allocate" in msg)


# ---------------------------------------------------------------------------
# allocator settings, which must be set before the backend initialises
# ---------------------------------------------------------------------------

def configure_allocator():
    """
    Deprecated spelling of `minagi.alloc.configure`, kept so nothing breaks.

    The real one lives in a module that does not import torch, because these
    settings must be in the environment BEFORE torch initialises a backend and
    importing this file is already too late. Calling it here still works only
    because `setdefault` on an already-set variable is harmless.
    """
    from .alloc import configure
    configure()


def enable_tf32():
    """
    TF32 on the CUDA matmul path. A no-op on any other backend.

    Kept behind a function because reading `torch.backends.cuda` at import
    time is fine, but it reads as a CUDA assumption in a file that is supposed
    not to have one.
    """
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except Exception:                                     # noqa: BLE001
        pass
