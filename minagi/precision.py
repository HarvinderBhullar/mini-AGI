"""
What precision the model computes in.

One setting for the whole process, because reading and generating are the same
code path and must not differ in what they cost or what they compute.

Only the arithmetic and the activations move. WEIGHTS stay fp32, on disk as
well as in memory, and that is not caution: an expert here pages out to RAM
and back several times per chunk, and every page-out in a shorter format
would be a fresh rounding. Measured on the real pool, against a typical Adam
update of 1.8e-05 (weight |w| median 1.3e-02):

    weights bfloat16   err 1.4e-05    0.80x one update
    weights float16    err 1.8e-06    0.10x one update

bfloat16 would round away most of what an expert had just learned, over and
over. float16 looks affordable and is not: its resolution at a typical weight
is 1.3e-05 against an update of 1.8e-05, so updates land barely above the
grid, and an expert is rewritten a median of 406 times.

Adam's MOMENTS are a different question with a different answer, and holding
them to the weights' standard would cost gigabytes for nothing. They are tiny
and span an enormous range - m median 3.5e-09, v median 2.0e-15 - so what they
need is exponent, which bfloat16 keeps in full, and not mantissa. Measured the
same way:

    m bfloat16   0.10% of its own magnitude    0.000x one update
    v bfloat16   0.10% of its own magnitude    0.000x one update
    m float16    86% - underflows              (v: 100%, to zero)

float16 destroys them outright: both sit far below its smallest subnormal.
bfloat16 costs a thousandth of their magnitude and two thirds of the file, so
the moments are stored bf16 and the weights fp32. Nothing re-derives a weight,
but Adam re-derives its moments continuously, which is the other reason the
error does not accumulate there.

What autocast does buy is the part that is actually large: the KV cache is 26
block-applications of keys and values across the whole context window, and it
halves. bfloat16 needs no loss scaling - it keeps fp32's exponent range and
spends its bits on mantissa instead. The places that cannot afford a short
mantissa force themselves back to fp32 on the spot: the halting accumulator,
the cross-entropy, the router logits and RMSNorm.
"""

import torch

_COMPUTE = {"dtype": torch.float32}

NAMES = {"fp32": torch.float32, "float32": torch.float32,
         "bf16": torch.bfloat16, "bfloat16": torch.bfloat16,
         "fp16": torch.float16, "float16": torch.float16}


def set_compute_dtype(dtype):
    """Takes a torch dtype or one of the names above."""
    if isinstance(dtype, str):
        if dtype not in NAMES:
            raise ValueError(f"unknown precision {dtype!r}; "
                             f"expected one of {sorted(NAMES)}")
        dtype = NAMES[dtype]
    _COMPUTE["dtype"] = dtype


def compute_dtype():
    return _COMPUTE["dtype"]


def amp(device=None):
    """The autocast region every forward runs inside."""
    dt = _COMPUTE["dtype"]
    dev = "cuda" if device is None else (
        device.type if hasattr(device, "type") else str(device).split(":")[0])
    return torch.autocast(dev, dtype=dt,
                          enabled=(dt is not torch.float32 and dev == "cuda"))


# -- storing moments in half the space ------------------------------------
#
# numpy has no bfloat16, so a bf16 array travels as int16 with the same bits.
# The dtype IS the marker: an array that comes back float32 was written by an
# older version and is used as-is, so old weight directories keep loading and
# convert themselves the first time each expert is written out.

def pack_bf16(t):
    """A float tensor as int16 carrying bfloat16 bits, for storage."""
    if not torch.is_tensor(t):
        t = torch.as_tensor(t)
    return t.to(torch.bfloat16).view(torch.int16).numpy()


def unpack_bf16(a):
    """Undo pack_bf16. Passes float arrays through, so old files still load."""
    t = torch.as_tensor(a)
    if t.dtype == torch.int16:
        return t.view(torch.bfloat16).to(torch.float32)
    return t.to(torch.float32)


def is_moment(name):
    """Whether a saved array is an Adam moment rather than a weight.

    Expert files name them w1_m / w1_v; the trunk's optimiser file uses
    `<param>|m` and `<param>|v`. `|t` is a step count and stays as it is.
    """
    return name.endswith(("_m", "_v", "|m", "|v"))
