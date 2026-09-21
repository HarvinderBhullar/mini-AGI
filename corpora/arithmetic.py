#!/usr/bin/env python3
"""
Synthetic arithmetic corpus for mini-AGI.

The model is byte-level, so a digit is already its own token and numbers are
written here exactly as they are written everywhere else in the corpus. Two
format decisions remain, and both are forced by measured facts rather than
taste:

1. ANSWERS ARE WRITTEN IN NORMAL ORDER, most significant digit first. The
   small-model literature reverses them, because carries propagate right to
   left and a reversed answer makes each output digit a local function of what
   came before. That is a real advantage and it is given up on purpose: this
   model reads wikipedia, chat, reasoning and code, where every number is
   written forwards, and a corpus that writes them backwards teaches two
   conflicting conventions for the same thing. Consistency across the eight
   subjects is worth more here than the decoding trick.

   What replaces it is the scratchpad. A trace that works right to left gives
   the model the same local structure reversal gave it, without changing how
   numbers are spelled.

2. THE SCRATCHPAD SHOWS THE WORK, not the result. Every step must be derivable
   from the ones before it. A note that states the answer and calls itself
   working teaches the model to emit a plausible number and copy it - which is
   exactly what a scratchpad reading `total=146738` produced: correct partial
   products, an unexplained total, and a final answer copied from it.

Unlike code, arithmetic is genuinely learnable by a model this size - the
constraint is format and data volume, not capacity. Expect high exact-match on
addition and subtraction, and materially worse on multiplication.

    python3 corpora/arithmetic.py --out data_math_char --n 4000000
"""

import os
import sys
import json
import random
import argparse

import numpy as np

# the repo root, not corpora/ - this is where minagi.tokenizer lives
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def rnd(rng, digits):
    lo = 10 ** (digits - 1) if digits > 1 else 0
    return rng.randint(lo, 10 ** digits - 1)


def _think(body):
    """
    An inspectable scratchpad, delimited exactly like a thinking trace.

    Working is written in normal reading order because it is meant to be read;
    only the final answer is reversed, because that is the part where carry
    order matters. Notes appear on only a fraction of examples so the model
    learns both a fast path and a deliberate one, and can pick per problem.
    """
    return f"<think> {body} </think> "


def gen_add(rng, d, notes=False):
    a, b = rnd(rng, d), rnd(rng, rng.randint(1, d))
    t = ""
    if notes:
        da, db = str(a)[::-1], str(b)[::-1]
        steps, carry = [], 0
        for i in range(max(len(da), len(db))):
            x = int(da[i]) if i < len(da) else 0
            y = int(db[i]) if i < len(db) else 0
            s = x + y + carry
            steps.append(f"{x}+{y}+{carry}={s%10}c{s//10}")
            carry = s // 10
        if carry:
            steps.append(f"c{carry}")
        t = _think(" ".join(steps))
    return f"add {a} + {b} = {t}{a + b}"


def gen_sub(rng, d, notes=False):
    a, b = rnd(rng, d), rnd(rng, rng.randint(1, d))
    t = ""
    if notes:
        # The borrow chain runs right to left, the mirror of gen_add's carry.
        # It is taken over the larger magnitude, because that is the only way
        # the chain is defined; the sign is then a separate fact to state.
        hi, lo = (a, b) if a >= b else (b, a)
        dh, dl = str(hi)[::-1], str(lo)[::-1]
        steps, borrow = [], 0
        for i in range(len(dh)):
            x = int(dh[i])
            y = int(dl[i]) if i < len(dl) else 0
            v = x - y - borrow
            nxt = 1 if v < 0 else 0
            steps.append(f"{x}-{y}-{borrow}={v % 10}b{nxt}")
            borrow = nxt
        if a < b:
            steps.append("sign -")
        t = _think(" ".join(steps))
    return f"sub {a} - {b} = {t}{a - b}"


def gen_mul(rng, d, notes=False):
    a = rnd(rng, d)
    b = rnd(rng, rng.randint(1, min(2, d)))
    t = ""
    if notes:
        # Every partial product, and then the additions that combine them.
        # Stating a bare `total=` here is what taught the model to emit a
        # plausible number and copy it into the answer without ever learning
        # to add the partials.
        vals, parts = [], []
        for i, dg in enumerate(str(b)[::-1]):
            v = a * int(dg) * (10 ** i)
            vals.append(v)
            parts.append(f"{a}*{dg}{'0' * i}={v}")
        acc = vals[0]
        for v in vals[1:]:
            parts.append(f"{acc}+{v}={acc + v}")
            acc += v
        t = _think(" ".join(parts))
    return f"mul {a} * {b} = {t}{a * b}"


def gen_cmp(rng, d, notes=False):
    a, b = rnd(rng, d), rnd(rng, d)
    op = rng.choice(["<", ">", "=="])
    truth = {"<": a < b, ">": a > b, "==": a == b}[op]
    t = ""
    if notes:
        sa, sb = str(a), str(b)
        if len(sa) != len(sb):
            body = f"len {len(sa)} vs {len(sb)}"
        else:
            # Equal lengths are the whole difficulty, and a length comparison
            # alone said nothing about them.
            k = next((i for i, (x, y) in enumerate(zip(sa, sb)) if x != y), None)
            body = (f"len {len(sa)} vs {len(sb)} equal"
                    if k is None else
                    f"len {len(sa)} vs {len(sb)} digit {k} {sa[k]} vs {sb[k]}")
        t = _think(body)
    return f"cmp {a} {op} {b} = {t}{'yes' if truth else 'no'}"


def gen_mod(rng, d, notes=False):
    a, b = rnd(rng, d), rng.randint(2, 99)
    t = ""
    if notes:
        # Both halves are checkable: b*q fits under a, b*(q+1) does not, and
        # the remainder is the subtraction. Stating `rem` alone derived nothing.
        q = a // b
        t = _think(f"{b}*{q}={b * q} {b}*{q + 1}={b * (q + 1)}>{a} "
                   f"{a}-{b * q}={a % b}")
    return f"mod {a} % {b} = {t}{a % b}"


def gen_gcd(rng, d, notes=False):
    import math as _m
    a, b = rnd(rng, min(d, 4)), rnd(rng, min(d, 4))
    t = ""
    if notes:
        steps, x, y = [], a, b
        while y and len(steps) < 8:
            steps.append(f"{x}%{y}={x%y}")
            x, y = y, x % y
        t = _think(" ".join(steps))
    return f"gcd {a} , {b} = {t}{_m.gcd(a, b)}"


def gen_sum(rng, d, notes=False):
    k = rng.randint(2, 5)
    xs = [rnd(rng, rng.randint(1, min(d, 4))) for _ in range(k)]
    body = " + ".join(str(x) for x in xs)
    t = ""
    if notes:
        run, steps = 0, []
        for x in xs:
            run += x
            steps.append(str(run))
        t = _think(" ".join(steps))
    return f"sum {body} = {t}{sum(xs)}"


def gen_round(rng, d, notes=False):
    a = rnd(rng, max(d, 2))
    p = rng.choice([10, 100, 1000])
    # Half-UP, not Python's round(), which is half-to-even: the trace below
    # says "r >= p/2 goes up", and a rule the working contradicts on exactly
    # the hard cases is worse than no working at all.
    q, r = divmod(a, p)
    res = (q + 1) * p if r * 2 >= p else q * p
    t = ""
    if notes:
        t = _think(f"{a}={q}*{p}+{r} {r}{'>=' if r * 2 >= p else '<'}{p // 2} "
                   f"-> {'up' if r * 2 >= p else 'down'}")
    return f"round {a} to {p} = {t}{res}"


TASKS = {
    "add": (gen_add, 8, 0.30),
    "sub": (gen_sub, 8, 0.20),
    "mul": (gen_mul, 4, 0.18),
    "cmp": (gen_cmp, 8, 0.09),
    "mod": (gen_mod, 6, 0.08),
    "gcd": (gen_gcd, 4, 0.06),
    "sum": (gen_sum, 4, 0.06),
    "round": (gen_round, 6, 0.03),
}

NOTES_FRAC = 0.3


def sample_line(rng):
    r = rng.random()
    acc = 0.0
    notes = rng.random() < NOTES_FRAC
    for name, (fn, maxd, w) in TASKS.items():
        acc += w
        if r <= acc:
            # uniform over digit counts so long problems are not rare
            return fn(rng, rng.randint(1, maxd), notes)
    return gen_add(rng, rng.randint(1, 8), notes)


def main():
    ap = argparse.ArgumentParser()
    # what `corpora expand` reads; there is only one output kind now
    ap.add_argument("--out", default="data_math_char")
    ap.add_argument("--n", type=int, default=2_000_000)
    ap.add_argument("--val", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--notes-frac", type=float, default=0.3,
                    help="share of examples carrying a <think> scratchpad")
    args = ap.parse_args()

    global NOTES_FRAC
    NOTES_FRAC = args.notes_frac
    from minagi.tokenizer import ByteTokenizer
    tok = ByteTokenizer()
    os.makedirs(args.out, exist_ok=True)
    rng = random.Random(args.seed)

    print(f"generating {args.n:,} problems (+{args.val:,} val)")
    print("sample:")
    for _ in range(6):
        print("  " + sample_line(rng))

    for split, count in (("train", args.n), ("val", args.val)):
        path = os.path.join(args.out, f"{split}.bin")
        total = 0
        with open(path, "wb") as f:
            batch = []
            for i in range(count):
                batch.append(sample_line(rng))
                if len(batch) >= 4096:
                    for enc in tok.encode_batch([b + "\n" for b in batch]):
                        arr = np.array(enc.ids, dtype=np.uint16)
                        f.write(arr.tobytes())
                        total += len(arr)
                    batch.clear()
                    if split == "train" and i % 200000 < 4096:
                        print(f"\r  {split}: {total/1e6:.1f}M tokens",
                              end="", file=sys.stderr)
            if batch:
                for enc in tok.encode_batch([b + "\n" for b in batch]):
                    arr = np.array(enc.ids, dtype=np.uint16)
                    f.write(arr.tobytes())
                    total += len(arr)
        print(f"\r  {split}: {total:,} tokens -> {path}")
        if split == "train":
            train_total = total
        else:
            val_total = total

    meta = {"vocab_size": tok.get_vocab_size(), "train_tokens": train_total,
            "val_tokens": val_total, "tasks": list(TASKS),
            "tokenizer": "byte",
            "note": "answers in normal order; scratchpads show derivable steps"}
    json.dump(meta, open(os.path.join(args.out, "meta.json"), "w"), indent=2)
    print(f"wrote {args.out}/meta.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
