#!/usr/bin/env python3
"""
Synthetic arithmetic corpus for mini-AGI.

Two format decisions do most of the work here, and both are forced by measured
facts rather than taste:

1. DIGITS ARE SPACE-SEPARATED. The Python-trained BPE merges numerals into
   inconsistent multi-digit tokens ('1234' -> ['12','34'], '9876543' ->
   ['9','87','6','54','3']), which destroys place-value alignment. Spacing the
   digits yields exactly one token per digit.

2. SUMS ARE WRITTEN LEAST-SIGNIFICANT-DIGIT FIRST. Carries propagate right to
   left, so a left-to-right decoder writing the most significant digit first
   must know the whole carry chain before its first token. Reversing the answer
   makes each output digit a local function of what came before. This is the
   main trick from the small-model arithmetic literature and it is worth far
   more than parameters at this scale.

Unlike code, arithmetic is genuinely learnable by a model this size - the
constraint is format and data volume, not capacity. Expect high exact-match on
addition and subtraction, and materially worse on multiplication.

    python3 math_data.py --out data_math --n 2000000
"""

import os
import sys
import json
import random
import argparse

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# Character-level models see every digit as its own token already, so the
# space-separation hack the BPE needed is pure overhead there. SEP is set once
# from the CLI and both formatters read it.
SEP = " "


def sp(n):
    """123 -> '1 2 3' under BPE, '123' at character level."""
    s = str(abs(int(n)))
    body = SEP.join(s)
    return f"-{SEP}{body}" if n < 0 else body


def rev(n):
    """
    Answer digits, least significant first.

    Carries propagate right to left, so a left-to-right decoder writing the
    most significant digit first would need the whole carry chain before its
    first token. Reversing makes each digit a local function of what precedes
    it. This matters more than parameter count at this scale.
    """
    s = str(abs(int(n)))
    body = SEP.join(reversed(s))
    return f"-{SEP}{body}" if n < 0 else body


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
    return f"add {sp(a)} + {sp(b)} = {t}{rev(a + b)}"


def gen_sub(rng, d, notes=False):
    a, b = rnd(rng, d), rnd(rng, rng.randint(1, d))
    t = ""
    if notes:
        t = _think(f"{a} - {b} borrow-chain {abs(a-b)} sign {'-' if a<b else '+'}")
    return f"sub {sp(a)} - {sp(b)} = {t}{rev(a - b)}"


def gen_mul(rng, d, notes=False):
    a = rnd(rng, d)
    b = rnd(rng, rng.randint(1, min(2, d)))
    t = ""
    if notes:
        parts, total = [], 0
        for i, dg in enumerate(str(b)[::-1]):
            p = a * int(dg) * (10 ** i)
            parts.append(f"{a}*{dg}{'0'*i}={p}")
            total += p
        t = _think(" ".join(parts) + f" total={total}")
    return f"mul {sp(a)} * {sp(b)} = {t}{rev(a * b)}"


def gen_cmp(rng, d, notes=False):
    a, b = rnd(rng, d), rnd(rng, d)
    op = rng.choice(["<", ">", "=="])
    truth = {"<": a < b, ">": a > b, "==": a == b}[op]
    t = _think(f"len {len(str(a))} vs {len(str(b))}") if notes else ""
    return f"cmp {sp(a)} {op} {sp(b)} = {t}{'yes' if truth else 'no'}"


def gen_mod(rng, d, notes=False):
    a, b = rnd(rng, d), rng.randint(2, 99)
    t = _think(f"{a} // {b} = {a//b} rem {a%b}") if notes else ""
    return f"mod {sp(a)} % {sp(b)} = {t}{rev(a % b)}"


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
    return f"gcd {sp(a)} , {sp(b)} = {t}{rev(_m.gcd(a, b))}"


def gen_sum(rng, d, notes=False):
    k = rng.randint(2, 5)
    xs = [rnd(rng, rng.randint(1, min(d, 4))) for _ in range(k)]
    body = " + ".join(sp(x) for x in xs)
    t = ""
    if notes:
        run, steps = 0, []
        for x in xs:
            run += x
            steps.append(str(run))
        t = _think(" ".join(steps))
    return f"sum {body} = {t}{rev(sum(xs))}"


def gen_round(rng, d, notes=False):
    a = rnd(rng, max(d, 2))
    p = rng.choice([10, 100, 1000])
    t = _think(f"{a}/{p}={a/p:.2f}") if notes else ""
    return f"round {sp(a)} to {sp(p)} = {t}{rev(int(round(a / p)) * p)}"


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
    ap.add_argument("--out", default="data_math")
    ap.add_argument("--tokenizer", default="data/tokenizer.json")
    ap.add_argument("--n", type=int, default=2_000_000)
    ap.add_argument("--val", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--notes-frac", type=float, default=0.3,
                    help="share of examples carrying a <think> scratchpad")
    ap.add_argument("--char", action="store_true",
                    help="character level: no digit spacing, vocab 256")
    args = ap.parse_args()

    global SEP, NOTES_FRAC
    NOTES_FRAC = args.notes_frac
    if args.char:
        SEP = ""

    if args.char:
        from minagi.tokenizer import ByteTokenizer
        tok = ByteTokenizer()
    else:
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(args.tokenizer)
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
            "tokenizer": "byte" if args.char else "bpe",
            "note": "digits space-separated; add/sub/mul answers reversed"}
    json.dump(meta, open(os.path.join(args.out, "meta.json"), "w"), indent=2)
    print(f"wrote {args.out}/meta.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
