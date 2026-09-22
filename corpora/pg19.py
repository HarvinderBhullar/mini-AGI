#!/usr/bin/env python3
"""
The PG19 test split, and nothing else.

    python3 -m corpora pg19 --out data/eval/pg19

PG19 is how every byte-level model with a published bits-per-byte is scored -
MambaByte, MegaByte, PerceiverAR, the byte-level Transformer - so a number on
it is the only way this model's loss can be put on the same axis as theirs.

ONLY THE TEST SPLIT. The full dataset is 11.74 GB across 28,752 books; the
test split is 100 books, about 41 MB. The Hugging Face repo is a loading
script rather than data, and it carries the split's file list, so the 100
books can be taken directly from the bucket without touching the other 28,652.

ONE BOOK PER FILE, AND NOTHING INSERTED. The corpus fetcher groups records
into files and writes <|endoftext|> between them; both are right for training
and wrong here. Those markers are real token ids, so they would be scored as
part of the text, and the published numbers are over consecutive bytes of a
book. A book is a file and the bytes are the book's.
"""

import argparse
import os
import sys
import urllib.request

LIST = ("https://huggingface.co/datasets/deepmind/pg19/resolve/main/"
        "data/{split}_files.txt")
ASSETS = "https://storage.googleapis.com/deepmind-gutenberg/"


def fetch(url, timeout=120):
    req = urllib.request.Request(url, headers={"User-Agent": "mini-AGI/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def main():
    ap = argparse.ArgumentParser(prog="python3 -m corpora pg19")
    ap.add_argument("--out", default="data/eval/pg19")
    ap.add_argument("--split", default="test",
                    choices=["test", "validation"],
                    help="test is what the published BPB numbers are on")
    ap.add_argument("--limit", type=int, default=0,
                    help="books to take; 0 for the whole split. The order is "
                         "the file list's, so a limit is a prefix and is "
                         "reproducible rather than a sample")
    a = ap.parse_args()

    names = fetch(LIST.format(split=a.split)).decode().split()
    if a.limit:
        names = names[:a.limit]
    os.makedirs(a.out, exist_ok=True)
    print(f"  {len(names)} books of the {a.split} split -> {a.out}")

    got = total = 0
    for i, name in enumerate(names):
        dest = os.path.join(a.out, os.path.basename(name))
        if os.path.exists(dest) and os.path.getsize(dest) > 0:
            total += os.path.getsize(dest)
            got += 1
            continue
        try:
            body = fetch(ASSETS + name)
        except Exception as e:                                # noqa: BLE001
            print(f"    {name}: {e}", file=sys.stderr)
            continue
        # Written as bytes, not decoded and re-encoded: the model reads bytes
        # and the published numbers are per byte of the file as it stands.
        with open(dest, "wb") as f:
            f.write(body)
        got += 1
        total += len(body)
        if (i + 1) % 10 == 0:
            print(f"    {i+1}/{len(names)}  {total/1e6:.1f} MB", flush=True)

    print(f"  {got} books, {total/1e6:.1f} MB ({total:,} bytes)")
    print(f"  score it:  python3 tools/eval_pg19.py --books {a.out}")
    return 0 if got == len(names) else 1


if __name__ == "__main__":
    sys.exit(main())
