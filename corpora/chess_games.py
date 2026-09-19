#!/usr/bin/env python3
"""
Build a chess corpus from the Lichess database.

FORMAT
    <g 1850 1-0> 1.e4 e5 2.Nf3 Nc6 3.Bb5 a6 ... </g>

Three decisions, each with a reason:

* SAN with move numbers, not UCI. SAN is what the training text of the world
  looks like and it is shorter; the move number gives the model an explicit ply
  counter, which it otherwise has to infer.

* The players' rating is in the header. At inference you prompt with a rating,
  and the model plays like that rating - conditioning on strength is what lets
  a single model play weakly or well on request. This is the trick from the
  search-free grandmaster-chess work.

* The result is in the header too, so the model can be asked for the moves of a
  game that White went on to win.

WHAT TO EXPECT
Chess is the most reachable of the hard targets at this size. A ~50M-parameter
model trained on PGN reaches roughly 1500 Elo in published work, and legal-move
rate above 99% is normal. The metric that matters first is legality: a model
that plays illegal moves is not playing chess, however good the moves look.

    python3 chess_data.py --months 2014-01 --min-elo 1600
"""

import argparse
import io
import os
import re
import sys
import time
import urllib.request

import numpy as np
import zstandard

BASE = "https://database.lichess.org/standard/lichess_db_standard_rated_%s.pgn.zst"

# PGN is regular enough that a regex beats a real parser by ~50x here, and the
# result is validated against python-chess on a sample afterwards.
RE_ELO = re.compile(r'\[(White|Black)Elo "(\d+)"\]')
RE_RESULT = re.compile(r'\[Result "([^"]+)"\]')
RE_TERM = re.compile(r'\[Termination "([^"]+)"\]')
# strip clock/eval comments and NAGs, keep the moves
RE_CLEAN = re.compile(r"\{[^}]*\}|\$\d+|\?|!")
RE_MOVETEXT = re.compile(r"^1\.")


def stream_games(url, chunk_mb=8):
    """Yield raw PGN game blocks from a remote .zst without landing the file."""
    dctx = zstandard.ZstdDecompressor()
    req = urllib.request.Request(url, headers={"User-Agent": "agi-chess/1.0"})
    with urllib.request.urlopen(req) as resp:
        with dctx.stream_reader(resp) as reader:
            buf = ""
            while True:
                raw = reader.read(chunk_mb * 1024 * 1024)
                if not raw:
                    break
                buf += raw.decode("utf-8", errors="ignore")
                # games are separated by a blank line after the movetext
                while True:
                    i = buf.find("\n\n[Event ")
                    if i < 0:
                        break
                    yield buf[:i]
                    buf = buf[i + 2:]
            if buf.strip():
                yield buf


def convert(block, min_elo, max_elo, min_moves, max_moves):
    elos = {m.group(1): int(m.group(2)) for m in RE_ELO.finditer(block)}
    if len(elos) != 2:
        return None
    lo = min(elos.values())
    if lo < min_elo or max(elos.values()) > max_elo:
        return None
    r = RE_RESULT.search(block)
    if not r or r.group(1) not in ("1-0", "0-1", "1/2-1/2"):
        return None
    term = RE_TERM.search(block)
    if term and term.group(1) not in ("Normal", "Time forfeit"):
        return None

    line = None
    for ln in block.split("\n"):
        if RE_MOVETEXT.match(ln.strip()):
            line = ln.strip()
            break
    if not line:
        return None
    moves = RE_CLEAN.sub("", line)
    moves = re.sub(r"\s+", " ", moves).strip()
    for tail in ("1-0", "0-1", "1/2-1/2", "*"):
        if moves.endswith(tail):
            moves = moves[:-len(tail)].strip()
    n_moves = len(re.findall(r"\d+\.", moves))
    if n_moves < min_moves or n_moves > max_moves:
        return None
    # bucket the rating: the model does not need 4-digit resolution and
    # coarse buckets give each one far more examples to learn from
    elo = int(round(lo / 100.0) * 100)
    return f"<g {elo} {r.group(1)}> {moves} </g>\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", nargs="+", default=["2014-01"])
    ap.add_argument("--out", default="data_chess_char")
    ap.add_argument("--min-elo", type=int, default=1600)
    ap.add_argument("--max-elo", type=int, default=3000)
    ap.add_argument("--min-moves", type=int, default=10)
    ap.add_argument("--max-moves", type=int, default=140)
    ap.add_argument("--max-games", type=int, default=1_000_000)
    ap.add_argument("--val-games", type=int, default=4000)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    txt_path = os.path.join(args.out, "games.txt")
    kept = seen = 0
    t0 = time.time()
    with open(txt_path, "w", encoding="utf-8") as out:
        for month in args.months:
            print(f"streaming {month} ...", flush=True)
            try:
                for block in stream_games(BASE % month):
                    seen += 1
                    g = convert(block, args.min_elo, args.max_elo,
                                args.min_moves, args.max_moves)
                    if g:
                        out.write(g)
                        kept += 1
                        if kept % 20000 == 0:
                            print(f"\r  {kept:,} kept / {seen:,} seen "
                                  f"({time.time()-t0:.0f}s)", end="",
                                  file=sys.stderr)
                    if kept >= args.max_games:
                        break
            except Exception as e:
                print(f"\n  {month}: stopped ({type(e).__name__}: {e})")
            if kept >= args.max_games:
                break
    print(f"\nkept {kept:,} of {seen:,} games -> {txt_path}")
    if kept == 0:
        return 1

    # legality check on a sample: if the text is not real chess, stop here
    import chess, chess.pgn
    bad = checked = 0
    with open(txt_path) as f:
        for i, line in enumerate(f):
            if i >= 200:
                break
            mv = line.split("> ", 1)[1].rsplit(" </g>", 1)[0]
            board = chess.Board()
            try:
                for tok in mv.split():
                    if tok.endswith("."):
                        continue
                    tok = tok.split(".")[-1]
                    if tok:
                        board.push_san(tok)
                checked += 1
            except Exception:
                bad += 1
    print(f"legality check: {checked} of {checked+bad} sampled games replay cleanly")

    from minagi.tokenizer import ByteTokenizer
    tok = ByteTokenizer()
    tr = os.path.join(args.out, "train.bin")
    va = os.path.join(args.out, "val.bin")
    n_tr = n_va = 0
    with open(txt_path) as f, open(tr, "wb") as ftr, open(va, "wb") as fva:
        for i, line in enumerate(f):
            a = np.array(tok.encode(line).ids, dtype=np.uint16)
            if i < args.val_games:
                fva.write(a.tobytes()); n_va += len(a)
            else:
                ftr.write(a.tobytes()); n_tr += len(a)
    import json
    json.dump({"vocab_size": 256, "tokenizer": "byte",
               "train_tokens": n_tr, "val_tokens": n_va, "games": kept,
               "min_elo": args.min_elo, "format": "<g ELO RESULT> SAN </g>"},
              open(os.path.join(args.out, "meta.json"), "w"), indent=2)
    print(f"train {n_tr:,} tokens, val {n_va:,} tokens -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
