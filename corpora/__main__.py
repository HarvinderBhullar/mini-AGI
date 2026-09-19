"""Build a corpus: python3 -m corpora <code|arithmetic|chat|chess> [args]

Each writes train.bin, val.bin and meta.json into a data_* directory. The
corpora are plain uint16 character streams; nothing about them is tied to a
model, so a corpus outlives any particular architecture.
"""
import sys

BUILDERS = {"code": "corpora.code",
            "arithmetic": "corpora.arithmetic",
            "chat": "corpora.chat",
            "chess": "corpora.chess_games"}

def main():
    if len(sys.argv) < 2 or sys.argv[1] not in BUILDERS:
        print(f"usage: python3 -m corpora <{'|'.join(BUILDERS)}> [options]",
              file=sys.stderr)
        return 2
    name = sys.argv.pop(1)
    import importlib
    return importlib.import_module(BUILDERS[name]).main()

if __name__ == "__main__":
    sys.exit(main())
