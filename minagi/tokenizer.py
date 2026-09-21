"""Bytes in, bytes out.

The vocabulary is the 256 byte values plus a handful of structural markers.
There is nothing to fall outside it, so a new kind of data needs no new
vocabulary - only the markers ever have to be agreed on.
"""

import json
import os



class _Enc:
    __slots__ = ("ids",)

    def __init__(self, ids):
        self.ids = ids


# Markers that mean something structural get their own token rather than being
# spelled out. "<think>" as seven bytes costs seven positions and forces the
# model to learn the literal character sequence; as one token it is a single
# dedicated embedding meaning "working this out", which is what it actually is.
SPECIALS = ["<think>", "</think>", "<user>", "</user>", "<bot>", "</bot>",
            "<g>", "</g>", "<|endoftext|>"]
SPECIAL_ID = {s: 256 + i for i, s in enumerate(SPECIALS)}
ID_SPECIAL = {v: k for k, v in SPECIAL_ID.items()}
import re as _re
_SPECIAL_RE = _re.compile("|".join(_re.escape(s) for s in SPECIALS))


class ByteTokenizer:
    """
    Character-level over raw UTF-8 bytes, plus a handful of structural tokens.

    Vocabulary is 256 bytes and 9 markers. Everything that is ordinary text is
    still one token per byte, so there is no such thing as an unknown character
    and every digit is its own token. The markers are the exception because
    they are not text the model is reading - they are signals about what it is
    doing, and a signal should not have to be spelled.
    """

    size = 256 + len(SPECIALS)

    def encode(self, text):
        ids, pos = [], 0
        for m in _SPECIAL_RE.finditer(text):
            ids.extend(text[pos:m.start()].encode("utf-8"))
            ids.append(SPECIAL_ID[m.group(0)])
            pos = m.end()
        ids.extend(text[pos:].encode("utf-8"))
        return _Enc(ids)

    def encode_batch(self, texts):
        return [self.encode(t) for t in texts]

    def decode(self, ids):
        out, buf = [], bytearray()
        for i in ids:
            i = int(i)
            if i in ID_SPECIAL:
                if buf:
                    out.append(buf.decode("utf-8", errors="replace"))
                    buf = bytearray()
                out.append(ID_SPECIAL[i])
            elif 0 <= i < 256:
                buf.append(i)
        if buf:
            out.append(buf.decode("utf-8", errors="replace"))
        return "".join(out)

    def get_vocab_size(self):
        return self.size

    def token_to_id(self, token):
        if token in SPECIAL_ID:
            return SPECIAL_ID[token]
        b = token.encode("utf-8")
        return b[0] if len(b) == 1 else None


def load_tokenizer(data_dir):
    meta_path = os.path.join(data_dir, "meta.json")
    if os.path.exists(meta_path):
        meta = json.load(open(meta_path))
        if meta.get("tokenizer") == "byte":
            return ByteTokenizer()
    from tokenizers import Tokenizer
    return Tokenizer.from_file(os.path.join(data_dir, "tokenizer.json"))


