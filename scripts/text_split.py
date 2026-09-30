#!/usr/bin/env python
"""Split text into single-batch pieces for F5-TTS.

Model: none (pure text logic; imported by synthesize_f5.py).

F5's multi-batch generation hangs/segfaults on MPS (torch 2.14), so long
texts are split at punctuation into pieces short enough to stay
single-batch. F5 budgets text in UTF-8 BYTES (Cyrillic = 2 bytes/char),
so the cap is in bytes, well under its ~190-byte batch threshold for
11-second references.
"""
import re

MAX_PIECE_BYTES = 150


def _blen(s: str) -> int:
    return len(s.encode("utf-8"))


def split_single_batch(text: str, max_bytes: int = MAX_PIECE_BYTES) -> list[str]:
    parts = re.split(r"(?<=[.!?…;]) +", text)
    pieces, cur = [], ""
    for p in parts:
        if cur and _blen(cur) + 1 + _blen(p) > max_bytes:
            pieces.append(cur)
            cur = p
        else:
            cur = f"{cur} {p}".strip()
        # a single sentence longer than the cap gets split at commas/dashes
        while _blen(cur) > max_bytes:
            limit = len(cur.encode("utf-8")[:max_bytes].decode("utf-8", "ignore"))
            cut = max(cur.rfind(c, 0, limit) for c in ",—:")
            # include the punctuation char in the piece; a hard fallback
            # split at the byte boundary must NOT take an extra char
            end = cut + 1 if cut > 0 else limit
            pieces.append(cur[:end].strip())
            cur = cur[end:].strip()
    if cur:
        pieces.append(cur)
    return [p for p in pieces if p]
