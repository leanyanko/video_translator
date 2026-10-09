#!/usr/bin/env python
"""Tiny argv helpers for the stage scripts.

Model: none. Replaces the repeated
    x = sys.argv[sys.argv.index("--flag") + 1] if "--flag" in sys.argv else default
which raises a bare IndexError when the value is missing. Same semantics
otherwise (order-free flags, positionals untouched); argparse would be
the next step once scripts gain main() functions.
"""
import sys


def flag(name: str, argv: list | None = None) -> bool:
    return name in (sys.argv if argv is None else argv)


def opt(name: str, default=None, cast=None, argv: list | None = None):
    """Value following `name`, cast with `cast` (e.g. float); `default`
    when absent. A flag given without a value is a usage error."""
    argv = sys.argv if argv is None else argv
    if name not in argv:
        return default
    i = argv.index(name)
    if i + 1 >= len(argv) or argv[i + 1].startswith("--"):
        sys.exit(f"usage error: {name} needs a value")
    value = argv[i + 1]
    try:
        return cast(value) if cast is not None else value
    except (TypeError, ValueError) as e:
        sys.exit(f"usage error: {name} {value!r}: {e}")
