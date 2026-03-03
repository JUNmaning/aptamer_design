from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class ViennaSSResult:
    dot_bracket: str
    mfe_kcal_mol: float
    raw_stdout: str


def rnafold_dotbracket(sequence: str, *, vienna_bin: str = "RNAfold") -> Tuple[str, float]:
    """
    Run ViennaRNA RNAfold and return (dot_bracket, mfe).
    This keeps your existing function name/behavior but makes parsing more robust.
    """
    res = fold_sequence_to_ss(sequence, vienna_bin=vienna_bin)
    return res.dot_bracket, res.mfe_kcal_mol


def fold_sequence_to_ss(sequence: str, *, vienna_bin: str = "RNAfold") -> ViennaSSResult:
    """
    Canonical interface expected by structure/simrna.py (and some older codepaths):
      fold_sequence_to_ss(seq) -> (dot_bracket, mfe)
    Here we return a dataclass with both.
    """
    seq = (sequence or "").strip().upper().replace("T", "U")
    if not seq:
        raise ValueError("Vienna: empty sequence")

    # RNAfold typical usage: echo SEQ | RNAfold --noPS
    # Output usually:
    #   SEQ
    #   ....(((...))).... (-3.40)
    p = subprocess.run(
        [vienna_bin, "--noPS"],
        input=(seq + "\n").encode("utf-8"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    out = (p.stdout or b"").decode("utf-8", errors="ignore")
    err = (p.stderr or b"").decode("utf-8", errors="ignore")
    if p.returncode != 0:
        raise RuntimeError(f"RNAfold failed (code={p.returncode}). stderr:\n{err}\nstdout:\n{out}")

    lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
    if len(lines) < 2:
        raise RuntimeError(f"RNAfold output too short:\n{out}")

    # Find the line containing dot-bracket and energy in parentheses.
    db_line = None
    for ln in lines[1:]:
        if "(" in ln and ")" in ln:
            db_line = ln
            break
    if db_line is None:
        raise RuntimeError(f"RNAfold: cannot find dot-bracket line:\n{out}")

    # Parse: "<dotbracket> ( -3.40 )" or "<dotbracket> (-3.40)"
    m = re.search(r"^([().]+)\s*\(\s*([+-]?\d+(?:\.\d+)?)\s*\)\s*$", db_line)
    if not m:
        raise RuntimeError(f"RNAfold: cannot parse dot-bracket/mfe from line: {db_line}")

    dot = m.group(1)
    mfe = float(m.group(2))
    return ViennaSSResult(dot_bracket=dot, mfe_kcal_mol=mfe, raw_stdout=out)
