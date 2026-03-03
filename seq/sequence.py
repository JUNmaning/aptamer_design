# seq/sequence.py
from __future__ import annotations

def normalize_sequence(seq: str) -> str:
    """Remove spaces/newlines and uppercase."""
    return "".join(seq.split()).upper()

def validate_dna(seq: str) -> None:
    seq = normalize_sequence(seq)
    bad = sorted(set(seq) - set("ACGT"))
    if bad:
        raise ValueError(f"Invalid DNA letters: {bad}. Only A/C/G/T allowed.")

def validate_rna(seq: str) -> None:
    seq = normalize_sequence(seq)
    bad = sorted(set(seq) - set("ACGU"))
    if bad:
        raise ValueError(f"Invalid RNA letters: {bad}. Only A/C/G/U allowed.")

def dna_to_rna(seq: str) -> str:
    """Convert DNA to RNA by T->U."""
    seq = normalize_sequence(seq)
    validate_dna(seq)
    return seq.replace("T", "U")
