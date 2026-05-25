"""
Global label space for SignSpeak AI.

The model produces a probability distribution over `NUM_CLASSES` outputs.
IDs are partitioned so a single softmax can be used for both languages:

    0  .. 28  -> ASL  (A-Z plus SPACE, DELETE, NOTHING)
    29 .. 60  -> ArSL (32 Arabic sign-language letters)
"""

from __future__ import annotations

from typing import Iterable

# ASL Alphabet (Kaggle: grassknoted/asl-alphabet)
ASL_CLASSES: list[str] = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
    "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
    "U", "V", "W", "X", "Y", "Z",
    "SPACE", "DELETE", "NOTHING",
]

# ArSL2018 letters in Arabic transliteration. These match the standard
# directory names of the Kaggle release of Latif et al. 2019.
ARSL_CLASSES: list[str] = [
    "ain", "al", "aleff", "bb", "dal", "dha", "dhad", "fa",
    "gaaf", "ghain", "ha", "haa", "jeem", "kaaf", "khaa", "la",
    "laam", "meem", "nun", "ra", "saad", "seen", "sheen", "ta",
    "taa", "thaa", "thal", "toot", "waw", "ya", "yaa", "zay",
]

# Human-readable Arabic glyphs for UI rendering.
ARSL_GLYPHS: dict[str, str] = {
    "ain": "\u0639", "al": "\u0623\u0644", "aleff": "\u0627",
    "bb": "\u0628", "dal": "\u062f", "dha": "\u0638",
    "dhad": "\u0636", "fa": "\u0641", "gaaf": "\u0642",
    "ghain": "\u063a", "ha": "\u0647", "haa": "\u062d",
    "jeem": "\u062c", "kaaf": "\u0643", "khaa": "\u062e",
    "la": "\u0644\u0627", "laam": "\u0644", "meem": "\u0645",
    "nun": "\u0646", "ra": "\u0631", "saad": "\u0635",
    "seen": "\u0633", "sheen": "\u0634", "ta": "\u062a",
    "taa": "\u0637", "thaa": "\u062b", "thal": "\u0630",
    "toot": "\u0629", "waw": "\u0648", "ya": "\u064a",
    "yaa": "\u0626", "zay": "\u0632",
}

ALL_CLASSES: list[str] = ASL_CLASSES + ARSL_CLASSES
NUM_CLASSES: int = len(ALL_CLASSES)              # 61
NUM_ASL: int = len(ASL_CLASSES)                  # 29
NUM_ARSL: int = len(ARSL_CLASSES)                # 32

ASL_INDICES: list[int] = list(range(0, NUM_ASL))
ARSL_INDICES: list[int] = list(range(NUM_ASL, NUM_CLASSES))


def is_asl(idx: int) -> bool:
    return 0 <= idx < NUM_ASL


def is_arsl(idx: int) -> bool:
    return NUM_ASL <= idx < NUM_CLASSES


def label_of(idx: int) -> str:
    """Return the human-readable label for a class id, including Arabic glyph."""
    if not 0 <= idx < NUM_CLASSES:
        raise ValueError(f"Class id out of range: {idx}")
    name = ALL_CLASSES[idx]
    if is_arsl(idx):
        return f"{ARSL_GLYPHS.get(name, '?')} ({name})"
    return name


def display_label(idx: int) -> str:
    """Compact display label used in the sentence builder."""
    if not 0 <= idx < NUM_CLASSES:
        return "?"
    name = ALL_CLASSES[idx]
    if name == "SPACE":
        return " "
    if name == "DELETE":
        return "\u232b"
    if name == "NOTHING":
        return ""
    if is_arsl(idx):
        return ARSL_GLYPHS.get(name, name)
    return name


def mask_for_language(language: str) -> list[int]:
    """Return the list of class ids that belong to the chosen language.

    ``language`` is one of ``"asl"``, ``"arsl"``, ``"auto"``.
    """
    lang = language.lower()
    if lang == "asl":
        return ASL_INDICES
    if lang == "arsl":
        return ARSL_INDICES
    return list(range(NUM_CLASSES))


def labels_to_text(indices: Iterable[int]) -> str:
    """Concatenate display labels into a single string for TTS output."""
    return "".join(display_label(i) for i in indices)
