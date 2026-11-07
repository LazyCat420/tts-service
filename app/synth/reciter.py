"""Text -> phoneme sequence with stress, syllables and ToBI break indices.

This is the front end for our own synthesizer (app/synth/frames.py +
prosody.py + klatt.py/worldish.py). It replaces the flat phoneme list used by
the legacy SAM demo with a structured sequence: every phone knows which
syllable and word it belongs to, whether it is stressed, and what break index
follows it, which is exactly what the prosody model needs.

The grapheme rules and the pinned lexicon are the ones this repo already uses
in sam-demo/sam.py (our own clean-room rules, no SAM ROM data); the structure
on top of them is new.

    text ──reciter──► [Phone(name, stress, syl, word, boundary), ...]
"""

from __future__ import annotations

import re
from dataclasses import dataclass

VOWELS = {
    "IY", "IH", "EH", "AE", "AA", "AO", "UH", "UW", "AH", "AX", "ER",
    "AY", "EY", "OW", "OY", "AW",
}

# Words that carry no lexical stress of their own in running speech.
FUNCTION_WORDS = {
    "a", "an", "the", "of", "to", "and", "or", "but", "is", "are", "was",
    "were", "be", "been", "am", "in", "on", "at", "for", "with", "by",
    "from", "as", "that", "this", "it", "its", "his", "her", "their",
    "me", "him", "them", "us", "do", "does", "did", "will", "would", "can",
    "could", "should",
}

# Function words keep their full (citation) vowel when stranded at the end of
# a clause ("To be or not to.", "What is it for?") instead of reducing.
UNREDUCED_FINAL = {"to": "T UW", "for": "F AO R", "from": "F R AH M",
                   "of": "AH V", "are": "AA R", "was": "W AH Z"}

WORD_GAP = "_"
SENTENCE_GAP = "__"


@dataclass
class Phone:
    name: str
    stress: int = 0
    syl: int = -1
    word: int = 0
    boundary: int = 0


LEXICON = {
    "hello": "HH AX L OW", "sam": "S AE M", "software": "S AO F T W EH R",
    "automatic": "AO T AX M AE T IH K", "mouth": "M AW TH",
    "robot": "R OW B AO T", "little": "L IH T AX L", "drift": "D R IH F T",
    "king": "K IH NG", "voices": "V OY S IH Z", "own": "OW N",
    "the": "DH AX", "a": "AX", "of": "AX V", "one": "W AH N", "two": "T UW",
    "you": "Y UW", "your": "Y AO R", "are": "ER", "is": "IH Z",
    "was": "W AX Z", "what": "W AH T", "who": "HH UW", "watch": "W AA CH",
    "out": "AW T", "for": "F ER", "that": "DH AE T", "quick": "K W IH K",
    "brown": "B R AW N", "fox": "F AA K S", "jumps": "JH AH M P S",
    "over": "OW V ER", "lazy": "L EY Z IY", "dog": "D AO G", "god": "G AA D",
    "oh": "OW", "my": "M AY", "whoa": "W OW", "needs": "N IY D Z",
    "build": "B IH L D", "our": "AW ER", "from": "F R AX M", "and": "AX N D",
    "to": "T AX", "let": "L EH T", "us": "AH S", "i": "AY", "come": "K AH M",
    "some": "S AH M", "done": "D AH N", "love": "L AH V", "give": "G IH V",
    "have": "HH AE V", "live": "L IH V", "move": "M UW V", "nine": "N AY N",
    "eight": "EY T", "seven": "S EH V AX N", "six": "S IH K S",
    "five": "F AY V", "four": "F AO R", "three": "TH R IY",
    "zero": "Z IH R OW", "ten": "T EH N", "year": "Y IY R",
    "system": "S IH S T AX M", "online": "AA N L AY N",
    "self": "S EH L F", "test": "T EH S T", "complete": "K AX M P L IY T",
    "third": "TH ER D", "check": "CH EH K", "run": "R AH N",
    "cycle": "S AY K AX L", "path": "P AE TH", "clear": "K L IY R",
    "wait": "W EY T", "buy": "B AY", "sell": "S EH L", "now": "N AW",
    "no": "N OW", "me": "M IY", "car": "K AA R", "that's": "DH AE T S",
    "be": "B IY", "he": "HH IY", "she": "SH IY", "we": "W IY",
    "do": "D UW", "go": "G OW", "so": "S OW", "by": "B AY",
    "an": "AX N", "am": "AE M", "in": "IH N", "on": "AA N", "at": "AE T",
    "with": "W IH DH", "not": "N AA T", "said": "S EH D", "can": "K AE N",
    "will": "W IH L", "water": "W AO DX ER", "better": "B EH DX ER",
}

RULES: list[tuple[str, str]] = [
    (r"^kn", "|N|"), (r"^wr", "|R|"), (r"^ps", "|S|"),
    (r"tch", "|CH|"), (r"dge", "|JH|"), (r"ch", "|CH|"), (r"sh", "|SH|"),
    (r"ph", "|F|"), (r"gh$", ""), (r"gh", "|G|"), (r"wh", "|W|"),
    (r"qu", "|K| |W|"), (r"ck", "|K|"), (r"ng", "|NG|"), (r"th", "|TH|"),
    (r"ee", "|IY|"), (r"ea", "|IY|"), (r"oo", "|UW|"), (r"ou", "|AW|"),
    (r"ow", "|OW|"), (r"oa", "|OW|"), (r"oi", "|OY|"), (r"oy", "|OY|"),
    (r"ai", "|EY|"), (r"ay", "|EY|"), (r"au", "|AO|"), (r"aw", "|AO|"),
    (r"ei", "|EY|"), (r"ie", "|IY|"), (r"ue", "|UW|"), (r"ui", "|UW|"),
    (r"ar", "|AA| |R|"), (r"or", "|AO| |R|"), (r"er", "|ER|"),
    (r"ir", "|ER|"), (r"ur", "|ER|"),
    (r"a([bcdfgklmnprstvz]*)ce\b", r"|EY| \1|S|"),
    (r"i([bcdfgklmnprstvz]*)ce\b", r"|AY| \1|S|"),
    (r"o([bcdfgklmnprstvz]*)ce\b", r"|OW| \1|S|"),
    (r"u([bcdfgklmnprstvz]*)ce\b", r"|UW| \1|S|"),
    (r"a([bcdfgklmnprstvz]*)ge\b", r"|EY| \1|JH|"),
    (r"i([bcdfgklmnprstvz]*)ge\b", r"|AY| \1|JH|"),
    (r"o([bcdfgklmnprstvz]*)ge\b", r"|OW| \1|JH|"),
    (r"u([bcdfgklmnprstvz]*)ge\b", r"|Y| |UW| \1|JH|"),
    (r"a([bcdfgklmnprstvz]+)e\b", r"|EY| \1"),
    (r"i([bcdfgklmnprstvz]+)e\b", r"|AY| \1"),
    (r"o([bcdfgklmnprstvz]+)e\b", r"|OW| \1"),
    (r"u([bcdfgklmnprstvz]+)e\b", r"|UW| \1"),
    (r"c(?=[eiy])", "|S|"), (r"c", "|K|"),
    (r"g(?=[eiy])", "|JH|"), (r"g", "|G|"),
    (r"bb", "b"), (r"cc", "c"), (r"dd", "d"), (r"ff", "f"),
    (r"gg", "g"), (r"ll", "l"), (r"mm", "m"), (r"nn", "n"),
    (r"pp", "p"), (r"rr", "r"), (r"ss", "s"), (r"tt", "t"), (r"zz", "z"),
    (r"s(?<=[aeiou])(?=[aeiou])", "|Z|"),
    (r"(?<=[aeiouy|][bcdfghjklmnpqrstvwxz])e$", ""),
    (r"e$", "|IY|"), (r"y$", "|IY|"), (r"y", "|IH|"),
    (r"^[aeiou]", "|AX|"),
]

LETTERS = {
    "a": "AE", "b": "B", "c": "K", "d": "D", "e": "EH", "f": "F", "g": "G",
    "h": "HH", "i": "IH", "j": "JH", "k": "K", "l": "L", "m": "M", "n": "N",
    "o": "AA", "p": "P", "q": "K", "r": "R", "s": "S", "t": "T", "u": "AH",
    "v": "V", "w": "W", "x": "K S", "y": "IH", "z": "Z",
}

DIGITS = ["zero", "one", "two", "three", "four", "five",
          "six", "seven", "eight", "nine"]

_TOKENS = re.compile(r"\|([A-Z]+)\||([a-z])")


def _word_phonemes(word: str) -> list[str]:
    if word in LEXICON:
        out = LEXICON[word].split()
    else:
        s = word
        for pat, rep in RULES:
            s = re.sub(pat, rep, s)
        out = []
        for m in _TOKENS.finditer(s):
            marker, letter = m.group(1), m.group(2)
            if marker:
                out.extend(marker.split())
            elif letter:
                out.extend(LETTERS[letter].split())
        if out and not any(p in VOWELS for p in out):
            # Consonant-only token (e.g. "dr", "tv", "shh", "hmm"): insert a
            # neutral schwa nucleus so every word forms a valid syllable.
            out.append("AX")
    # Intervocalic alveolar flapping: T/D between a full vowel and a
    # weak/reduced vowel becomes a flap (American English "water", "little",
    # "automatic").
    if len(out) >= 3:
        for idx in range(1, len(out) - 1):
            if (out[idx] in ("T", "D") and out[idx - 1] in VOWELS
                    and out[idx + 1] in ("AX", "IH", "ER")):
                out[idx] = "DX"
    return out


def reciter(text: str) -> list[Phone]:
    """English text -> structured phones.

    `stress`: 0 unstressed, 1 stressed, 2 nuclear (last stressed syllable of
    the final clause). `boundary`: ToBI break index after the phone (0 none,
    3 clause comma/semicolon/colon, 4 sentence end).
    """
    text = re.sub(r"\d", lambda m: f" {DIGITS[int(m.group())]} ", text.lower())

    # Split into clauses, remembering what punctuation ended each one.
    parts = re.split(r"([.!?;:,]+)", text)
    clauses: list[tuple[str, int]] = []
    for i in range(0, len(parts) - 1, 2):
        body, punct = parts[i], parts[i + 1]
        if not body.strip():
            continue
        bi = 4 if any(c in punct for c in ".!?") else 3
        clauses.append((body, bi))
    if parts[-1].strip():                      # trailing text with no punctuation
        clauses.append((parts[-1], 4))
    if not clauses:
        return []

    phones: list[Phone] = []
    syl_counter = 0
    word_counter = 0
    for ci, (body, bi) in enumerate(clauses):
        words = [w.replace("'", "") for w in re.findall(r"[a-z']+", body)]
        words = [w for w in words if w]
        if not words:
            continue
        clause_start = len(phones)
        for wi_idx, word in enumerate(words):
            phs = _word_phonemes(word)
            if not phs:
                continue
            wi = word_counter
            word_counter += 1
            stressed = word not in FUNCTION_WORDS
            if wi_idx == len(words) - 1 and word in UNREDUCED_FINAL:
                phs = UNREDUCED_FINAL[word].split()
                stressed = True
            vowel_positions = [k for k, p in enumerate(phs) if p in VOWELS]
            # Primary stress: the first vowel of a content word, except that
            # two-syllable words with a final long vowel keep it on that vowel.
            stress_at = vowel_positions[0] if vowel_positions else -1
            if len(vowel_positions) >= 2 and phs[vowel_positions[-1]] in {
                    "IY", "UW", "OW", "EY", "AY"}:
                stress_at = vowel_positions[-1]
            first_syl = syl_counter
            for k, name in enumerate(phs):
                is_vowel = name in VOWELS
                if is_vowel:
                    syl_counter += 1
                phones.append(Phone(
                    name=name,
                    stress=1 if (stressed and k == stress_at) else 0,
                    syl=syl_counter - 1 if is_vowel else max(first_syl, syl_counter - 1),
                    word=wi,
                ))
            # Words inside a clause connect directly: no inter-word silence.
        # Every clause needs an accent peak: if no vowel in this clause got
        # stress (e.g. a clause of nothing but function words), accent the
        # last vowel.
        clause_vowels = [p for p in phones[clause_start:] if p.name in VOWELS]
        if clause_vowels and not any(p.stress > 0 for p in clause_vowels):
            clause_vowels[-1].stress = 1
        # Nuclear stress: the last stressed vowel of the final clause.
        if ci == len(clauses) - 1:
            for p in reversed(phones):
                if p.stress == 1:
                    p.stress = 2
                    break
        if ci < len(clauses) - 1:
            phones.append(Phone(SENTENCE_GAP, 0, -1, phones[-1].word if phones else 0))
        # Break index on the last phone of the clause (gaps carry it too).
        if phones:
            phones[-1].boundary = bi
    return phones


if __name__ == "__main__":
    import sys
    for p in reciter(" ".join(sys.argv[1:]) or "Hello. I am SAM."):
        print(p)
