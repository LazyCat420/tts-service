"""Clean-room SAM-family formant speech synthesizer.

Nothing here is copied from SAM's ROM or from the s-macke/SAM port: the reciter
rules, the phoneme table and the renderer are written from the published
description of the algorithm. What SAM does, in one sentence: a buzzy glottal
pulse train is shaped by three cascaded formant resonators whose frequencies
and amplitudes are table-driven per phoneme, and unvoiced consonants switch the
source from the buzz to noise.

    glottal buzz (f0)  ─┐
                        ├─→ F1 ─→ F2 ─→ F3 ─→ out      (cascade)
    noise (fricatives) ─┘
                        └─→ F1, F2, F3 in parallel, summed with gains (default)

Four knobs, with SAM's own semantics (measured on the ported engine):

    pitch   0-255   base f0
    speed   0-255   phoneme duration (higher = faster)
    throat  0-255   F2 scale  (factor 128 = unchanged, linear)
    mouth   0-255   F1 scale  (factor 128 = unchanged, linear)

Usage:
    python3 sam.py -wav out.wav -pitch 60 -speed 92 -throat 190 -mouth 190 "I am a robot"
"""

from __future__ import annotations

import argparse
import math
import random
import re
import sys
import wave

SR = 22050

# ── Phoneme table ───────────────────────────────────────────────────────────
# name -> (F1, F2, F3, duration_s, voiced_amp, noise_amp, noise_hz)
# Adult-male formants from standard phonetics tables. The buzz through these
# values is what makes the SAM timbre; the exact numbers only move the vowel.

V = {  # vowels and diphthong targets
    "IY": (270, 2290, 3010, 0.105, 1.00, 0.0, 0),
    "IH": (390, 1990, 2550, 0.070, 0.95, 0.0, 0),
    "EH": (530, 1840, 2480, 0.090, 0.95, 0.0, 0),
    "AE": (660, 1720, 2410, 0.115, 1.00, 0.0, 0),
    "AA": (730, 1090, 2440, 0.115, 1.00, 0.0, 0),
    "AO": (570,  840, 2410, 0.115, 1.00, 0.0, 0),
    "UH": (440, 1020, 2240, 0.075, 0.90, 0.0, 0),
    "UW": (300,  870, 2240, 0.100, 0.90, 0.0, 0),
    "AH": (640, 1190, 2390, 0.085, 0.95, 0.0, 0),
    "AX": (500, 1500, 2500, 0.055, 0.75, 0.0, 0),
    "ER": (490, 1350, 1690, 0.100, 0.95, 0.0, 0),
    # nasals and liquids
    "M":  (280,  900, 2200, 0.080, 0.60, 0.0, 0),
    "N":  (280, 1700, 2600, 0.075, 0.60, 0.0, 0),
    "NG": (280, 2300, 2750, 0.080, 0.60, 0.0, 0),
    "L":  (360, 1300, 2700, 0.070, 0.70, 0.0, 0),
    "R":  (460, 1100, 1600, 0.070, 0.70, 0.0, 0),
    "W":  (290,  610, 2150, 0.060, 0.60, 0.0, 0),
    "Y":  (260, 2070, 3020, 0.060, 0.60, 0.0, 0),
    # unvoiced fricatives
    "S":  (320, 3800, 5000, 0.090, 0.00, 0.55, 6200),
    "SH": (300, 2500, 3200, 0.095, 0.00, 0.50, 2900),
    "F":  (400, 1200, 2400, 0.085, 0.00, 0.30, 4500),
    "TH": (400, 1400, 2600, 0.080, 0.00, 0.25, 5800),
    "HH": (500, 1500, 2500, 0.070, 0.00, 0.22, 1800),
    "CH": (320, 2200, 3000, 0.095, 0.00, 0.50, 2900),
    # voiced fricatives
    "Z":  (320, 3800, 5000, 0.085, 0.45, 0.35, 6200),
    "ZH": (300, 2400, 3100, 0.090, 0.45, 0.30, 2900),
    "V":  (400, 1200, 2400, 0.080, 0.50, 0.22, 4500),
    "DH": (400, 1400, 2600, 0.070, 0.50, 0.18, 5000),
    "JH": (300, 2200, 3000, 0.085, 0.45, 0.30, 2900),
    # stops: rendered as closure + burst
    "P":  (300, 1000, 2300, 0.090, 0.00, 0.60, 1200),
    "B":  (300, 1000, 2300, 0.085, 0.35, 0.45, 1200),
    "T":  (320, 1700, 2600, 0.085, 0.00, 0.60, 3800),
    "D":  (320, 1700, 2600, 0.080, 0.35, 0.45, 3800),
    "K":  (350, 2000, 2800, 0.090, 0.00, 0.60, 2300),
    "G":  (350, 2000, 2800, 0.085, 0.35, 0.45, 2300),
}
STOPS = {"P", "B", "T", "D", "K", "G"}

# Diphthongs: (start phoneme, end phoneme, duration)
DIPHTHONGS = {
    "AY": ("AA", "IY", 0.165),
    "EY": ("EH", "IY", 0.150),
    "OW": ("AO", "UW", 0.155),
    "OY": ("AO", "IY", 0.170),
    "AW": ("AA", "UW", 0.160),
}

# ── Reciter: letters → phonemes ─────────────────────────────────────────────
# Rules are applied in order to the lowercase letter string. Replacement text
# is written as |-delimited UPPERCASE markers, so later letter rules cannot
# re-match inside a marker. Longest, most specific patterns come first.

LEXICON = {
    # words that English spelling rules get wrong often enough to be worth
    # pinning; a real reciter carries thousands of these
    "hello": "HH AX L OW",
    "sam": "S AE M",
    "software": "S AO F T W EH R",
    "automatic": "AO T AX M AE T IH K",
    "mouth": "M AW TH",
    "robot": "R OW B AO T",
    "little": "L IH T AX L",
    "drift": "D R IH F T",
    "king": "K IH NG",
    "voices": "V OY S IH Z",
    "own": "OW N",
    "the": "DH AX",
    "a": "AX",
    "of": "AH V",
    "one": "W AH N",
    "two": "T UW",
    "you": "Y UW",
    "your": "Y AO R",
    "are": "AA R",
    "is": "IH Z",
    "was": "W AH Z",
    "what": "W AH T",
    "who": "HH UW",
    "watch": "W AA CH",
    "out": "AW T",
    "for": "F AO R",
    "that": "DH AE T",
    "quick": "K W IH K",
    "brown": "B R AW N",
    "fox": "F AA K S",
    "jumps": "JH AH M P S",
    "over": "OW V ER",
    "lazy": "L EY Z IY",
    "dog": "D AO G",
    "god": "G AA D",
    "oh": "OW",
    "my": "M AY",
    "whoa": "W OW",
    "needs": "N IY D Z",
    "build": "B IH L D",
    "our": "AW ER",
    "from": "F R AH M",
    "and": "AE N D",
    "to": "T UW",
    "let": "L EH T",
    "us": "AH S",
    "i": "AY",
    "come": "K AH M",
    "some": "S AH M",
    "done": "D AH N",
    "love": "L AH V",
    "give": "G IH V",
    "have": "HH AE V",
    "live": "L IH V",
    "move": "M UW V",
    "nine": "N AY N",
    "eight": "EY T",
    "seven": "S EH V AX N",
    "six": "S IH K S",
    "five": "F AY V",
    "four": "F AO R",
    "three": "TH R IY",
    "zero": "Z IH R OW",
    "ten": "T EH N",
    "year": "Y IY R",
}

RULES: list[tuple[str, str]] = [
    # silent / irregular consonant clusters
    (r"^kn", "|N|"),
    (r"^wr", "|R|"),
    (r"^ps", "|S|"),
    (r"tch", "|CH|"),
    (r"dge", "|JH|"),
    (r"ch", "|CH|"),
    (r"sh", "|SH|"),
    (r"ph", "|F|"),
    (r"gh$", ""),
    (r"gh", "|G|"),
    (r"wh", "|W|"),
    (r"qu", "|K| |W|"),
    (r"ck", "|K|"),
    (r"ng", "|NG|"),
    (r"th", "|TH|"),
    # vowel digraphs
    (r"ee", "|IY|"),
    (r"ea", "|IY|"),
    (r"oo", "|UW|"),
    (r"ou", "|AW|"),
    (r"ow", "|OW|"),
    (r"oa", "|OW|"),
    (r"oi", "|OY|"),
    (r"oy", "|OY|"),
    (r"ai", "|EY|"),
    (r"ay", "|EY|"),
    (r"au", "|AO|"),
    (r"aw", "|AO|"),
    (r"ei", "|EY|"),
    (r"ie", "|IY|"),
    (r"ue", "|UW|"),
    (r"ui", "|UW|"),
    # r-coloured vowels
    (r"ar", "|AA| |R|"),
    (r"or", "|AO| |R|"),
    (r"er", "|ER|"),
    (r"ir", "|ER|"),
    (r"ur", "|ER|"),
    # magic e with a soft c / g ending: nice, page, race, huge
    (r"a([bcdfgklmnprstvz]*)ce\b", r"|EY| \1|S|"),
    (r"i([bcdfgklmnprstvz]*)ce\b", r"|AY| \1|S|"),
    (r"o([bcdfgklmnprstvz]*)ce\b", r"|OW| \1|S|"),
    (r"u([bcdfgklmnprstvz]*)ce\b", r"|UW| \1|S|"),
    (r"a([bcdfgklmnprstvz]*)ge\b", r"|EY| \1|JH|"),
    (r"i([bcdfgklmnprstvz]*)ge\b", r"|AY| \1|JH|"),
    (r"o([bcdfgklmnprstvz]*)ge\b", r"|OW| \1|JH|"),
    (r"u([bcdfgklmnprstvz]*)ge\b", r"|Y| |UW| \1|JH|"),
    # magic e: a_e / i_e / o_e / u_e
    (r"a([bcdfgklmnprstvz]+)e\b", r"|EY| \1"),
    (r"i([bcdfgklmnprstvz]+)e\b", r"|AY| \1"),
    (r"o([bcdfgklmnprstvz]+)e\b", r"|OW| \1"),
    (r"u([bcdfgklmnprstvz]+)e\b", r"|UW| \1"),
    # soft c / g
    (r"c(?=[eiy])", "|S|"),
    (r"c", "|K|"),
    (r"g(?=[eiy])", "|JH|"),
    (r"g", "|G|"),
    # double consonants collapse
    (r"bb", "b"), (r"cc", "c"), (r"dd", "d"), (r"ff", "f"),
    (r"gg", "g"), (r"ll", "l"), (r"mm", "m"), (r"nn", "n"),
    (r"pp", "p"), (r"rr", "r"), (r"ss", "s"), (r"tt", "t"), (r"zz", "z"),
    # s between vowels is voiced
    (r"s(?<=[aeiou])(?=[aeiou])", "|Z|"),
    # final e is silent, final y is IY
    (r"e$", ""),
    (r"y$", "|IY|"),
    (r"y", "|IH|"),
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

# A rule's output is |PHONEME| markers; this picks them back out, in order,
# interleaved with any plain letters the rules left behind.
_TOKENS = re.compile(r"\|([A-Z]+)\||([a-z])")

WORD_GAP = "_"      # 70 ms between words
SENTENCE_GAP = "__"  # 250 ms between sentences


def reciter(text: str) -> list[str]:
    """English text → phoneme names."""
    text = re.sub(r"\d", lambda m: f" {DIGITS[int(m.group())]} ", text.lower())
    out: list[str] = []
    sentences = [s for s in re.split(r"[.!?;:,]+", text) if s.strip()]
    for si, sentence in enumerate(sentences):
        words = [w.replace("'", "") for w in re.findall(r"[a-z']+", sentence)]
        words = [w for w in words if w]
        for wi, word in enumerate(words):
            if word in LEXICON:
                out.extend(LEXICON[word].split())
            else:
                s = word
                for pat, rep in RULES:
                    s = re.sub(pat, rep, s)
                for m in _TOKENS.finditer(s):
                    marker, letter = m.group(1), m.group(2)
                    if marker:
                        out.extend(marker.split())
                    elif letter:
                        out.extend(LETTERS[letter].split())
            if wi < len(words) - 1:
                out.append(WORD_GAP)
        if si < len(sentences) - 1:
            out.append(SENTENCE_GAP)
    return out


# ── Renderer ────────────────────────────────────────────────────────────────

def _resonator(f: float, bw: float):
    r = math.exp(-math.pi * bw / SR)
    b1 = 2.0 * r * math.cos(2.0 * math.pi * f / SR)
    b2 = -r * r
    a0 = 1.0 - b1 - b2
    return a0, b1, b2


class _Res:
    __slots__ = ("a0", "b1", "b2", "y1", "y2")

    def __init__(self):
        self.y1 = self.y2 = 0.0
        self.a0 = self.b1 = self.b2 = 0.0

    def set(self, f: float, bw: float):
        self.a0, self.b1, self.b2 = _resonator(f, bw)

    def run(self, x: float) -> float:
        y = self.a0 * x + self.b1 * self.y1 + self.b2 * self.y2
        self.y2 = self.y1
        self.y1 = y
        return y


def _poly_blep(t: float, dt: float) -> float:
    if t < dt:
        t /= dt
        return t + t - t * t - 1.0
    if t > 1.0 - dt:
        t = (t - 1.0) / dt
        return t * t + t + t + 1.0
    return 0.0


class Biquad:
    """RBJ cookbook biquad, used for the output band-limiting."""

    __slots__ = ("b0", "b1", "b2", "a1", "a2", "x1", "x2", "y1", "y2")

    def __init__(self):
        self.b0 = self.b1 = self.b2 = self.a1 = self.a2 = 0.0
        self.x1 = self.x2 = self.y1 = self.y2 = 0.0

    def lowpass(self, f: float, q: float = 0.7071):
        w = 2.0 * math.pi * f / SR
        c, s = math.cos(w), math.sin(w)
        alpha = s / (2.0 * q)
        a0 = 1.0 + alpha
        self.b0 = (1.0 - c) / 2.0 / a0
        self.b1 = (1.0 - c) / a0
        self.b2 = self.b0
        self.a1 = -2.0 * c / a0
        self.a2 = (1.0 - alpha) / a0

    def highpass(self, f: float, q: float = 0.7071):
        w = 2.0 * math.pi * f / SR
        c, s = math.cos(w), math.sin(w)
        alpha = s / (2.0 * q)
        a0 = 1.0 + alpha
        self.b0 = (1.0 + c) / 2.0 / a0
        self.b1 = -(1.0 + c) / a0
        self.b2 = self.b0
        self.a1 = -2.0 * c / a0
        self.a2 = (1.0 - alpha) / a0

    def run(self, x: float) -> float:
        y = (self.b0 * x + self.b1 * self.x1 + self.b2 * self.x2
             - self.a1 * self.y1 - self.a2 * self.y2)
        self.x2, self.x1 = self.x1, x
        self.y2, self.y1 = self.y1, y
        return y


def _alpha(tau_ms: float) -> float:
    """One-pole coefficient for a time constant in ms: a = 1 - e^(-1/(tau·fs))."""
    return 1.0 - math.exp(-1.0 / (tau_ms * 1e-3 * SR))


# Per-formant gains for the parallel branch (Klatt-style). The cascade branch
# normalises every resonator to unity DC gain, so the product peaks as
# (F/bw)^3 — at high throat values F3 then dominates and the voice screams.
# Explicit gains keep the formants in their intended balance.
FORMANT_GAIN = (1.0, 0.55, 0.30)


def _bandwidth(f: float) -> float:
    """Formant bandwidth grows with centre frequency: 6% + 50 Hz floor.

    A fixed 70/110/180 Hz is right only near the adult-male default; scaled far
    up it rings (high Q), scaled far down the formants merge into buzz.
    """
    return max(50.0, 0.06 * f + 50.0)


def synth(phonemes: list[str], pitch=64, speed=72, throat=128, mouth=128,
          sr: int = SR, seed: int = 7, source: str = "saw",
          lp_hz: float = 4800.0, hp_hz: float = 80.0,
          noise_gain: float = 0.35, noise_bw: float = 700.0,
          engine: str = "parallel") -> list[float]:
    """Render a phoneme list to floats in [-1, 1].

    `engine`: "parallel" (default, Klatt parallel branch) or "cascade".
    `source`: "saw" (glottal buzz), "square" (chip voice) or "pulse".
    `lp_hz`/`hp_hz`: output band-limiting; 0 disables.
    """
    f0_base = 20.0 + pitch * 1.15          # pitch 64 → ~94 Hz
    mouth_scale = mouth / 128.0            # SAM: mouth scales F1
    throat_scale = throat / 128.0          # SAM: throat scales F2, F3 untouched
    dur_scale = 72.0 / max(speed, 1)

    # Flatten the phoneme list into (f1, f2, f3, voiced, noise, noise_hz, dur)
    segs: list[tuple[float, float, float, float, float, float, float]] = []
    for p in phonemes:
        if p == WORD_GAP:
            segs.append((500.0, 1500.0, 2500.0, 0.0, 0.0, 0.0, 0.070))
            continue
        if p == SENTENCE_GAP:
            segs.append((500.0, 1500.0, 2500.0, 0.0, 0.0, 0.0, 0.250))
            continue
        if p in DIPHTHONGS:
            a, b, d = DIPHTHONGS[p]
            segs.append((V[a][0], V[a][1], V[a][2], V[a][4], 0.0, 0.0, d / 2))
            segs.append((V[b][0], V[b][1], V[b][2], V[b][4], 0.0, 0.0, d / 2))
            continue
        f1, f2, f3, dur, va, na, nh = V[p]
        if p in STOPS:
            # closure (silence), then the burst
            segs.append((f1, f2, f3, 0.0, 0.0, 0.0, dur * 0.6))
            segs.append((f1, f2, f3, va, na, nh, dur * 0.4))
            continue
        segs.append((f1, f2, f3, va, na, nh, dur))

    rng = random.Random(seed)
    f_res = [_Res(), _Res(), _Res()]
    n_res = _Res()
    lp, hp = Biquad(), Biquad()
    if lp_hz:
        lp.lowpass(lp_hz)
    if hp_hz:
        hp.highpass(hp_hz)

    f1 = segs[0][0]
    f2 = segs[0][1]
    f3 = segs[0][2]
    va = segs[0][3]
    na = segs[0][4]
    nh = segs[0][5] if segs[0][5] else 2000.0
    nh = float(nh)

    total = sum(s[6] for s in segs) * dur_scale
    out: list[float] = []
    phase = 0.0
    fs = _alpha(15.0)  # formant glide: 15 ms time constant
    asm = _alpha(5.0)  # amplitude glide: 5 ms (0.12 was ~0.36 ms → roughness)
    elapsed = 0.0

    for seg in segs:
        f1t, f2t, f3t, vat, nat, nht, dur = seg
        n = max(1, int(dur * dur_scale * sr))
        for _ in range(n):
            f1 += (f1t - f1) * fs
            f2 += (f2t - f2) * fs
            f3 += (f3t - f3) * fs
            va += (vat - va) * asm
            na += (nat - na) * asm
            if nht:
                nh += (float(nht) - nh) * fs

            # pitch: gentle declination plus a little jitter
            prog = elapsed / max(total, 1e-6)
            f0 = f0_base * (1.0 - 0.18 * prog)
            f0 *= 1.0 + 0.008 * math.sin(2 * math.pi * 5.2 * elapsed)
            f0 *= 1.0 + 0.003 * rng.uniform(-1.0, 1.0)

            phase += f0 / sr
            if phase >= 1.0:
                phase -= 1.0
            dt = f0 / sr
            if source == "square":
                buzz = (1.0 if phase < 0.5 else -1.0) \
                    + _poly_blep(phase, dt) - _poly_blep((phase + 0.5) % 1.0, dt)
            elif source == "pulse":
                # Rosenberg glottal pulse: 40% open phase, 16% return phase
                tp, tn = 0.4, 0.16
                if phase < tp:
                    buzz = 0.5 * (1.0 - math.cos(math.pi * phase / tp))
                elif phase < tp + tn:
                    buzz = math.cos(math.pi * (phase - tp) / (2.0 * tn))
                else:
                    buzz = 0.0
            else:
                buzz = 2.0 * phase - 1.0 - _poly_blep(phase, dt)

            # Formants, kept ordered and away from Nyquist. Crossing F1/F2 is
            # what turns a scaled-up preset into noise: the resonators merge.
            fa = f1 * mouth_scale
            fb = f2 * throat_scale
            fc = f3
            fa = max(150.0, min(fa, fb - 120.0))
            fb = max(fa + 120.0, min(fb, fc - 150.0))
            fc = max(fb + 150.0, min(fc, 0.45 * SR))

            if engine == "cascade":
                f_res[0].set(fa, _bandwidth(fa))
                f_res[1].set(fb, _bandwidth(fb))
                f_res[2].set(fc, _bandwidth(fc))
                y = f_res[0].run(buzz * va)
                y = f_res[1].run(y)
                y = f_res[2].run(y)
            else:
                y = 0.0
                for k, fk in enumerate((fa, fb, fc)):
                    f_res[k].set(fk, _bandwidth(fk))
                    y += f_res[k].run(buzz * va) * FORMANT_GAIN[k]

            if na > 1e-4:
                n_res.set(nh, noise_bw)
                y += n_res.run(rng.uniform(-1.0, 1.0) * na) * noise_gain

            # Band-limit the output: without this the raw noise path leaves
            # 6% of the energy above 5 kHz and thousands of 25%-of-peak sample
            # steps (measured) — that is the "distorted hiss".
            if lp_hz:
                y = lp.run(y)
            if hp_hz:
                y = hp.run(y)
            out.append(y)
            elapsed += 1.0 / sr

    peak = max(1e-9, max(abs(v) for v in out))
    return [v / peak * 0.85 for v in out]


def silence(seconds: float) -> list[float]:
    return [0.0] * int(seconds * SR)


def write_wav(path: str, samples: list[float], sr: int = SR):
    frames = bytearray()
    for v in samples:
        s = int(max(-1.0, min(1.0, v)) * 32767)
        frames += s.to_bytes(2, "little", signed=True)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(bytes(frames))


def render(text: str, pitch=64, speed=72, throat=128, mouth=128, **kw) -> list[float]:
    return synth(reciter(text), pitch, speed, throat, mouth, **kw)


def say(text: str, path: str, pitch=64, speed=72, throat=128, mouth=128, **kw):
    samples = render(text, pitch, speed, throat, mouth, **kw)
    write_wav(path, samples)
    return samples


PRESETS = {
    # name: (speed, pitch, throat, mouth) — the original SAM manual's values
    "SAM":               (72, 64, 128, 128),
    "Little Robot":      (92, 60, 190, 190),
    "Elf":               (72, 64, 110, 160),
    "Stuffy Guy":        (82, 72, 110, 105),
    "Little Old Lady":   (82, 32, 145, 145),
    "Extra-Terrestrial": (100, 64, 150, 200),
}


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="SAM-family formant speech synth")
    ap.add_argument("text", nargs="+")
    ap.add_argument("-wav", required=True)
    ap.add_argument("-pitch", type=int, default=64)
    ap.add_argument("-speed", type=int, default=72)
    ap.add_argument("-throat", type=int, default=128)
    ap.add_argument("-mouth", type=int, default=128)
    a = ap.parse_args(argv)
    samples = say(" ".join(a.text), a.wav, a.pitch, a.speed, a.throat, a.mouth)
    print(f"{a.wav}: {len(samples) / SR:.2f}s, phonemes={len(reciter(' '.join(a.text)))}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
