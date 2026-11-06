"""Voices made out of computer sounds: five techniques, all DSP, no samples.

Each function takes the output of `sam.py` (or its own oscillator bank) and
turns it into something a machine would say. Nothing here is neural, nothing
is recorded.

    chip_square()  square-wave glottal source through the same formant cascade
    vocoder()      16-band envelope transfer from speech onto a square carrier
    ring_mod()     Dalek-style multiplication by a low sine
    bitcrush()     8-bit / 11 kHz quantisation, the lo-fi "computer" grit
    blips()        animalese-style per-letter square blips (prosody only)

Run `python3 computer_voices.py` for a demo of each.
"""

from __future__ import annotations

import math
import random

import sam
from sam import SR

# ── chip voice: square source instead of the glottal saw ───────────────────


def chip_square(text: str, pitch=60, speed=92, throat=190, mouth=190) -> list[float]:
    """The C64 sound: a square wave (odd harmonics, 1/n) through the formant
    cascade. A square's spectrum is denser and brighter than a saw's, which is
    what makes chip voices read as "computer" — and why they need band-limiting.
    """
    return sam.render(text, pitch=pitch, speed=speed, throat=throat,
                      mouth=mouth, source="square")


# ── vocoder: speech envelopes onto a computer tone ─────────────────────────

BANDS = 16
BAND_LO, BAND_HI = 120.0, 5000.0


def _band_edges():
    ratio = (BAND_HI / BAND_LO) ** (1.0 / (BANDS - 1))
    return [BAND_LO * ratio ** i for i in range(BANDS)]


class _BP:
    """RBJ band-pass (constant skirt gain, peak gain = Q)."""

    __slots__ = ("b0", "b1", "b2", "a1", "a2", "x1", "x2", "y1", "y2")

    def __init__(self, f: float, q: float = 2.0):
        w = 2.0 * math.pi * f / SR
        c, s = math.cos(w), math.sin(w)
        alpha = s / (2.0 * q)
        a0 = 1.0 + alpha
        self.b0 = alpha / a0
        self.b1 = 0.0
        self.b2 = -alpha / a0
        self.a1 = -2.0 * c / a0
        self.a2 = (1.0 - alpha) / a0
        self.x1 = self.x2 = self.y1 = self.y2 = 0.0

    def run(self, x: float) -> float:
        y = (self.b0 * x + self.b2 * self.x2
             - self.a1 * self.y1 - self.a2 * self.y2)
        self.x2, self.x1 = self.x1, x
        self.y2, self.y1 = self.y1, y
        return y


def _carrier(n: int, kind: str, f0: float, rng: random.Random) -> list[float]:
    out = []
    phase = 0.0
    for _ in range(n):
        phase += f0 / SR
        if phase >= 1.0:
            phase -= 1.0
        if kind == "square":
            out.append(1.0 if phase < 0.5 else -1.0)
        elif kind == "saw":
            out.append(2.0 * phase - 1.0)
        else:
            out.append(rng.uniform(-1.0, 1.0))
    return out


def vocoder(x: list[float], bands: int = BANDS, carrier: str = "square",
            carrier_f0: float = 110.0, env_hz: float = 40.0,
            seed: int = 5) -> list[float]:
    """Classic channel vocoder.

    For each band k:  e_k = LP_{env}(|BP_k(x)|),  y += e_k · BP_k(carrier)

    Band count drives intelligibility (≈5 bands is already usable, 16 is the
    classic robot sound); carrier type only changes the timbre.
    """
    edges = _band_edges()
    rng = random.Random(seed)
    car = _carrier(len(x), carrier, carrier_f0, rng)
    env_a = 1.0 - math.exp(-2.0 * math.pi * env_hz / SR)

    out = [0.0] * len(x)
    for f in edges:
        ana, carf = _BP(f), _BP(f)
        env = 0.0
        for i, v in enumerate(x):
            env += (abs(ana.run(v)) - env) * env_a
            out[i] += env * carf.run(car[i])
    peak = max(1e-9, max(abs(v) for v in out))
    return [v / peak * 0.85 for v in out]


# ── ring modulation ────────────────────────────────────────────────────────


def ring_mod(x: list[float], fm: float = 45.0, wet: float = 0.35) -> list[float]:
    """y = x·((1-wet) + wet·sin(2π fm t)).

    Every harmonic k·f0 becomes a pair at k·f0 ± fm. Keep fm well under the
    formant bandwidths (~60-200 Hz) and vowel identity survives; the sideband
    flutter is the machine cue. Above ~150 Hz the formants smear and speech
    collapses.
    """
    out = []
    for i, v in enumerate(x):
        m = 1.0 - wet + wet * math.sin(2.0 * math.pi * fm * i / SR)
        out.append(v * m)
    peak = max(1e-9, max(abs(v) for v in out))
    return [v / peak * 0.85 for v in out]


# ── bit-crush ──────────────────────────────────────────────────────────────


def bitcrush(x: list[float], bits: int = 8, hold: int = 2,
             dither: bool = True) -> list[float]:
    """Quantise to `bits` and sample-and-hold every `hold` samples.

    Uniform quantiser step Δ = 2V/2^bits, error power Δ²/12, so a full-scale
    sine gives SNR = 6.02·bits + 1.76 dB. TPDF dither (±1 LSB triangular) turns
    signal-correlated distortion into white noise, which is what keeps quiet
    passages from turning into gating fuzz.
    """
    levels = 2 ** (bits - 1)
    rng = random.Random(11)
    out = []
    held = 0.0
    for i, v in enumerate(x):
        if i % hold == 0:
            d = (rng.random() - rng.random()) / levels if dither else 0.0
            held = round((v + d) * levels) / levels
        out.append(held)
    peak = max(1e-9, max(abs(v) for v in out))
    return [v / peak * 0.85 for v in out]


# ── animalese-style blips ──────────────────────────────────────────────────

_VOWELS = set("aeiouy")


def blips(text: str, rate: float = 20.0, base_f0: float = 240.0,
          seed: int = 3) -> list[float]:
    """One short square blip per character, pitch classed by letter.

    This carries rhythm and pitch contour only — no formants, so it reads as
    "talking" without being recoverable as words. That is exactly the
    animalese trade: ~30 syllables/s, pitch per personality.
    """
    rng = random.Random(seed)
    gap = 1.0 / rate
    out: list[float] = []
    for ch in text.lower():
        if not ch.isalpha():
            out.extend([0.0] * int(gap * 0.6 * SR))
            continue
        f0 = base_f0 * (0.85 if ch in _VOWELS else 1.15)
        f0 *= 1.0 + rng.uniform(-0.06, 0.06)
        dur = gap * (0.9 if ch in _VOWELS else 0.55)
        n = int(dur * SR)
        phase = 0.0
        blip = []
        for i in range(n):
            phase += f0 / SR
            if phase >= 1.0:
                phase -= 1.0
            env = min(1.0, i / (0.002 * SR)) * math.exp(-i / (0.030 * SR))
            blip.append((1.0 if phase < 0.5 else -1.0) * env)
        out.extend(blip)
        out.extend([0.0] * int(gap * 0.1 * SR))
    peak = max(1e-9, max(abs(v) for v in out))
    return [v / peak * 0.85 for v in out]


if __name__ == "__main__":
    import sys

    text = " ".join(sys.argv[1:]) or "Hello. I am a computer voice."
    base = sam.render(text, pitch=60, speed=92, throat=190, mouth=190)
    sam.write_wav("/tmp/cv-chip.wav", chip_square(text))
    sam.write_wav("/tmp/cv-vocoder.wav", vocoder(base))
    sam.write_wav("/tmp/cv-ring.wav", ring_mod(base, 45.0, 0.35))
    sam.write_wav("/tmp/cv-crush.wav", bitcrush(base, 8, 2))
    sam.write_wav("/tmp/cv-blips.wav", blips(text))
    print("wrote /tmp/cv-*.wav")
