"""Faithful SAM voice: the real 1982 engine, ported — not a lookalike.

`sam.py` is a Klatt-style cascade synth tuned to sound SAM-ish. This module is
the actual thing: reciter → parser → renderer, with SAM's own tables, its three
direct oscillators (two sine, one rectangle) reset at every glottal pulse, its
4-bit output scaling and its 5-sample C64 simulation timetable.

Measured against the reference engine (discordier/sam, a JS port of Macke's C
port): byte-identical output on every case tested.

    from sam_faithful import faithful_say
    faithful_say("Hello. I am SAM.", "hello.wav", pitch=64, speed=72)

Licensing: SAM is abandonware — the tables come from a reverse-engineered ROM.
Fine for personal/experimental use; not something to ship commercially without
checking with SoftVoice, Inc.
"""

from __future__ import annotations

import os
import sys
import wave

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "sam_engine"))

import parse as _parse            # noqa: E402
import render as _render          # noqa: E402

SR = 22050


def faithful_pcm(text: str, pitch: int = 64, speed: int = 72,
                 mouth: int = 128, throat: int = 128) -> bytes:
    """Text → unsigned 8-bit PCM at 22.05 kHz, exactly as SAM produces it."""
    return _render.render(_parse.parse(text), pitch, speed, mouth, throat)


def faithful_samples(text: str, pitch: int = 64, speed: int = 72,
                     mouth: int = 128, throat: int = 128) -> list[int]:
    """Text → 16-bit signed samples; SAM's own scaling is (byte - 128) / 256."""
    return [(b - 128) * 128 for b in faithful_pcm(text, pitch, speed, mouth, throat)]


def faithful_say(text: str, path: str, pitch: int = 64, speed: int = 72,
                 mouth: int = 128, throat: int = 128) -> list[int]:
    samples = faithful_samples(text, pitch, speed, mouth, throat)
    frames = bytearray()
    for v in samples:
        frames += int(max(-32768, min(32767, v))).to_bytes(2, "little", signed=True)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(bytes(frames))
    return samples


def phonemes(text: str) -> str:
    return _parse.phonemes(text)


PRESETS = {
    # the manual's own values; SAM's defaults are (72, 64, 128, 128)
    "SAM":               (72, 64, 128, 128),
    "Little Robot":      (92, 60, 190, 190),
    "Elf":               (72, 64, 110, 160),
    "Stuffy Guy":        (82, 72, 110, 105),
    "Little Old Lady":   (82, 32, 145, 145),
    "Extra-Terrestrial": (100, 64, 150, 200),
}


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Faithful SAM voice")
    ap.add_argument("text", nargs="+")
    ap.add_argument("-wav", required=True)
    ap.add_argument("-pitch", type=int, default=64)
    ap.add_argument("-speed", type=int, default=72)
    ap.add_argument("-mouth", type=int, default=128)
    ap.add_argument("-throat", type=int, default=128)
    a = ap.parse_args()
    text = " ".join(a.text)
    s = faithful_say(text, a.wav, a.pitch, a.speed, a.mouth, a.throat)
    print(f"{a.wav}: {len(s) / SR:.2f}s, phonemes={phonemes(text)!r}")
