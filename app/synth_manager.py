"""Service-side wrapper for the from-scratch formant synthesiser.

Pipeline: text -> reciter (phones) -> prosody (durations + F0) -> frames
(formant/amplitude tracks) -> klatt renderer -> post conditioning -> WAV.

Everything is CPU-only numpy, no model files, so it starts instantly and has
no download step. Piper stays the default human voice; this engine is selected
per request with ``engine="synth"``.
"""

from __future__ import annotations

import hashlib
import io
import struct
import threading

import numpy as np

from app.synth import frames, klatt, post, presets
from app.synth.reciter import reciter

SAMPLE_RATE = 22050

_lock = threading.Lock()
_cache: dict[str, bytes] = {}
_CACHE_MAX = 64


def _wav_bytes(samples: np.ndarray, sr: int = SAMPLE_RATE) -> bytes:
    pcm = np.clip(np.asarray(samples, dtype=np.float64), -1.0, 1.0)
    pcm = (pcm * 32767.0).astype("<i2")
    buf = io.BytesIO()
    data = pcm.tobytes()
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", 36 + len(data)))
    buf.write(b"WAVEfmt ")
    buf.write(struct.pack("<IHHIIHH", 16, 1, 1, sr, sr * 2, 2, 16))
    buf.write(b"data")
    buf.write(struct.pack("<I", len(data)))
    buf.write(data)
    return buf.getvalue()


def render(text: str, voice_name: str = "narrator") -> np.ndarray:
    """Render `text` in `voice_name` and return float samples at 22050 Hz."""
    voice = presets.voice(voice_name)
    phones = reciter(text)
    if not phones:
        return np.zeros(int(0.2 * SAMPLE_RATE))
    fr = frames.build(phones, voice)
    audio = klatt.render(fr, sr=SAMPLE_RATE, rng=np.random.default_rng(0))
    return post.finish(np.asarray(audio, dtype=np.float64), SAMPLE_RATE, voice)


def synthesize(text: str, voice_name: str = "narrator") -> bytes:
    """Render `text` and return WAV bytes (cached per text+voice)."""
    key = hashlib.sha1(f"{voice_name}\x00{text}".encode("utf-8")).hexdigest()
    with _lock:
        hit = _cache.get(key)
    if hit is not None:
        return hit
    audio = render(text, voice_name)
    data = _wav_bytes(audio)
    with _lock:
        if len(_cache) >= _CACHE_MAX:
            _cache.pop(next(iter(_cache)))
        _cache[key] = data
    return data


def list_voices() -> list[str]:
    return presets.names()


def voice_info(name: str) -> dict:
    return presets.voice(name)
