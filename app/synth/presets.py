"""Voice parameter vectors for the from-scratch synthesiser.

A voice is a plain dict consumed by `app.synth.frames.build` (formants, open
quotient, tilt, breath, nasality) and `app.synth.prosody.plan` (pitch, speed,
rate), plus a `post` sub-dict consumed by `app.synth.post.finish`.

`pitch` and `speed` keep SAM's 0-255 scale so the 1982 presets and this engine
share one control surface: prosody maps pitch to ``20 + 1.15*pitch`` Hz and
speed to a rate scale of ``72/speed``.

Voice types (plan §5):

    narrator  adult male, neutral, the default human voice
    female    adult female, higher F0 and a 1.16 vocal tract
    child     short tract (1.20), high F0, faster
    giant     long tract (0.82), low F0, slower, darker tilt
    whisper   unvoiced aspiration only
    aged      low F0, breathy, slight nasality
    droid     synthetic: narrow tract, strong tilt, nasal colour
    sam       the 1982 preset values, kept for the retro montage
"""

from __future__ import annotations

# ── voice types ────────────────────────────────────────────────────────────
# vtl  : vocal-tract length scale (1.0 = adult male)
# oq   : glottal open quotient (0.3 pressed .. 0.8 breathy)
# tilt : source spectral tilt in dB at 3 kHz
# breath: aspiration mixed into every voiced frame (0..1)
# nasality: nasal pole/zero injected into non-nasal frames (0..1)
VOICE_TYPES = {
    "narrator": dict(
        pitch=70, speed=66, rate_scale=1.0, vtl=1.0, oq=0.55, tilt_db=7.0,
        breath=0.05, nasality=0.0, f4_gain=1.0, f5_gain=1.0,
        post=dict(lufs=-16.0, deess=0.25, ltas="neutral"),
    ),
    "female": dict(
        pitch=112, speed=70, rate_scale=1.0, vtl=1.16, oq=0.60, tilt_db=6.0,
        breath=0.06, nasality=0.0, f4_gain=1.0, f5_gain=1.0,
        post=dict(lufs=-16.0, deess=0.25, ltas="neutral"),
    ),
    "child": dict(
        pitch=132, speed=78, rate_scale=1.0, vtl=1.20, oq=0.65, tilt_db=6.0,
        breath=0.04, nasality=0.0, f4_gain=1.0, f5_gain=1.0,
        post=dict(lufs=-16.0, deess=0.20, ltas="neutral"),
    ),
    "giant": dict(
        pitch=34, speed=56, rate_scale=1.0, vtl=0.82, oq=0.45, tilt_db=11.0,
        breath=0.03, nasality=0.0, f4_gain=1.0, f5_gain=1.0,
        post=dict(lufs=-16.0, deess=0.15, ltas="dark"),
    ),
    "whisper": dict(
        pitch=70, speed=66, rate_scale=1.0, vtl=1.0, oq=0.55, tilt_db=10.0,
        breath=0.0, nasality=0.0, f4_gain=0.45, f5_gain=0.25, whisper=0.85,
        post=dict(lufs=-20.0, deess=0.35, ltas="dark"),
    ),
    "aged": dict(
        pitch=52, speed=60, rate_scale=1.0, vtl=0.97, oq=0.68, tilt_db=9.0,
        breath=0.18, nasality=0.10, f4_gain=0.9, f5_gain=0.8,
        post=dict(lufs=-16.0, deess=0.20, ltas="neutral"),
    ),
    "droid": dict(
        pitch=58, speed=62, rate_scale=1.0, vtl=0.90, oq=0.40, tilt_db=13.0,
        breath=0.02, nasality=0.35, f4_gain=0.7, f5_gain=0.5,
        post=dict(lufs=-16.0, deess=0.30, ltas="bright", celp=0.8),
    ),
    "sam": dict(
        pitch=64, speed=72, rate_scale=1.0, vtl=1.0, oq=0.50, tilt_db=6.0,
        breath=0.0, nasality=0.0, f4_gain=1.0, f5_gain=1.0,
        post=dict(lufs=-16.0, deess=0.25, ltas="neutral"),
    ),
}

# SAM's 1982 manual presets, translated to this parameter set. `throat` and
# `mouth` map onto the tract length and the high-formant trims.
SAM_PRESETS = {
    "SAM": dict(
        pitch=64, speed=72, vtl=1.0, oq=0.50, tilt_db=6.0, f4_gain=1.0, f5_gain=1.0,
    ),
    "Little Robot": dict(
        pitch=60, speed=92, vtl=0.96, oq=0.42, tilt_db=10.0, nasality=0.25,
        f4_gain=0.8, f5_gain=0.6,
    ),
    "Elf": dict(
        pitch=64, speed=72, vtl=1.14, oq=0.62, tilt_db=5.0, f4_gain=1.2, f5_gain=1.2,
    ),
    "Stuffy Guy": dict(
        pitch=72, speed=82, vtl=0.98, oq=0.50, tilt_db=8.0, nasality=0.55,
        f4_gain=0.7, f5_gain=0.5,
    ),
    "Little Old Lady": dict(
        pitch=32, speed=82, vtl=1.12, oq=0.70, tilt_db=8.0, breath=0.15,
        f4_gain=0.9, f5_gain=0.8,
    ),
    "Extra-Terrestrial": dict(
        pitch=64, speed=100, vtl=1.06, oq=0.35, tilt_db=12.0, nasality=0.3,
        f4_gain=0.6, f5_gain=0.4,
    ),
}

DEFAULT = "narrator"


def _merge(base: dict, extra: dict) -> dict:
    out = dict(base)
    out.update(extra)
    post = dict(base.get("post", {}))
    post.update(extra.get("post", {}))
    if post:
        out["post"] = post
    return out


def voice(name: str | None = None) -> dict:
    """Return a fresh, complete voice dict for `name` (default `narrator`)."""
    if not name:
        name = DEFAULT
    if name in VOICE_TYPES:
        return _merge(VOICE_TYPES[name], {})
    if name in SAM_PRESETS:
        return _merge(VOICE_TYPES["sam"], SAM_PRESETS[name])
    raise KeyError(f"unknown voice {name!r}; have {sorted(names())}")


def names() -> list[str]:
    """Every voice name, SAM presets first so the montage keeps its order."""
    return list(SAM_PRESETS) + [k for k in VOICE_TYPES if k != "sam"]


def override(base: dict, **kw) -> dict:
    """Copy `base` with overrides; `post` keys merge instead of replacing."""
    return _merge(base, kw)
