"""Phones -> per-frame synthesizer parameters (the contract between the
front end and the renderers).

Everything downstream (klatt.py, worldish.py) consumes a `Frames` object: a
set of arrays sampled at FRAME_HZ (5 ms). The arrays carry exactly the
parameters a Klatt-style source-filter synthesizer needs, plus the analytic
aperiodicity the WORLD-style renderer uses.

Two deliberate differences from the legacy sam-demo renderer:

1. Formant trajectories are built from *targets* and joined by finite linear
   ramps whose length depends on manner (stop 0 ms, fricative 20 ms, nasal
   50 ms, vowel 45 ms, sonorant 70 ms). The old renderer used one 15 ms
   one-pole for every boundary, which low-passes away the F2 slope that
   carries stop place.
2. Stops get real sub-segments: closure, an instantaneous burst with a
   place-specific spectrum, and an aspiration (VOT) interval with the
   formants parked at the place locus. That is what makes /p t k/ and
   /b d g/ separable.
"""

from __future__ import annotations

import math

from dataclasses import dataclass, fields

import numpy as np

FRAME_HZ = 200                 # 5 ms frames
MAX_FORMANT_HZ = 4800.0        # keep below 0.45*22050 and 0.45*24000

# ── parallel formant amplitudes ────────────────────────────────────────────
# A five-stage cascade multiplies every stage's out-of-band attenuation:
# measured on /AA/ it put F3 36 dB and F4 45 dB below F1 (natural speech is
# about 10 and 16 dB). Both renderers therefore run all five formants in
# parallel with explicit target peak amplitudes F1..F5.
FORMANT_AMP = (1.0, 0.55, 0.30, 0.16, 0.08)   # target peak amplitude F1..F5

# ── phoneme table ──────────────────────────────────────────────────────────
# f: F1..F5 (Hz), bw: bandwidths, av: voicing gain, kind, plus kind-specific
# fields. Values are the standard adult-male formant tables used throughout
# this repo, extended from three formants to five.
PH = {
    # vowels
    "IY": dict(f=[270, 2290, 3010, 3500, 4200], bw=[60, 90, 150, 200, 250], av=1.00, kind="vowel"),
    "IH": dict(f=[390, 1990, 2550, 3300, 4200], bw=[60, 90, 150, 200, 250], av=0.95, kind="vowel"),
    "EH": dict(f=[530, 1840, 2480, 3300, 4200], bw=[60, 90, 150, 200, 250], av=0.95, kind="vowel"),
    "AE": dict(f=[660, 1720, 2410, 3300, 4200], bw=[60, 90, 150, 200, 250], av=1.00, kind="vowel"),
    "AA": dict(f=[730, 1090, 2440, 3300, 4200], bw=[60, 90, 150, 200, 250], av=1.00, kind="vowel"),
    "AO": dict(f=[570, 840, 2410, 3300, 4200], bw=[60, 90, 150, 200, 250], av=1.00, kind="vowel"),
    "UH": dict(f=[440, 1020, 2240, 3300, 4200], bw=[60, 90, 150, 200, 250], av=0.90, kind="vowel"),
    "UW": dict(f=[300, 870, 2240, 3300, 4200], bw=[60, 90, 150, 200, 250], av=0.90, kind="vowel"),
    "AH": dict(f=[640, 1190, 2390, 3300, 4200], bw=[60, 90, 150, 200, 250], av=0.95, kind="vowel"),
    "AX": dict(f=[500, 1500, 2500, 3300, 4200], bw=[70, 100, 160, 200, 250], av=0.75, kind="vowel"),
    "ER": dict(f=[490, 1350, 1690, 3300, 4200], bw=[60, 100, 140, 200, 250], av=0.95, kind="vowel"),
    # nasals: nasal pole/zero pair engaged
    "M": dict(f=[480, 1270, 2130, 3300, 4200], bw=[40, 200, 200, 250, 300], av=0.60, kind="nasal"),
    "N": dict(f=[480, 1340, 2470, 3300, 4200], bw=[40, 300, 300, 250, 300], av=0.60, kind="nasal"),
    "NG": dict(f=[480, 2250, 2750, 3300, 4200], bw=[40, 200, 250, 250, 300], av=0.60, kind="nasal"),
    # liquids and glides
    "L": dict(f=[360, 1300, 2700, 3300, 4200], bw=[50, 100, 200, 250, 300], av=0.70, kind="sonorant"),
    "R": dict(f=[460, 1100, 1600, 3300, 4200], bw=[50, 120, 160, 250, 300], av=0.70, kind="sonorant"),
    "W": dict(f=[290, 610, 2150, 3300, 4200], bw=[50, 90, 180, 250, 300], av=0.60, kind="sonorant"),
    "Y": dict(f=[260, 2070, 3020, 3500, 4200], bw=[50, 90, 160, 250, 300], av=0.60, kind="sonorant"),
    "DX": dict(f=[420, 1650, 2550, 3300, 4200], bw=[60, 100, 150, 200, 250], av=0.85, kind="flap"),
    # fricatives: fric[k] are the six parallel-band gains, bypass is the
    # flat noise path. Numbers follow Klatt's Table III shape (s = strong
    # 4.9 kHz peak, sh = broad 1.8-4 kHz hump, f/th = weak bypass).
    "F": dict(f=[340, 1100, 2400, 3300, 4200], bw=[60, 90, 150, 200, 250], av=0.0,
              kind="fric", af=0.00392, fric=[0, 0, 0, 0, 0, 0], bypass=1.0),
    "TH": dict(f=[320, 1300, 2500, 3300, 4200], bw=[60, 90, 150, 200, 250], av=0.0,
               kind="fric", af=0.0042, fric=[0, 0, 0, 0, 0.25, 0], bypass=0.7),
    "S": dict(f=[320, 1400, 2600, 3400, 4200], bw=[60, 90, 150, 200, 250], av=0.0,
              kind="fric", af=0.01456, fric=[0, 0, 0, 0, 0, 1.0], bypass=0.0),
    "SH": dict(f=[300, 1600, 2600, 3400, 4200], bw=[60, 90, 150, 200, 250], av=0.0,
               kind="fric", af=0.014, fric=[0, 1.0, 0.85, 0.8, 0.6, 0], bypass=0.0),
    "V": dict(f=[220, 1100, 2400, 3300, 4200], bw=[60, 90, 150, 200, 250], av=0.35,
              kind="fric", af=0.00252, fric=[0, 0, 0, 0, 0, 0], bypass=1.0),
    "DH": dict(f=[270, 1300, 2500, 3300, 4200], bw=[60, 90, 150, 200, 250], av=0.35,
               kind="fric", af=0.00252, fric=[0, 0, 0, 0, 0.2, 0], bypass=0.7),
    "Z": dict(f=[240, 1400, 2600, 3400, 4200], bw=[60, 90, 150, 200, 250], av=0.30,
              kind="fric", af=0.0084, fric=[0, 0, 0, 0, 0, 0.8], bypass=0.0),
    "ZH": dict(f=[260, 1600, 2600, 3400, 4200], bw=[60, 90, 150, 200, 250], av=0.30,
               kind="fric", af=0.00728, fric=[0, 0.8, 0.7, 0.6, 0.5, 0], bypass=0.0),
    # HH: pure aspiration through the (neutral) tract
    "HH": dict(f=[500, 1500, 2500, 3300, 4200], bw=[70, 100, 160, 200, 250], av=0.0,
               kind="asp", ah=0.0875),
}

# Stops: closure fraction, burst spectrum (fric bands + bypass), VOT in ms,
# and the F2 locus the aspiration interval sits at.
STOPS = {
    "P": dict(voiced=False, burst=[0, 0, 0, 0, 0, 0], bypass=1.0, vot=45, locus2=720, locus1=300),
    "B": dict(voiced=True,  burst=[0, 0, 0, 0, 0, 0], bypass=1.0, vot=0,  locus2=720, locus1=300),
    "T": dict(voiced=False, burst=[0, 0.2, 0.6, 1.0, 0.7, 0.2], bypass=0.3, vot=60, locus2=1800, locus1=300),
    "D": dict(voiced=True,  burst=[0, 0.2, 0.5, 0.9, 0.6, 0.2], bypass=0.3, vot=0,  locus2=1800, locus1=300),
    "K": dict(voiced=False, burst=[0, 0.3, 1.0, 0.6, 0.3, 0], bypass=0.2, vot=70, locus2=3000, locus1=300),
    "G": dict(voiced=True,  burst=[0, 0.3, 0.9, 0.5, 0.3, 0], bypass=0.2, vot=0,  locus2=3000, locus1=300),
}

# Affricates: closure + burst + the fricative portion of a release.
AFFRICATES = {
    "CH": dict(voiced=False, burst=[0, 0.2, 0.6, 1.0, 0.7, 0.2], bypass=0.3,
               release="SH", vot=0),
    "JH": dict(voiced=True, burst=[0, 0.2, 0.5, 0.9, 0.6, 0.2], bypass=0.3,
               release="ZH", vot=0),
}

DIPHTHONGS = {
    "AY": ("AA", "IY"), "EY": ("EH", "IY"), "OW": ("AO", "UW"),
    "OY": ("AO", "IY"), "AW": ("AA", "UW"),
}

# Ramp length (ms) for the transition INTO a phone of this kind.
BURST_AF = 0.01     # stop/affricate burst level, ~-15 dB re a vowel
VOT_AH = 0.008       # stop aspiration level during the VOT

RAMP_MS = {
    "vowel": 45.0, "nasal": 50.0, "sonorant": 70.0, "flap": 18.0, "fric": 20.0,
    "asp": 20.0, "stop": 0.0, "affricate": 0.0, "gap": 25.0,
}


@dataclass
class Frames:
    f0: np.ndarray
    av: np.ndarray
    ah: np.ndarray
    af: np.ndarray
    formants: np.ndarray
    bws: np.ndarray
    fnp: np.ndarray
    fnz: np.ndarray
    fric: np.ndarray
    bypass: np.ndarray
    oq: np.ndarray
    tilt_db: np.ndarray
    f4_gain: float = 1.0
    f5_gain: float = 1.0

    @property
    def n(self) -> int:
        return int(self.f0.shape[0])

    @property
    def duration_s(self) -> float:
        return self.n / FRAME_HZ


def _empty(n: int) -> Frames:
    return Frames(
        f0=np.zeros(n), av=np.zeros(n), ah=np.zeros(n), af=np.zeros(n),
        formants=np.zeros((n, 5)), bws=np.zeros((n, 5)),
        fnp=np.zeros(n), fnz=np.zeros(n), fric=np.zeros((n, 6)),
        bypass=np.zeros(n), oq=np.full(n, 0.5), tilt_db=np.full(n, 6.0),
    )


def _clamp_formants(formants: np.ndarray) -> np.ndarray:
    """Keep F1<F2<...<F5 and below MAX_FORMANT_HZ on active frames."""
    f = formants.copy()
    active = np.any(f > 0.0, axis=1)
    if not np.any(active):
        return f
    f[active, 0] = np.clip(f[active, 0], 150.0, 1200.0)
    for i in range(1, 5):
        cap = MAX_FORMANT_HZ - (4 - i) * 120.0
        f[active, i] = np.minimum(np.maximum(f[active, i], f[active, i - 1] + 120.0), cap)
    return f


def _fill(fr: Frames, start: int, stop: int, f, bw, av, ah=0, af=0,
          fric=None, bypass=0.0, fnp=0.0, fnz=0.0) -> None:
    fr.formants[start:stop] = f
    fr.bws[start:stop] = bw
    fr.av[start:stop] = av
    fr.ah[start:stop] = ah
    fr.af[start:stop] = af
    if fric is not None:
        fr.fric[start:stop] = fric
    fr.bypass[start:stop] = bypass
    fr.fnp[start:stop] = fnp
    fr.fnz[start:stop] = fnz


def _ramp(fr: Frames, b: int, length_ms: float) -> None:
    """Join the targets on either side of frame b with a linear ramp.

    `b` is the first frame of the new phone. The ramp is centred on the
    boundary so the transition neither lags nor leads the segment boundary.
    """
    if length_ms <= 0.0 or b <= 0 or b >= fr.n:
        return
    L = max(1, int(round(length_ms * FRAME_HZ / 1000.0)))
    start = max(0, b - L // 2)
    end = min(fr.n, start + L)
    if end - start < 2:
        return
    span = end - start
    prev_f = fr.formants[b - 1].copy()
    prev_bw = fr.bws[b - 1].copy()
    prev_av = float(fr.av[b - 1])
    prev_ah = float(fr.ah[b - 1])
    prev_af = float(fr.af[b - 1])
    prev_fric = fr.fric[b - 1].copy()
    prev_bypass = float(fr.bypass[b - 1])
    prev_fnp = float(fr.fnp[b - 1])
    prev_fnz = float(fr.fnz[b - 1])

    next_f = fr.formants[b].copy()
    next_bw = fr.bws[b].copy()
    next_av = float(fr.av[b])
    next_ah = float(fr.ah[b])
    next_af = float(fr.af[b])
    next_fric = fr.fric[b].copy()
    next_bypass = float(fr.bypass[b])
    next_fnp = float(fr.fnp[b])
    next_fnz = float(fr.fnz[b])

    # Silence/gap frames carry formants=0; never ramp vocal-tract resonances
    # down to 0 Hz across a silence boundary (hold the adjacent phone's tract).
    if prev_f[0] <= 0.0 and next_f[0] > 0.0:
        prev_f, prev_bw = next_f, next_bw
    elif next_f[0] <= 0.0 and prev_f[0] > 0.0:
        next_f, next_bw = prev_f, prev_bw

    # Nasal transition: non-nasal is pole==zero at 270 Hz (complete
    # cancellation), so ramp fnz between 270 Hz and 450 Hz while holding
    # fnp at 270 Hz instead of sweeping fnp from 0 Hz through F0.
    both_non_nasal = (prev_fnp <= 0.0) and (next_fnp <= 0.0)
    if not both_non_nasal:
        if prev_fnp <= 0.0:
            prev_fnp = next_fnp
            prev_fnz = next_fnp
        elif next_fnp <= 0.0:
            next_fnp = prev_fnp
            next_fnz = prev_fnp

    for i in range(start, end):
        a = (i - start + 1) / (span + 1)
        fr.formants[i] = prev_f * (1.0 - a) + next_f * a
        fr.bws[i] = prev_bw * (1.0 - a) + next_bw * a
        fr.av[i] = prev_av * (1.0 - a) + next_av * a
        fr.ah[i] = prev_ah * (1.0 - a) + next_ah * a
        fr.af[i] = prev_af * (1.0 - a) + next_af * a
        fr.fric[i] = prev_fric * (1.0 - a) + next_fric * a
        fr.bypass[i] = prev_bypass * (1.0 - a) + next_bypass * a
        if not both_non_nasal:
            fr.fnp[i] = prev_fnp * (1.0 - a) + next_fnp * a
            fr.fnz[i] = prev_fnz * (1.0 - a) + next_fnz * a


def _phone_frames(name: str, nf: int, fr: Frames, start: int) -> None:
    """Fill nf frames for one phone, including stop sub-segments."""
    if name == "_" or name == "__":
        # handled by the caller (silence, boundary pause)
        return
    if name in DIPHTHONGS:
        a, b = DIPHTHONGS[name]
        half = max(1, nf // 2)
        s0 = start
        for part, count in ((a, half), (b, nf - half)):
            info = PH[part]
            _fill(fr, start, start + count, np.array(info["f"], float),
                  np.array(info["bw"], float), info["av"])
            start += count
        if nf - half > 0:
            _ramp(fr, s0 + half, RAMP_MS["vowel"])
        return
    if name in STOPS:
        st = STOPS[name]
        closure = max(1, int(round(nf * 0.55)))
        burst = min(max(1, int(round(0.008 * FRAME_HZ))), max(0, nf - closure))
        vot = int(round(st["vot"] / 1000.0 * FRAME_HZ))
        vot = min(vot, max(0, nf - closure - burst))
        rest = max(0, nf - closure - burst - vot)
        locus_f = np.array([st["locus1"], st["locus2"], 2500.0, 3300.0, 4200.0])
        locus_bw = np.array([60.0, 120.0, 150.0, 200.0, 250.0])
        # closure: silence, formants parked at the locus
        _fill(fr, start, start + closure, locus_f, locus_bw, 0.0)
        start += closure
        # burst: instantaneous frication, never ramped
        if burst:
            _fill(fr, start, start + burst, locus_f, locus_bw, 0.0, af=BURST_AF,
                  fric=np.array(st["burst"], float), bypass=st["bypass"])
            start += burst
        # aspiration (VOT): noise through the tract at the locus
        if vot:
            _fill(fr, start, start + vot, locus_f, locus_bw, 0.0, ah=VOT_AH)
            start += vot
        # any remaining frames: voiced, still at the locus (the ramp into the
        # following vowel is applied by the boundary pass)
        if rest:
            _fill(fr, start, start + rest, locus_f, locus_bw,
                  0.45 if st["voiced"] else 0.0)
        return
    if name in AFFRICATES:
        af = AFFRICATES[name]
        closure = max(1, int(round(nf * 0.45)))
        burst = min(max(1, int(round(0.008 * FRAME_HZ))), max(0, nf - closure))
        rest = max(0, nf - closure - burst)
        locus_f = np.array([300.0, 1800.0, 2600.0, 3400.0, 4200.0])
        locus_bw = np.array([60.0, 120.0, 150.0, 200.0, 250.0])
        _fill(fr, start, start + closure, locus_f, locus_bw, 0.0)
        start += closure
        if burst:
            _fill(fr, start, start + burst, locus_f, locus_bw, 0.0, af=BURST_AF,
                  fric=np.array(af["burst"], float), bypass=af["bypass"])
            start += burst
        rel = PH[af["release"]]
        _fill(fr, start, start + rest, np.array(rel["f"], float),
              np.array(rel["bw"], float), rel["av"], af=rel["af"],
              fric=np.array(rel["fric"], float), bypass=rel["bypass"])
        return

    info = PH.get(name)
    if info is None:
        # unknown phone: treat as a schwa so nothing crashes
        info = PH["AX"]
    f = np.array(info["f"], float)
    bw = np.array(info["bw"], float)
    kind = info["kind"]
    if kind == "fric":
        _fill(fr, start, start + nf, f, bw, info["av"], af=info["af"],
              fric=np.array(info["fric"], float), bypass=info["bypass"])
    elif kind == "asp":
        _fill(fr, start, start + nf, f, bw, 0.0, ah=info["ah"])
    elif kind == "nasal":
        _fill(fr, start, start + nf, f, bw, info["av"], fnp=270.0, fnz=450.0)
    else:
        _fill(fr, start, start + nf, f, bw, info["av"])


def _kind_of(name: str) -> str:
    if name in ("_", "__"):
        return "gap"
    if name in STOPS:
        return "stop"
    if name in AFFRICATES:
        return "affricate"
    if name in DIPHTHONGS:
        return "vowel"
    return PH.get(name, PH["AX"])["kind"]


def build(phones, voice=None, prosody=None) -> Frames:
    """Phones + voice + prosody -> Frames.

    `prosody` defaults to app.synth.prosody.plan. Pass a stub module/object
    with the same `plan(phones, voice)` signature for tests.
    """
    voice = dict(voice or {})
    if prosody is None:
        from . import prosody as prosody_mod
        prosody = prosody_mod

    durations, f0 = prosody.plan(phones, voice)
    n = int(round(sum(durations) * FRAME_HZ))
    fr = _empty(max(1, n))

    idx = 0
    boundaries: list[tuple[int, float]] = []
    for phone, dur in zip(phones, durations):
        nf = int(round(dur * FRAME_HZ))
        nf = min(nf, fr.n - idx)
        if nf <= 0:
            continue
        kind = _kind_of(phone.name)
        if kind == "gap":
            pass                                  # already silent
        else:
            _phone_frames(phone.name, nf, fr, idx)
        if idx > 0:
            boundaries.append((idx, RAMP_MS.get(kind, 30.0)))
        idx += nf

    # Hold adjacent vocal-tract resonances across silent gap frames so formants
    # and bandwidths are never 0 Hz anywhere in the trajectory.
    active_f = np.flatnonzero(fr.formants[:, 0] > 0.0)
    if active_f.size > 0:
        for i in range(fr.n):
            if fr.formants[i, 0] <= 0.0:
                pos = int(np.searchsorted(active_f, i))
                ref = int(active_f[min(pos, active_f.size - 1)])
                fr.formants[i] = fr.formants[ref]
                fr.bws[i] = fr.bws[ref]
    else:
        ax = PH["AX"]
        fr.formants[:] = np.array(ax["f"], float)
        fr.bws[:] = np.array(ax["bw"], float)

    for b, length in boundaries:
        _ramp(fr, b, length)

    # ── voice transform ────────────────────────────────────────────────────
    vtl = float(voice.get("vtl", 1.0))
    if vtl != 1.0:
        fr.formants *= vtl
    fr.formants = _clamp_formants(fr.formants)
    # sam-demo's Hz-scaled bandwidth floor: a VTL-scaled-up voice (Little
    # Robot, Extra-Terrestrial, child, female) keeps table bandwidths while
    # the formants move up, so the resonators ring with overly narrow Q.
    # Floor the bandwidths proportionally to the scaled formant Hz.
    fr.bws = np.maximum(fr.bws, 0.05 * fr.formants + 45.0)
    fr.oq[:] = float(voice.get("oq", 0.5))
    fr.tilt_db[:] = float(voice.get("tilt_db", 6.0))
    fr.f4_gain = float(voice.get("f4_gain", 1.0))
    fr.f5_gain = float(voice.get("f5_gain", 1.0))
    breath = float(voice.get("breath", 0.0))
    if breath > 0.0:
        fr.ah = np.maximum(fr.ah, breath * fr.av)
    whisper = float(voice.get("whisper", 0.0))
    if whisper > 0.0:
        # whisper: replace glottal voicing with aspiration on voiced segments
        # while keeping silent word gaps, pauses, and stop closures silent.
        fr.ah = np.maximum(fr.ah, whisper * fr.av)
        fr.av[:] = 0.0
    nasality = float(voice.get("nasality", 0.0))
    if nasality > 0.0:
        # Klatt pole-zero pair: pole fixed at 270 Hz; zero separates from the
        # pole (270 Hz = complete cancellation) toward 450 Hz as nasality -> 1.
        nasal = (fr.fnp > 0.0) | ((fr.av <= 1e-4) & (fr.ah <= 1e-4))
        fr.fnp = np.where(nasal, fr.fnp, 270.0)
        fr.fnz = np.where(nasal, fr.fnz, 270.0 + 180.0 * min(1.0, nasality))

    # ── pitch ──────────────────────────────────────────────────────────────
    f0 = np.asarray(f0, float)[:fr.n]
    if f0.shape[0] < fr.n:
        f0 = np.pad(f0, (0, fr.n - f0.shape[0]), mode="edge")
    pos_f0 = np.flatnonzero(f0 > 0.0)
    if pos_f0.size > 0:
        f0 = np.interp(np.arange(fr.n, dtype=float), pos_f0.astype(float), f0[pos_f0])
    f0 = np.where(fr.av > 1e-4, f0, 0.0)
    fr.f0 = f0
    return fr


def _smooth_cols(mat: np.ndarray, alpha: float) -> np.ndarray:
    if mat.shape[0] <= 1 or alpha >= 1.0:
        return mat
    out = np.empty_like(mat)
    cur = mat[0].copy()
    out[0] = cur
    b = 1.0 - alpha
    for i in range(1, mat.shape[0]):
        cur = b * cur + alpha * mat[i]
        out[i] = cur
    return out


def interpolate(fr: Frames, n_samples: int) -> Frames:
    """Linear per-sample interpolation of every frame parameter."""
    if n_samples <= 0:
        n_samples = 1
    src = np.linspace(0.0, fr.n - 1.0, fr.n)
    dst = np.linspace(0.0, fr.n - 1.0, n_samples)
    av_s = np.interp(dst, src, fr.av)
    # Interpolate F0 across voiced frames only so F0 never glides down through
    # 0..50 Hz at voiced/unvoiced transitions.
    voiced_idx = np.flatnonzero(fr.f0 > 0.0)
    if voiced_idx.size > 0:
        f0_filled = np.interp(src, src[voiced_idx], fr.f0[voiced_idx])
        f0_s = np.where(av_s > 1e-4, np.interp(dst, src, f0_filled), 0.0)
    else:
        f0_s = np.zeros(n_samples)
    # Non-nasal frames have fnp==0; treat non-nasal as pole==zero at 270 Hz so
    # sample interpolation never sweeps fnp through 0..200 Hz.
    if np.any(fr.fnp > 0.0):
        fnp_src = np.where(fr.fnp > 0.0, fr.fnp, 270.0)
        fnz_src = np.where(fr.fnp > 0.0, fr.fnz, fnp_src)
        fnp_i = np.interp(dst, src, fnp_src)
        fnz_i = np.interp(dst, src, fnz_src)
        active = np.abs(fnz_i - fnp_i) > 0.5
        fnp_s = np.where(active, fnp_i, 0.0)
        fnz_s = np.where(active, fnz_i, 0.0)
    else:
        fnp_s = np.zeros(n_samples)
        fnz_s = np.zeros(n_samples)
    out = Frames(
        f0=f0_s,
        av=av_s,
        ah=np.interp(dst, src, fr.ah),
        af=np.interp(dst, src, fr.af),
        formants=np.stack([np.interp(dst, src, fr.formants[:, k])
                           for k in range(5)], axis=1),
        bws=np.stack([np.interp(dst, src, fr.bws[:, k])
                      for k in range(5)], axis=1),
        fnp=fnp_s,
        fnz=fnz_s,
        fric=np.stack([np.interp(dst, src, fr.fric[:, k])
                       for k in range(6)], axis=1),
        bypass=np.interp(dst, src, fr.bypass),
        oq=np.interp(dst, src, fr.oq),
        tilt_db=np.interp(dst, src, fr.tilt_db),
        f4_gain=fr.f4_gain,
        f5_gain=fr.f5_gain,
    )
    # sam-demo's sample-level one-pole exponential Hz smoothing (tau 8 ms):
    # the piecewise-linear frame ramps keep their slope over 5 ms corners,
    # and the corner kinks that excite clicks at coefficient updates are
    # rounded in Hz. Steady vowels are fixed points of the filter, so they
    # stay identical.
    sr_est = max(1.0, n_samples * FRAME_HZ / max(fr.n, 1))
    alpha_f = 1.0 - math.exp(-1.0 / max(1.0, 0.008 * sr_est))
    out.formants = _clamp_formants(_smooth_cols(out.formants, alpha_f))
    out.bws = _smooth_cols(out.bws, alpha_f)
    return out


def field_names() -> list[str]:
    return [f.name for f in fields(Frames) if f.name not in ("f4_gain", "f5_gain")]


def peak_amplitudes(voice=None) -> np.ndarray:
    """Target peak amplitude for F1..F5, after the voice's F4/F5 trims."""
    a = np.array(FORMANT_AMP, dtype=float)
    if isinstance(voice, Frames):
        a[3] *= float(voice.f4_gain)
        a[4] *= float(voice.f5_gain)
    elif voice:
        a[3] *= float(voice.get("f4_gain", 1.0))
        a[4] *= float(voice.get("f5_gain", 1.0))
    return a


def resonator_mag(hz, formant, bw, sr):
    """Exact magnitude of the unity-DC two-pole resonator, vectorised.

    The usual closed-form ``a / sqrt((1-(f/F)^2)^2 + (f*BW/F^2)^2)`` is only an
    approximation: at F1 it is wrong by a factor of ~170 (it predicts 0.027
    where the digital filter really peaks at 4.47), which would make every
    formant balance computed from it meaningless. This evaluates the transfer
    function on the unit circle exactly.
    """
    hz = np.asarray(hz, dtype=float)
    formant = np.asarray(formant, dtype=float)
    bw = np.maximum(np.asarray(bw, dtype=float), 1.0)
    theta = 2.0 * np.pi * formant / sr
    r = np.exp(-np.pi * bw / sr)
    a = 1.0 - 2.0 * r * np.cos(theta) + r * r
    b = 2.0 * r * np.cos(theta)
    c = -r * r
    w = 2.0 * np.pi * hz / sr
    re = 1.0 - b * np.cos(w) - c * np.cos(2.0 * w)
    im = b * np.sin(w) + c * np.sin(2.0 * w)
    return a / np.sqrt(np.maximum(re * re + im * im, 1e-24))
