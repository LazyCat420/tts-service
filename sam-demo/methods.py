"""Four ways to turn formant targets into a waveform, for A/B listening.

Reuses sam.py's phoneme tables and segment builder, swaps the renderer:

  cascade  - saw -> F1 -> F2 -> F3 (2-pole resonators), the current synth
  parallel - three resonators side by side, summed with per-formant gains (Klatt)
  additive - sum of harmonics whose amplitudes come from the formant response
  pulse    - cascade, but driven by a Rosenberg glottal pulse instead of a saw

Bandwidth is either the fixed 70/110/180 Hz or scaled with the formant:
bw(f) = 0.06*f + 50 (Praat uses 0.05*F above F4; Klatt's floor is 40 Hz).
"""
from __future__ import annotations

import math
import random
import sys

import sam

SR = sam.SR
TWOPI = 2.0 * math.pi


# ── resonators ────────────────────────────────────────────────────────────────

def _coef(f, bw, sr=SR):
    r = math.exp(-math.pi * bw / sr)
    b1 = 2.0 * r * math.cos(TWOPI * f / sr)
    b2 = -r * r
    return 1.0 - b1 - b2, b1, b2


class Res:
    __slots__ = ("a0", "b1", "b2", "y1", "y2")

    def __init__(self):
        self.a0 = self.b1 = self.b2 = 0.0
        self.y1 = self.y2 = 0.0

    def set(self, f, bw):
        self.a0, self.b1, self.b2 = _coef(f, bw)

    def run(self, x):
        y = self.a0 * x + self.b1 * self.y1 + self.b2 * self.y2
        self.y2, self.y1 = self.y1, y
        return y


def _blep(t, dt):
    if t < dt:
        t /= dt
        return t + t - t * t - 1.0
    if t > 1.0 - dt:
        t = (t - 1.0) / dt
        return t * t + t + t + 1.0
    return 0.0


def _alpha(tau_ms, sr=SR):
    return 1.0 - math.exp(-1.0 / (tau_ms * 1e-3 * sr))


# ── segment builder (same as sam.synth) ──────────────────────────────────────

def segments(phonemes, speed=72):
    dur_scale = 72.0 / max(speed, 1)
    segs = []
    for p in phonemes:
        if p == sam.WORD_GAP:
            segs.append((500.0, 1500.0, 2500.0, 0.0, 0.0, 0.0, 0.070))
        elif p == sam.SENTENCE_GAP:
            segs.append((500.0, 1500.0, 2500.0, 0.0, 0.0, 0.0, 0.250))
        elif p in sam.DIPHTHONGS:
            a, b, d = sam.DIPHTHONGS[p]
            segs.append((sam.V[a][0], sam.V[a][1], sam.V[a][2], sam.V[a][4], 0.0, 0.0, d / 2))
            segs.append((sam.V[b][0], sam.V[b][1], sam.V[b][2], sam.V[b][4], 0.0, 0.0, d / 2))
        else:
            f1, f2, f3, dur, va, na, nh = sam.V[p]
            if p in sam.STOPS:
                segs.append((f1, f2, f3, 0.0, 0.0, 0.0, dur * 0.6))
                segs.append((f1, f2, f3, va, na, nh, dur * 0.4))
            else:
                segs.append((f1, f2, f3, va, na, nh, dur))
    return segs, dur_scale


# ── harmonic response of a formant set ──────────────────────────────────────

def formant_gain(f, formants, gains, bws):
    """|H(f)| for a parallel bank of 2-pole resonators (analog magnitude)."""
    total = 0.0
    for (fk, gk, bw) in zip(formants, gains, bws):
        num = fk * fk
        den = math.hypot(fk * fk - f * f, f * bw)
        total += gk * num / den
    return total


# ── renderers ────────────────────────────────────────────────────────────────

def render(phonemes, pitch=64, speed=72, throat=128, mouth=128,
           method="cascade", bw_mode="scaled", source="saw",
           lp_hz=4800.0, hp_hz=80.0, noise_gain=0.35, seed=7,
           tilt_hz=500.0, max_partial_hz=5000.0):
    f0_base = 20.0 + pitch * 1.15
    # SAM's own semantics, measured on the ported engine: mouth scales F1,
    # throat scales F2, F3 is untouched (factor 128 = identity, linear).
    mouth_scale = mouth / 128.0
    throat_scale = throat / 128.0
    segs, dur_scale = segments(phonemes, speed)

    rng = random.Random(seed)
    res = [Res(), Res(), Res()]
    nres = Res()
    lp = sam.Biquad()
    hp = sam.Biquad()
    if lp_hz:
        lp.lowpass(lp_hz)
    if hp_hz:
        hp.highpass(hp_hz)

    f1, f2, f3 = segs[0][0], segs[0][1], segs[0][2]
    va, na, nh = segs[0][3], segs[0][4], float(segs[0][5] or 2000.0)

    total = sum(s[6] for s in segs) * dur_scale
    out = []
    phase = 0.0
    fs = _alpha(15.0)
    asm = _alpha(5.0)
    elapsed = 0.0
    gains = (1.0, 0.55, 0.30)

    for seg in segs:
        f1t, f2t, f3t, vat, nat, nht, dur = seg
        n = max(1, int(dur * dur_scale * SR))
        for _ in range(n):
            f1 += (f1t - f1) * fs
            f2 += (f2t - f2) * fs
            f3 += (f3t - f3) * fs
            va += (vat - va) * asm
            na += (nat - na) * asm
            if nht:
                nh += (float(nht) - nh) * fs

            prog = elapsed / max(total, 1e-6)
            f0 = f0_base * (1.0 - 0.18 * prog)
            f0 *= 1.0 + 0.008 * math.sin(TWOPI * 5.2 * elapsed)
            f0 *= 1.0 + 0.003 * rng.uniform(-1.0, 1.0)

            formants = [f1 * mouth_scale, f2 * throat_scale, f3]
            # keep ordering and Nyquist distance
            formants[0] = max(150.0, min(formants[0], formants[1] - 120.0))
            formants[1] = max(formants[0] + 120.0, min(formants[1], formants[2] - 150.0))
            formants[2] = max(formants[1] + 150.0, min(formants[2], 0.45 * SR))

            if bw_mode == "scaled":
                bws = [max(50.0, 0.06 * f + 50.0) for f in formants]
            else:
                bws = [70.0, 110.0, 180.0]

            phase += f0 / SR
            if phase >= 1.0:
                phase -= 1.0
            dt = f0 / SR

            if method in ("cascade", "pulse"):
                if source == "square":
                    buzz = (1.0 if phase < 0.5 else -1.0) \
                        + _blep(phase, dt) - _blep((phase + 0.5) % 1.0, dt)
                elif source == "pulse" or method == "pulse":
                    tp = 0.4
                    tn = 0.16
                    if phase < tp:
                        buzz = 0.5 * (1.0 - math.cos(math.pi * phase / tp))
                    elif phase < tp + tn:
                        buzz = math.cos(math.pi * (phase - tp) / (2.0 * tn))
                    else:
                        buzz = 0.0
                else:
                    buzz = 2.0 * phase - 1.0 - _blep(phase, dt)
                y = buzz * va
                for k in range(3):
                    res[k].set(formants[k], bws[k])
                    y = res[k].run(y)
            elif method == "parallel":
                buzz = 2.0 * phase - 1.0 - _blep(phase, dt)
                y = 0.0
                for k in range(3):
                    res[k].set(formants[k], bws[k])
                    y += res[k].run(buzz * va) * gains[k]
            elif method == "additive":
                y = 0.0
                total_amp = 0.0
                kmax = int(min(max_partial_hz, 0.45 * SR) / f0)
                for k in range(1, kmax + 1):
                    fk = k * f0
                    amp = formant_gain(fk, formants, gains, bws)
                    # glottal tilt: two-pole rolloff, -12 dB/oct above the corner
                    tilt = 1.0 / (1.0 + (fk / tilt_hz) ** 2)
                    a = amp * tilt
                    total_amp += a
                    y += a * math.sin(TWOPI * k * phase)
                y *= va * 2.0 / max(total_amp, 1e-9)
            else:
                raise ValueError(method)

            if na > 1e-4:
                nres.set(nh, 700.0)
                y += nres.run(rng.uniform(-1.0, 1.0) * na) * noise_gain

            if lp_hz:
                y = lp.run(y)
            if hp_hz:
                y = hp.run(y)
            out.append(y)
            elapsed += 1.0 / SR

    peak = max(1e-9, max(abs(v) for v in out))
    return [v / peak * 0.85 for v in out]
