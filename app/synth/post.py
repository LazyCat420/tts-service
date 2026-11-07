"""Output-stage conditioning for the synthesiser (numpy + stdlib only).

Four stages, all standard published techniques:

1. de-esser   : dynamic attenuation of the 5-8 kHz band on frames whose
                sibilant energy exceeds the running speech average.
2. LTAS tilt  : a smooth spectral-slope correction. Natural speech falls about
                6-9 dB/octave above 500 Hz; formant synthesis tends to be
                flatter, which reads as "bright" or "harsh".
3. CELP postfilter: zero-phase Hz-band formant presence & harmonic clarity
                enhancer, boosting the average-PSD formant peaks against their
                local valleys in the 250..3800 Hz band via the STFT magnitude,
                with the output RMS-matched to the input.
4. Loudness   : BS.1770 K-weighted, gated integrated loudness, normalised to
                the voice's target LUFS.
"""

from __future__ import annotations

import math

import numpy as np

# ── filters ────────────────────────────────────────────────────────────────


def _biquad(kind: str, sr: float, f: float, q: float = 0.7071,
            gain_db: float = 0.0) -> tuple[float, float, float, float, float]:
    """RBJ biquad coefficients (b0, b1, b2, a1, a2)."""
    w = 2.0 * math.pi * f / sr
    c, s = math.cos(w), math.sin(w)
    al = s / (2.0 * q)
    if kind == "lp":
        b0 = b2 = (1.0 - c) / 2.0
        b1 = 1.0 - c
    elif kind == "hp":
        b0 = b2 = (1.0 + c) / 2.0
        b1 = -(1.0 + c)
    elif kind == "shelf":
        # RBJ high-shelf filter (used by BS.1770 K-weighting at 1681.97 Hz)
        a = 10.0 ** (gain_db / 40.0)
        b0 = a * ((a + 1.0) + (a - 1.0) * c + 2.0 * math.sqrt(a) * al)
        b1 = -2.0 * a * ((a - 1.0) + (a + 1.0) * c)
        b2 = a * ((a + 1.0) + (a - 1.0) * c - 2.0 * math.sqrt(a) * al)
        a0 = (a + 1.0) - (a - 1.0) * c + 2.0 * math.sqrt(a) * al
        a1 = 2.0 * ((a - 1.0) - (a + 1.0) * c)
        a2 = (a + 1.0) - (a - 1.0) * c - 2.0 * math.sqrt(a) * al
        return b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0
    elif kind == "peak":
        a = 10.0 ** (gain_db / 40.0)
        b0 = 1.0 + al * a
        b1 = -2.0 * c
        b2 = 1.0 - al * a
        a0 = 1.0 + al / a
        a1 = -2.0 * c
        a2 = 1.0 - al / a
        return b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0
    else:
        raise ValueError(kind)
    a0 = 1.0 + al
    return b0 / a0, b1 / a0, b2 / a0, -2.0 * c / a0, (1.0 - al) / a0


def _run(x: np.ndarray, co: tuple[float, float, float, float, float]) -> np.ndarray:
    b0, b1, b2, a1, a2 = co
    y = np.empty_like(x)
    x1 = x2 = y1 = y2 = 0.0
    for i, v in enumerate(x):
        o = b0 * v + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
        x2 = x1
        x1 = v
        y2 = y1
        y1 = o
        y[i] = o
    return y


# ── de-esser ───────────────────────────────────────────────────────────────


def deess(x: np.ndarray, sr: int, amount: float = 0.25,
          lo: float = 5000.0, hi: float | None = None) -> np.ndarray:
    """Attenuate the sibilance band on frames that carry too much of it.

    `amount` 0..1 scales the maximum reduction (amount=1 -> -12 dB).
    """
    if amount <= 0.0:
        return x
    hi = hi or min(8000.0, 0.45 * sr)
    hp = _run(x, _biquad("hp", sr, lo))
    if hi < 0.45 * sr:
        hp = _run(hp, _biquad("lp", sr, hi))
    win = max(64, int(sr * 0.02))
    hop = win // 2
    n = 1 + max(0, (len(x) - win) // hop)
    if n <= 1:
        return x
    idx = np.arange(win)[None, :] + hop * np.arange(n)[:, None]
    env_hi = np.sqrt(np.mean(hp[idx] ** 2, axis=1))
    env_all = np.sqrt(np.mean(x[idx] ** 2, axis=1))
    ref = float(np.median(env_all)) or 1e-9
    ratio = env_hi / np.maximum(env_all, 1e-9)
    gate = np.clip((ratio - 0.35) / 0.45, 0.0, 1.0) * np.clip(env_hi / (0.6 * ref), 0.0, 1.0)
    gain_db = -12.0 * amount * gate
    centres = hop * np.arange(n) + win // 2
    smooth = np.interp(np.arange(len(x)), centres, gain_db)
    return x * (10.0 ** (smooth / 20.0))


# ── LTAS tilt ──────────────────────────────────────────────────────────────


LTAS_SLOPE = {"neutral": -6.0, "dark": -9.0, "bright": -3.0}


def _stft_apply(x: np.ndarray, sr: int, gain: np.ndarray, nfft: int = 4096) -> np.ndarray:
    """Magnitude-only STFT multiply with Hann overlap-add (phase preserved)."""
    if len(x) == 0:
        return x
    hop = nfft // 2
    win = 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(nfft) / nfft)
    pad = np.zeros(len(x) + 2 * hop)
    pad[hop:hop + len(x)] = x
    n = 1 + max(0, (len(pad) - nfft) // hop)
    idx = np.arange(nfft)[None, :] + hop * np.arange(n)[:, None]
    spec = np.fft.rfft(pad[idx] * win, axis=1) * gain[None, :]
    frames = np.fft.irfft(spec, n=nfft, axis=1) * win
    acc = np.zeros_like(pad)
    np.add.at(acc, idx.ravel(), frames.ravel())
    norm = np.zeros_like(pad)
    np.add.at(norm, idx.ravel(), np.tile(win ** 2, n))
    acc = np.where(norm > 1e-6, acc / np.where(norm > 1e-6, norm, 1.0), 0.0)
    return acc[hop:hop + len(x)]


def ltas_tilt(x: np.ndarray, sr: int, kind: str = "neutral",
              pivot: float = 500.0, limit_db: float = 6.0) -> np.ndarray:
    """Correct the long-term spectral slope to the target dB/octave.

    Measured in third-octave bands, so it is a gentle overall shape change and
    cannot fight the formant structure. The correction is capped at
    `limit_db` so a pathological input cannot be over-equalised.
    """
    slope = LTAS_SLOPE.get(kind, -6.0)
    n = len(x)
    if n < 1024:
        return x
    nfft = 4096
    hop = nfft // 2
    starts = np.arange(0, max(1, n - nfft + 1), hop)
    if starts.size == 0:
        return x
    idx = starts[:, None] + np.arange(nfft)[None, :]
    psd = np.mean(np.abs(np.fft.rfft(x[idx] * np.hanning(nfft), axis=1)) ** 2, axis=0)
    freqs = np.fft.rfftfreq(nfft, 1.0 / sr)
    centres = 100.0 * 2.0 ** (np.arange(0, 25) / 3.0)
    centres = centres[(centres >= 80.0) & (centres < 0.45 * sr)]
    if centres.size < 4:
        return x
    levels = []
    for c in centres:
        lo, hi = c / 2.0 ** (1.0 / 6.0), c * 2.0 ** (1.0 / 6.0)
        m = (freqs >= lo) & (freqs < hi)
        levels.append(10.0 * math.log10(float(np.mean(psd[m])) + 1e-20) if m.any() else -100.0)
    levels = np.asarray(levels)
    above = (centres >= pivot) & (centres <= 4000.0)
    if above.sum() < 2:
        return x
    fit = np.polyfit(np.log2(centres[above]), levels[above], 1)
    delta = max(-limit_db, min(limit_db, slope - float(fit[0])))
    corr_db = np.clip(delta * np.log2(np.maximum(centres, pivot) / pivot), -limit_db, limit_db)
    gain = 10.0 ** (np.interp(freqs, centres, corr_db) / 20.0)
    return _stft_apply(x, sr, gain)


# ── CELP postfilter ────────────────────────────────────────────────────────


def celp_postfilter(x: np.ndarray, sr: int, strength: float = 0.8,
                    gamma1: float = 0.55, gamma2: float = 0.80,
                    mu: float = 0.45, order: int = 10) -> np.ndarray:
    """Zero-phase Hz-band formant presence & harmonic clarity enhancer.

    Replaces the legacy global LPC-10 postfilter: one time-domain IIR fit
    across the whole utterance warped harmonic phases and amplified the
    inter-harmonic valleys. Instead, the average PSD is split into a wide
    macro spectral trend (~600 Hz moving average) and the local formant
    peaks (~150 Hz moving average) inside the 250..3800 Hz intelligibility
    band; the local peak-to-valley contrast is boosted as a smooth, zero-
    phase magnitude gain applied through the STFT, so harmonic phases stay
    untouched and only genuine formant peaks are emphasised. The output is
    RMS-matched to the input, so the overall level is unchanged.
    """
    if strength <= 0.0:
        return x
    x = np.asarray(x, float)
    nfft = min(2048, 2 ** int(math.log2(max(256, len(x)))))
    if nfft < 1024:
        return x  # too short for a stable average PSD / STFT overlap-add
    hop = nfft // 2
    starts = np.arange(0, max(1, len(x) - nfft + 1), hop)
    idx = starts[:, None] + np.arange(nfft)[None, :]
    psd = np.mean(np.abs(np.fft.rfft(x[idx] * np.hanning(nfft), axis=1)) ** 2, axis=0)
    freqs = np.fft.rfftfreq(nfft, 1.0 / sr)
    local_db = 10.0 * np.log10(psd + 1e-20)

    def _hz_ma(db: np.ndarray, width_hz: float) -> np.ndarray:
        half = max(1, int(round(0.5 * width_hz / (sr / nfft))))
        pad = np.concatenate([np.full(half, db[0]), db, np.full(half, db[-1])])
        c = np.concatenate([[0.0], np.cumsum(pad)])
        return (c[2 * half + 1: 2 * half + 1 + len(db)] - c[0:len(db)]) / (2 * half + 1)

    local_db = _hz_ma(local_db, 150.0)
    macro_db = _hz_ma(10.0 * np.log10(psd + 1e-20), 600.0)

    # Smooth taper: full weight across 250..3800 Hz, easing to 0 dB below
    # 200 Hz and above 4000 Hz so the enhancer never touches the sub-bass
    # or the sibilant air band.
    lo_t = np.clip((freqs - 200.0) / 50.0, 0.0, 1.0)
    hi_t = np.clip((4000.0 - freqs) / 200.0, 0.0, 1.0)
    taper = lo_t * hi_t
    taper = taper * taper * (3.0 - 2.0 * taper)
    contrast_db = np.clip(local_db - macro_db, -3.0, 4.5) * (0.45 * strength) * taper
    gain = 10.0 ** (contrast_db / 20.0)
    y = _stft_apply(x, sr, gain, nfft=nfft)
    in_rms = float(np.sqrt(np.mean(x ** 2)))
    out_rms = float(np.sqrt(np.mean(y ** 2)))
    if in_rms > 1e-9 and out_rms > 1e-9:
        y = y * (in_rms / out_rms)
    return y


# ── loudness ───────────────────────────────────────────────────────────────


def _k_weight(x: np.ndarray, sr: int) -> np.ndarray:
    """BS.1770 K-weighting: a +4 dB high-shelf at 1681.97 Hz then a 38.13 Hz RLB high-pass."""
    y = _run(x, _biquad("shelf", sr, 1681.97, 0.7071, 3.999))
    return _run(y, _biquad("hp", sr, 38.13, 0.5003))


def loudness_lufs(x: np.ndarray, sr: int) -> float:
    """Gated integrated loudness per BS.1770 (mono)."""
    y = _k_weight(np.asarray(x, float), sr)
    win = int(0.4 * sr)
    hop = int(0.1 * sr)
    if len(y) < win:
        return -70.0
    blocks = np.array([np.sum(y[i:i + win] ** 2) / win
                       for i in range(0, len(y) - win + 1, hop)])
    if blocks.size == 0:
        return -70.0
    with np.errstate(divide="ignore"):
        lufs = -0.691 + 10.0 * np.log10(np.maximum(blocks, 1e-20))
    gated = lufs[lufs > -70.0]
    if gated.size == 0:
        return -70.0
    rel = -0.691 + 10.0 * np.log10(np.mean(10 ** ((gated + 0.691) / 10.0))) - 10.0
    kept = gated[gated > rel]
    if kept.size == 0:
        kept = gated
    return float(-0.691 + 10.0 * np.log10(np.mean(10 ** ((kept + 0.691) / 10.0))))


def match_loudness(x: np.ndarray, sr: int, target_lufs: float = -16.0,
                   peak_ceiling: float = 0.98) -> np.ndarray:
    y = np.asarray(x, float)
    cur = loudness_lufs(y, sr)
    if cur >= -60.0:
        gain_db = max(-12.0, min(12.0, target_lufs - cur))
        y = y * 10.0 ** (gain_db / 20.0)
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak > peak_ceiling:
        # linear peak fit, never soft-clipping: tanh on a large gain turns the
        # whole waveform into a square wave and reads as crackle
        y = y * (peak_ceiling / peak)
    return y


# ── pipeline ───────────────────────────────────────────────────────────────


def finish(x: np.ndarray, sr: int, voice: dict | None = None) -> np.ndarray:
    """Run the post chain described by `voice['post']`."""
    v = voice or {}
    post = v.get("post", {})
    default_celp = 0.0 if float(v.get("whisper", 0.0)) > 0.0 else 0.20
    y = np.asarray(x, float)
    y = deess(y, sr, float(post.get("deess", 0.0)))
    y = ltas_tilt(y, sr, post.get("ltas", "neutral"))
    y = celp_postfilter(y, sr, float(post.get("celp", default_celp)))
    y = match_loudness(y, sr, float(post.get("lufs", -16.0)))
    return y
