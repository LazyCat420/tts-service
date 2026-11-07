"""WORLD-style resynthesis of a Frames track into mono audio (numpy only).

Pipeline (all numpy, batched over frames):
  1. Analytic formant envelope per 5 ms frame (complex parallel five-branch
     resonator bank, nasal pole/zero, spectral tilt, radiation) evaluated on
     the 2048-point FFT grid.
  2. Excitation: tilted pulse train (voiced), 5-band mixed periodic/aperiodic
     split with modified velvet noise, aspiration, 6-band parallel frication,
     and flat low-pass bypass.
  3. Magnitude-only STFT multiplication (2048-pt Hann, hop 512) of the vocal-
     tract excitation by the interpolated envelope, summed with parallel
     frication and bypass; ISTFT with Hann OLA.
  4. Band-limit 80 Hz .. 0.45 sr with FFT masks; peak-normalise to 0.85.
"""

from __future__ import annotations

import functools

import numpy as np

from .frames import FRAME_HZ, Frames, peak_amplitudes, resonator_mag

N_FFT = 2048
HOP = 512
N_BINS = N_FFT // 2 + 1
N_MEL = 80
N_CEP = 34
PEAK = 0.85

NASAL_POLE_BW = 100.0
NASAL_ZERO_BW = 100.0
FRIC_CENTRES = np.array([500.0, 1000.0, 1800.0, 2800.0, 3800.0, 4900.0])
MVN_EDGES = np.array([0.0, 500.0, 1500.0, 3000.0, 6000.0, np.inf])
MVN_TD = 4
VOICED_AP_FLOOR = 0.015
ASP_POLE = 0.75
ASP_KERNEL_LEN = 60
BYPASS_LP_PASS_HZ = 4000.0
BYPASS_LP_STOP_HZ = 6000.0
TILT_DECAY_PER_DB = 0.033
TILT_DECAY_MAX = 0.95
PULSE_TAIL_LEN = 200
BAND_LIMIT_LO_HZ = 80.0
BAND_LIMIT_HI_FRAC = 0.45


def _hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + np.asarray(f, dtype=np.float64) / 700.0)


def _mel_to_hz(m):
    return 700.0 * (10.0 ** (np.asarray(m, dtype=np.float64) / 2595.0) - 1.0)


@functools.lru_cache(maxsize=4)
def _tables(sr: int):
    """Constant matrices for a given sample rate (cached)."""
    freqs = np.arange(N_BINS) * (sr / N_FFT)

    # HTK triangular mel filterbank on FFT bins, rows normalised to unit sum.
    mel_edges = np.linspace(_hz_to_mel(0.0), _hz_to_mel(sr / 2.0), N_MEL + 2)
    edges_hz = _mel_to_hz(mel_edges)
    lo = edges_hz[:-2, None]
    ce = edges_hz[1:-1, None]
    hi = edges_hz[2:, None]
    tri = np.minimum((freqs[None, :] - lo) / (ce - lo), (hi - freqs[None, :]) / (hi - ce))
    mel_w = np.maximum(tri, 0.0)
    mel_w /= mel_w.sum(axis=1, keepdims=True)

    # Mel-domain interpolation back to FFT bins (bins x bands), band centres = triangle peaks.
    centres_mel = mel_edges[1:-1]
    bin_mel = _hz_to_mel(freqs)
    eye = np.eye(N_MEL)
    mel_to_bins = np.stack(
        [np.interp(bin_mel, centres_mel, eye[:, i]) for i in range(N_MEL)], axis=1
    )

    # DCT-II analysis (N_MEL x N_CEP) and truncated DCT-III synthesis (N_CEP x N_MEL).
    i = np.arange(N_MEL)
    j = np.arange(N_CEP)
    dct_fwd = np.cos(np.pi * np.outer(i + 0.5, j) / N_MEL)
    dct_inv = dct_fwd.T * (2.0 / N_MEL)
    dct_inv[0] *= 0.5

    # Mixed-excitation band masks (hard edges, sum to 1 at every bin).
    mvn_masks = np.stack(
        [(freqs >= MVN_EDGES[b]) & (freqs < MVN_EDGES[b + 1]) for b in range(5)]
    ).astype(np.float64)

    # Frication: Hann-shaped bands, full width = centre / 2.
    c = FRIC_CENTRES[:, None]
    w = c / 2.0
    d = freqs[None, :] - c
    fric_masks = np.where(np.abs(d) <= w / 2.0, 0.5 * (1.0 + np.cos(2.0 * np.pi * d / w)), 0.0)

    # Bypass low-pass: unity to 4 kHz, raised-cosine to 0 at 6 kHz.
    x = (freqs - BYPASS_LP_PASS_HZ) / (BYPASS_LP_STOP_HZ - BYPASS_LP_PASS_HZ)
    bypass_lp = np.where(x <= 0, 1.0, np.where(x >= 1, 0.0, 0.5 * (1.0 + np.cos(np.pi * x))))

    return {
        "mel_w": mel_w,
        "mel_to_bins": mel_to_bins,
        "dct_fwd": dct_fwd,
        "dct_inv": dct_inv,
        "mvn_masks": mvn_masks,
        "fric_masks": fric_masks,
        "bypass_lp": bypass_lp,
        "freqs": freqs,
    }


def _at(arr: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """Linear interpolation of arr (frame axis 0) at fractional frame positions."""
    last = arr.shape[0] - 1
    pos = np.clip(pos, 0.0, last)
    i0 = np.floor(pos).astype(np.int64)
    i1 = np.minimum(i0 + 1, last)
    w = (pos - i0).reshape((-1,) + (1,) * (arr.ndim - 1))
    return arr[i0] * (1.0 - w) + arr[i1] * w


def _res_complex(hz: np.ndarray, formant: np.ndarray, bw: np.ndarray, sr: int) -> np.ndarray:
    """Complex transfer function H(e^{jw}) of the digital two-pole resonator."""
    bw = np.maximum(bw, 1.0)
    r = np.exp(-np.pi * bw / sr)
    c = -(r * r)
    b = 2.0 * r * np.cos(2.0 * np.pi * formant / sr)
    a = 1.0 - b - c
    w = 2.0 * np.pi * hz / sr
    z1 = np.exp(-1j * w)
    z2 = np.exp(-2j * w)
    denom = 1.0 - b * z1 - c * z2
    return a / np.where(np.abs(denom) > 1e-12, denom, 1e-12)


def _envelope(frames: Frames, sr: int, tb: dict) -> np.ndarray:
    """Analytic parallel-bank envelope magnitude, shape (n_frames, N_BINS)."""
    f = tb["freqs"]
    formants = np.asarray(frames.formants, dtype=np.float64)
    bws = np.asarray(frames.bws, dtype=np.float64)
    f0 = np.asarray(frames.f0, dtype=np.float64)[:, None]
    n = formants.shape[0]

    # Match klatt's source transfer shape (two-pole 400 Hz tilt + radiation zero)
    # and evaluate each branch's compensation at its nearest harmonic of F0 so
    # the radiated harmonic peaks land directly on FORMANT_AMP.
    d = np.exp(-2.0 * np.pi * 400.0 / sr)
    w = 2.0 * np.pi * f[None, :] / sr
    src_shape = np.maximum(2.0 * np.sin(0.5 * w), 1e-4) / (1.0 - 2.0 * d * np.cos(w) + d * d)

    def _src_at(hz: np.ndarray) -> np.ndarray:
        wh = 2.0 * np.pi * hz / sr
        return np.maximum(2.0 * np.sin(0.5 * wh), 1e-4) / (1.0 - 2.0 * d * np.cos(wh) + d * d)

    oq = np.asarray(frames.oq, dtype=np.float64)[:, None]
    tilt = np.asarray(frames.tilt_db, dtype=np.float64)[:, None]
    decay = np.clip(TILT_DECAY_PER_DB * tilt, 0.0, TILT_DECAY_MAX)
    pulse_tf = (1.0 - decay) / np.sqrt(1.0 - 2.0 * decay * np.cos(w) + decay * decay)
    delta_tilt = (tilt - 6.0) + 12.0 * (oq - 0.50)

    amps = peak_amplitudes(frames)
    f1_ref = np.maximum(formants[:, :1], 1.0)
    h_sum = np.zeros((n, N_BINS), dtype=np.complex128)
    lvl_0 = np.ones((n, 1))
    for k in range(5):
        centre = formants[:, k:k + 1]
        bw = bws[:, k:k + 1]
        hk_harm = np.where(f0 > 0, np.maximum(1.0, np.round(centre / np.maximum(f0, 1.0))) * f0, centre)
        lvl_k = 0.5 * (resonator_mag(hk_harm, centre, bw, sr) * _src_at(hk_harm)) + 0.5 * (resonator_mag(centre, centre, bw, sr) * _src_at(centre))
        if k == 0:
            lvl_0 = lvl_k
        tilt_mult = (centre / f1_ref) ** (-delta_tilt / 18.0)
        g_k = (amps[k] * tilt_mult) * (lvl_0 / np.maximum(lvl_k, 1e-12))
        h_sum += g_k * _res_complex(f[None, :], centre, bw, sr)

    mag = np.abs(h_sum) * src_shape / np.maximum(pulse_tf, 1e-6)

    # Nasal pole/zero: pole boosts around fnp (~270 Hz), zero carves a notch
    # around fnz (~450 Hz). When fnp == fnz, pole_mag == zero_mag (unity).
    fnp = np.asarray(frames.fnp, dtype=np.float64)[:, None]
    fnz = np.asarray(frames.fnz, dtype=np.float64)[:, None]
    on_p = fnp > 0
    on_z = fnz > 0
    pole_hz = np.where(on_p, fnp, 1.0)
    zero_hz = np.where(on_z, fnz, 1.0)
    pole_mag = (resonator_mag(f[None, :], pole_hz, NASAL_POLE_BW, sr)
                / resonator_mag(pole_hz, pole_hz, NASAL_POLE_BW, sr))
    zero_mag = (resonator_mag(f[None, :], zero_hz, NASAL_ZERO_BW, sr)
                / resonator_mag(zero_hz, zero_hz, NASAL_ZERO_BW, sr))
    mag *= np.where(on_p, pole_mag, 1.0)
    mag /= np.where(on_z, np.maximum(zero_mag, 0.1), 1.0)

    mag /= np.maximum(mag.max(axis=1, keepdims=True), 1e-15)
    return mag


def _velvet(n_samples: int, rng: np.random.Generator) -> np.ndarray:
    """Modified velvet noise: one impulse per Td-sample block, +-1, pair-balanced, unit RMS."""
    m_count = n_samples // MVN_TD + 2
    m = np.arange(m_count)
    pos = m * MVN_TD + np.floor(rng.random(m_count) * MVN_TD).astype(np.int64)
    pair_sign = np.where(rng.random(m_count // 2 + 1) < 0.5, 1.0, -1.0)
    sign = pair_sign[m // 2] * np.where(m % 2 == 0, 1.0, -1.0)
    out = np.zeros(n_samples)
    keep = pos < n_samples
    out[pos[keep]] = sign[keep]
    return out * np.sqrt(MVN_TD)


def _tilted_pulses(f0_s: np.ndarray, av_s: np.ndarray, tilt_s: np.ndarray, sr: int) -> np.ndarray:
    """Period-start impulse train (unit RMS at av=1) through a unit-DC one-pole."""
    n = f0_s.shape[0]
    inc = f0_s / sr
    phase = np.cumsum(inc)
    is_pulse = (np.floor(phase) > np.floor(phase - inc)) & (f0_s > 0)
    p_idx = np.nonzero(is_pulse)[0]
    out = np.zeros(n)
    if p_idx.size == 0:
        return out
    # Scale by sqrt(sr / f0) so a unit-av pulse train has unit RMS, matching _velvet.
    amp = av_s[p_idx] * np.sqrt(sr / np.maximum(f0_s[p_idx], 20.0))
    decay = np.clip(TILT_DECAY_PER_DB * tilt_s[p_idx], 0.0, TILT_DECAY_MAX)
    gain = amp * (1.0 - decay)
    for k in range(PULSE_TAIL_LEN):
        idx = p_idx + k
        ok = idx < n
        if not ok.any():
            break
        np.add.at(out, idx[ok], (gain * decay ** k)[ok])
    return out


def _grid(n_samples: int):
    """STFT framing: frame t covers original samples [t*HOP - N_FFT/2, t*HOP + N_FFT/2)."""
    t_count = n_samples // HOP + 2
    pad = N_FFT // 2
    padded_len = max(n_samples + 2 * pad, (t_count - 1) * HOP + N_FFT)
    idx = np.arange(t_count)[:, None] * HOP + np.arange(N_FFT)[None, :]
    win = 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(N_FFT) / N_FFT)
    norm = np.zeros(padded_len)
    np.add.at(norm, idx.ravel(), np.broadcast_to(win * win, (t_count, N_FFT)).ravel())
    return t_count, pad, padded_len, idx, win, norm


def _analyse(x: np.ndarray, grid) -> np.ndarray:
    _, pad, padded_len, idx, win, _ = grid
    xp = np.zeros(padded_len)
    xp[pad:pad + x.shape[0]] = x
    return np.fft.rfft(xp[idx] * win, axis=1)


def _synthesise(spec: np.ndarray, n_samples: int, grid) -> np.ndarray:
    _, pad, padded_len, idx, win, norm = grid
    frames = np.fft.irfft(spec, n=N_FFT, axis=1) * win
    acc = np.zeros(padded_len)
    np.add.at(acc, idx.ravel(), frames.ravel())
    safe = np.where(norm > 1e-6, norm, 1.0)
    acc = np.where(norm > 1e-6, acc / safe, 0.0)
    return acc[pad:pad + n_samples]


def render(frames: Frames, sr: int = 24000, rng: np.random.Generator | None = None) -> np.ndarray:
    """Render a Frames track to mono float64 audio with peak 0.85."""
    n = len(frames.f0)
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    if rng is None:
        rng = np.random.default_rng(0)
    n_samples = int(round(n * sr / FRAME_HZ))
    if n_samples <= 0:
        return np.zeros(0, dtype=np.float64)

    tb = _tables(int(sr))

    f0 = np.asarray(frames.f0, dtype=np.float64)
    av = np.asarray(frames.av, dtype=np.float64)
    ah = np.asarray(frames.ah, dtype=np.float64)
    af = np.asarray(frames.af, dtype=np.float64)
    fric = np.asarray(frames.fric, dtype=np.float64)
    bypass = np.asarray(frames.bypass, dtype=np.float64)
    tilt = np.asarray(frames.tilt_db, dtype=np.float64)

    # Per-sample controls (sample n sits at fractional frame index n*FRAME_HZ/sr).
    pos_s = np.arange(n_samples) * (FRAME_HZ / sr)
    f0_s = _at(f0, pos_s)
    av_s = _at(av, pos_s)
    ah_s = _at(ah, pos_s)
    af_s = _at(af, pos_s)
    tilt_s = _at(tilt, pos_s)

    # Excitation sources, time domain.
    pulses = _tilted_pulses(f0_s, av_s, tilt_s, sr)
    velvet = _velvet(n_samples, rng) * av_s
    aspiration = np.convolve(
        rng.uniform(-1.0, 1.0, n_samples),
        ASP_POLE ** np.arange(ASP_KERNEL_LEN),
    )[:n_samples] * ah_s
    fric_white = rng.uniform(-1.0, 1.0, n_samples) * af_s
    bypass_white = rng.uniform(-1.0, 1.0, n_samples) * af_s

    # STFT grid and per-frame controls (frame t centre = original sample t*HOP).
    grid = _grid(n_samples)
    t_count = grid[0]
    pos_t = np.arange(t_count) * (HOP * FRAME_HZ / sr)

    envelope = _envelope(frames, sr, tb)
    env_t = _at(envelope, pos_t)
    f0_t = _at(f0, pos_t)
    fric_t = _at(fric, pos_t)
    bypass_t = _at(bypass, pos_t)
    ap_raw = np.clip(_at(af, pos_t) + _at(ah, pos_t), 0.0, 1.0)

    # Per-band aperiodicity: voiced frames ramp from 0.05 (lowest band) to ap (highest band);
    # unvoiced frames are fully aperiodic in every band.
    voiced = (f0_t > 0)[:, None]
    ap_v = np.maximum(ap_raw, VOICED_AP_FLOOR)[:, None]
    band_frac = np.arange(5) / 4.0
    ap_tb = np.where(voiced, VOICED_AP_FLOOR + (ap_v - VOICED_AP_FLOOR) * band_frac[None, :], 1.0)

    mvn = tb["mvn_masks"]
    periodic_gain = (1.0 - ap_tb) @ mvn
    aperiodic_gain = ap_tb @ mvn

    s_pulse = _analyse(pulses, grid)
    s_velvet = _analyse(velvet, grid)
    s_asp = _analyse(aspiration, grid)
    s_fric = _analyse(fric_white, grid)
    s_bypass = _analyse(bypass_white, grid)

    fric_gain = fric_t @ tb["fric_masks"]
    bypass_gain = bypass_t[:, None] * tb["bypass_lp"][None, :]

    tract_spec = (s_pulse * periodic_gain + s_velvet * aperiodic_gain + s_asp) * env_t
    par_spec = (s_fric * fric_gain + s_bypass * bypass_gain) * 0.25
    y = _synthesise(tract_spec + par_spec, n_samples, grid)

    # Band-limit 80 Hz .. 0.45 sr with FFT masks.
    spectrum = np.fft.rfft(y)
    freqs = np.fft.rfftfreq(n_samples, 1.0 / sr)
    keep = (freqs >= BAND_LIMIT_LO_HZ) & (freqs <= BAND_LIMIT_HI_FRAC * sr)
    y = np.fft.irfft(np.where(keep, spectrum, 0.0), n=n_samples)

    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak > 0.0:
        y = y * (PEAK / peak)
    return y.astype(np.float64)
