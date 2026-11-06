"""Stdlib-only harmonic + noise (HNM-style) analysis/resynthesis.

Public API:
    partials(samples, sr, frame, hop, max_voiced_hz) -> per-frame [(freq_hz, amp)]
    hnm_resynth(samples, sr, frame, hop, max_voiced_hz) -> list[float]

Frames are centred at t = i*hop for i = 0 .. len(samples)//hop + 1.
"""

import math

_TABLES = {}
_PEAK_REL_DB = -40.0  # peaks more than 40 dB below the frame's in-band max are dropped
_TRACK_TOL = 0.05     # relative frequency tolerance for tracking
_EPS = 1e-12


def _tables(n):
    if n not in _TABLES:
        bits = n.bit_length() - 1
        rev = [int(format(i, "0%db" % bits)[::-1], 2) if bits else 0 for i in range(n)]
        twr = [math.cos(-2.0 * math.pi * k / n) for k in range(n // 2)]
        twi = [math.sin(-2.0 * math.pi * k / n) for k in range(n // 2)]
        _TABLES[n] = (rev, twr, twi)
    return _TABLES[n]


def _fft(re, im):
    """In-place-style iterative radix-2 complex FFT; n must be a power of two."""
    n = len(re)
    rev, twr, twi = _tables(n)
    re = [re[i] for i in rev]
    im = [im[i] for i in rev]
    size = 2
    while size <= n:
        half = size // 2
        step = n // size
        for start in range(0, n, size):
            k = 0
            for j in range(start, start + half):
                l = j + half
                wr = twr[k]
                wi = twi[k]
                tr = wr * re[l] - wi * im[l]
                ti = wr * im[l] + wi * re[l]
                re[l] = re[j] - tr
                im[l] = im[j] - ti
                re[j] += tr
                im[j] += ti
                k += step
        size *= 2
    return re, im


def _hann(n):
    return [0.5 - 0.5 * math.cos(2.0 * math.pi * i / n) for i in range(n)]


def _frame_at(samples, centre, n):
    start = centre - n // 2
    out = [0.0] * n
    lo = max(0, -start)
    hi = min(n, len(samples) - start)
    for i in range(lo, hi):
        out[i] = samples[start + i]
    return out


def _n_frames(n_samples, hop):
    return n_samples // hop + 2


def _spectrum_mags(xframe, window):
    n = len(xframe)
    xw = [a * b for a, b in zip(xframe, window)]
    re, im = _fft(xw, [0.0] * n)
    return [math.hypot(re[k], im[k]) for k in range(n // 2 + 1)]


def _peaks(mags, sr, n, max_voiced_hz):
    """Parabolic (log-magnitude) interpolated peaks below max_voiced_hz -> [(f, amp)]."""
    kmax = min(n // 2 - 1, int(math.ceil(max_voiced_hz * n / sr)))
    if kmax < 2:
        return []
    band = mags[1:kmax + 1]
    top = max(band)
    if top <= 0.0:
        return []
    thr = top * 10.0 ** (_PEAK_REL_DB / 20.0)
    out = []
    for k in range(1, kmax + 1):
        m = mags[k]
        if m < thr or m <= mags[k - 1] or m < mags[k + 1]:
            continue
        a = math.log(mags[k - 1] + _EPS)
        b = math.log(m + _EPS)
        c = math.log(mags[k + 1] + _EPS)
        den = a - 2.0 * b + c
        p = 0.5 * (a - c) / den if den < 0.0 else 0.0
        p = max(-0.5, min(0.5, p))
        f = (k + p) * sr / n
        if f >= max_voiced_hz:
            continue
        amp = math.exp(b - 0.25 * (a - c) * p) * 4.0 / n  # Hann coherent gain: sinusoid A -> peak N/4*A
        out.append((f, amp))
    return out


def _analyse(samples, sr, frame, hop, max_voiced_hz):
    """Returns (peaks per frame, high-band magnitudes per frame, kc = first high bin)."""
    window = _hann(frame)
    kc = int(math.ceil(max_voiced_hz * frame / sr))
    peaks, highs = [], []
    for i in range(_n_frames(len(samples), hop)):
        mags = _spectrum_mags(_frame_at(samples, i * hop, frame), window)
        peaks.append(_peaks(mags, sr, frame, max_voiced_hz))
        highs.append(mags[kc:frame // 2])
    return peaks, highs, kc


def _track(peak_frames):
    """Nearest-frequency matching within 5 %. Returns per-frame [(track_id, f, amp)]."""
    active = {}  # track_id -> last frequency
    next_id = 0
    out = []
    for peaks in peak_frames:
        cands = []
        for tid, fp in active.items():
            for m, (fn, _) in enumerate(peaks):
                d = abs(fn - fp) / fp
                if d <= _TRACK_TOL:
                    cands.append((d, tid, m))
        cands.sort()
        used_t, used_m = set(), set()
        frame_out = []
        new_active = {}
        for d, tid, m in cands:
            if tid in used_t or m in used_m:
                continue
            used_t.add(tid)
            used_m.add(m)
            f, a = peaks[m]
            new_active[tid] = f
            frame_out.append((tid, f, a))
        for m, (f, a) in enumerate(peaks):
            if m in used_m:
                continue
            new_active[next_id] = f
            frame_out.append((next_id, f, a))
            next_id += 1
        active = new_active
        out.append(frame_out)
    return out


def partials(samples, sr=22050, frame=1024, hop=256, max_voiced_hz=4000.0):
    """Per-frame spectral peaks below max_voiced_hz as [(freq_hz, amplitude)]."""
    peaks, _, _ = _analyse(samples, sr, frame, hop, max_voiced_hz)
    return peaks


def _lcg(seed):
    state = seed & 0xFFFFFFFF
    while True:
        state = (state * 1664525 + 1013904223) & 0xFFFFFFFF
        yield state / 4294967296.0


def hnm_resynth(samples, sr=22050, frame=1024, hop=256, max_voiced_hz=4000.0):
    """Harmonic (phase-continuous sinusoids) + noise resynthesis, peak-normalised to 0.9."""
    n = len(samples)
    if n == 0:
        return []
    peaks, highs, kc = _analyse(samples, sr, frame, hop, max_voiced_hz)
    tracked = _track(peaks)
    nf = len(tracked)

    # Harmonic part: each track's amplitude/frequency is Hann-cross-faded between
    # frame centres; phase is accumulated per sample (phi(t) = phi(t-1) + 2*pi*f/sr).
    x = [0.0] * n
    phase = {}
    for i in range(nf - 1):
        c0 = i * hop
        c1 = (i + 1) * hop
        t_end = min(c1, n)
        if c0 >= t_end:
            break
        cur = {tid: (f, a) for tid, f, a in tracked[i]}
        nxt = {tid: (f, a) for tid, f, a in tracked[i + 1]}
        for tid in set(cur) | set(nxt):
            if tid in cur and tid in nxt:
                f0, a0 = cur[tid]
                f1, a1 = nxt[tid]
            elif tid in cur:  # dying: fade out at current frequency
                f0, a0 = cur[tid]
                f1, a1 = f0, 0.0
            else:  # born: fade in at next frequency
                f1, a1 = nxt[tid]
                f0, a0 = f1, 0.0
            ph = phase.get(tid, 0.0)
            w = 2.0 * math.pi / sr
            for t in range(c0, t_end):
                u = (t - c0) / hop
                g = 0.5 - 0.5 * math.cos(math.pi * u)
                a = a0 + (a1 - a0) * g
                f = f0 + (f1 - f0) * g
                ph += w * f
                x[t] += a * math.cos(ph)
            phase[tid] = ph

    # Noise part: random-phase high-band spectra with the analysed magnitudes,
    # Hann-windowed overlap-add, then a global gain matching analysed high-band energy.
    window = _hann(frame)
    rnd = _lcg(12345)
    noise = [0.0] * n
    e_target = 0.0
    for i in range(nf):
        mags = highs[i]
        re = [0.0] * frame
        im = [0.0] * frame
        for j, m in enumerate(mags):
            k = kc + j
            th = 2.0 * math.pi * next(rnd)
            yr = m * math.cos(th)
            yi = m * math.sin(th)
            re[k] = yr
            im[k] = yi
            re[frame - k] = yr
            im[frame - k] = -yi
            e_target += m * m
        # real(IFFT(Y)) = real(FFT(conj(Y)))/N
        fr, _ = _fft(re, [-v for v in im])
        start = i * hop - frame // 2
        for q in range(frame):
            t = start + q
            if 0 <= t < n:
                noise[t] += fr[q] / frame * window[q]
    # Calibrate noise high-band energy (same analysis as target) to the analysed energy.
    e_noise = 0.0
    for i in range(nf):
        nf_frame = _frame_at(noise, i * hop, frame)
        mags = _spectrum_mags(nf_frame, window)
        e_noise += sum(m * m for m in mags[kc:frame // 2])
    if e_noise > 0.0 and e_target > 0.0:
        g = math.sqrt(e_target / e_noise)
        for t in range(n):
            x[t] += g * noise[t]

    peak = max(abs(v) for v in x)
    if peak <= 0.0:
        return [0.0] * n
    s = 0.9 / peak
    return [v * s for v in x]
