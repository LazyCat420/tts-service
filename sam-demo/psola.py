import math

def _period(x, c, sr, fmin, fmax):
    h = int(0.015 * sr)
    seg = x[max(0, c - h):c + h]
    lo, hi = int(sr / fmax), int(sr / fmin)
    if len(seg) < hi + 2:
        return 0
    r0 = sum(v * v for v in seg)
    if r0 < 1e-9:
        return 0
    cs = {}
    for lag in range(lo, min(hi, len(seg) - 1) + 1):
        cs[lag] = sum(seg[i] * seg[i + lag] for i in range(len(seg) - lag)) / r0
    best = max(cs.values())
    if best < 0.3:
        return 0
    peaks = [l for l in cs if cs[l] >= 0.85 * best and cs[l] >= cs.get(l - 1, -9) and cs[l] >= cs.get(l + 1, -9)]
    return min(peaks)  # smallest lag = highest freq, rejects octave-down errors

def _marks(x, sr, fmin, fmax):
    """returns list of (position, local_period)"""
    n = len(x)
    res = []
    pos = 0
    dflt = int(sr / 100)
    while pos < n:
        p = _period(x, pos, sr, fmin, fmax) or dflt
        q = int(p // 4)
        lo, hi = max(0, pos - q), min(n, pos + q + 1)
        m = max(range(lo, hi), key=lambda i: x[i]) if res or True else pos
        if res and m <= res[-1][0]:
            m = res[-1][0] + p
        res.append((m, p))
        pos = m + p
    return res

def pitch_marks(samples, sr=22050, fmin=60, fmax=400):
    return [m for m, _ in _marks(list(samples), sr, fmin, fmax)]

def td_psola(samples, sr=22050, time_factor=1.0, pitch_factor=1.0):
    x = list(samples)
    n = len(x)
    mk = _marks(x, sr, 60, 400)
    pos = [m for m, _ in mk]
    m_out = int(n * time_factor)
    y = [0.0] * (m_out + 4 * 1000)
    wsum = [0.0] * len(y)
    t = float(pos[0]) * time_factor
    while t < m_out:
        src = t / time_factor
        j = min(range(len(pos)), key=lambda i: abs(pos[i] - src)) if len(pos) < 50 else _near(pos, src)
        c, p = pos[j], mk[j][1]
        L = p
        for k in range(-L, L + 1):
            si, oi = c + k, int(round(t)) + k
            if 0 <= si < n and 0 <= oi < len(y):
                w = 0.5 * (1 + math.cos(math.pi * k / L))
                y[oi] += x[si] * w
                wsum[oi] += w
        t += p / pitch_factor
    y = y[:m_out]
    ws = wsum[:m_out]
    y = [v / w if w > 0.2 else v for v, w in zip(y, ws)]
    px = max(abs(v) for v in x) or 1.0
    py = max(abs(v) for v in y) or 1.0
    return [v * px / py for v in y]

def _near(pos, s):
    lo, hi = 0, len(pos) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if pos[mid] < s: lo = mid + 1
        else: hi = mid
    if lo > 0 and abs(pos[lo - 1] - s) <= abs(pos[lo] - s):
        return lo - 1
    return lo
