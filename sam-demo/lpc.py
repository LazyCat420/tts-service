import math, random

def _acf(x, p):
    n = len(x)
    return [sum(x[i] * x[i + k] for i in range(n - k)) for k in range(p + 1)]

def _levinson(r, p):
    a = [1.0] + [0.0] * p
    e = r[0] * (1 + 1e-9) + 1e-12
    for i in range(1, p + 1):
        acc = r[i] + sum(a[j] * r[i - j] for j in range(1, i))
        k = -acc / e
        na = a[:]
        for j in range(1, i):
            na[j] = a[j] + k * a[i - j]
        na[i] = k
        a = na
        e *= (1 - k * k)
    return a[1:], e

def _f0(x, sr, fmin=60, fmax=400):
    lo, hi = int(sr / fmax), min(int(sr / fmin), len(x) - 1)
    r0 = sum(v * v for v in x) + 1e-12
    best, bl = 0.0, 0
    for lag in range(lo, hi + 1):
        c = sum(x[i] * x[i + lag] for i in range(len(x) - lag)) / r0
        if c > best:
            best, bl = c, lag
    return (sr / bl if bl else 0.0), best

def lpc_resynth(samples, sr=22050, order=10, frame=400, hop=100, preemph=0.97):
    x = [samples[0]] + [samples[i] - preemph * samples[i - 1] for i in range(1, len(samples))]
    n = len(x)
    win = [0.54 - 0.46 * math.cos(2 * math.pi * i / (frame - 1)) for i in range(frame)]
    w2 = sum(w * w for w in win)
    A, G, F, V = [], [], [], []
    for s in range(0, max(1, n - frame + 1), hop):
        seg = x[s:s + frame]
        seg += [0.0] * (frame - len(seg))
        xw = [seg[i] * win[i] for i in range(frame)]
        a, e = _levinson(_acf(xw, order), order)
        a = [a[i] * 0.98 ** (i + 1) for i in range(order)]  # mild bandwidth expansion
        f0, v = _f0(seg, sr)
        v = 0.0 if v < 0.3 else min(1.0, v)
        A.append(a); G.append(math.sqrt(max(e, 0) / w2)); F.append(f0 if v else 0.0); V.append(v)
    nf = len(A)
    rnd = random.Random(1)
    y = [0.0] * n
    hist = [0.0] * order
    phase = 0.0
    for t in range(n):
        pos = (t - frame / 2) / hop
        pos = min(max(pos, 0.0), nf - 1)
        i0 = int(pos); i1 = min(i0 + 1, nf - 1); fr = pos - i0
        a = [A[i0][k] * (1 - fr) + A[i1][k] * fr for k in range(order)]
        g = G[i0] * (1 - fr) + G[i1] * fr
        v = V[i0]; f0 = F[i0] or 100.0
        period = sr / f0
        phase += 1.0
        pulse = 0.0
        if phase >= period:
            phase -= period
            pulse = math.sqrt(period)
        exc = g * (v * pulse + (1 - v) * rnd.gauss(0, 1) * 1.0 + 0.2 * v * rnd.gauss(0, 1))
        out = exc - sum(a[k] * hist[k] for k in range(order))
        hist = [out] + hist[:-1]
        y[t] = out
    o = [0.0] * n
    acc = 0.0
    for t in range(n):
        acc = y[t] + preemph * acc
        o[t] = acc
    pk = max(abs(v) for v in o) or 1.0
    return [v * 0.9 / pk for v in o]
