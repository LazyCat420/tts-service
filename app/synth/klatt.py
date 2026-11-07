"""Parallel 5-formant Klatt synthesizer with period-synchronous glottal tilt source (numpy + stdlib)."""
import math

import numpy as np

from . import frames as _frames

_FRIC_F = (500.0, 1000.0, 1800.0, 2800.0, 3800.0, 4900.0)
_FRIC_BW = (400.0, 500.0, 600.0, 800.0, 1000.0, 1000.0)
_HOP = 16  # samples between coefficient updates
TILT_HZ = 400.0  # corner of the two-pole source tilt
ASP_SCALE = 0.12  # noise level at ah=1 relative to the voicing source


def _res_coef(f, bw, sr):
    r = np.exp(-math.pi * np.asarray(bw, dtype=float) / sr)
    c = -r * r * np.ones_like(np.asarray(f, dtype=float))
    b = 2.0 * r * np.cos(2.0 * math.pi * np.asarray(f, dtype=float) / sr)
    return 1.0 - b - c, b, c


def _res_complex(hz, formant, bw, sr):
    hz = np.asarray(hz, dtype=float)
    formant = np.asarray(formant, dtype=float)
    bw = np.maximum(np.asarray(bw, dtype=float), 1.0)
    theta = 2.0 * np.pi * formant / sr
    r = np.exp(-np.pi * bw / sr)
    a = 1.0 - 2.0 * r * np.cos(theta) + r * r
    b = 2.0 * r * np.cos(theta)
    c = -r * r
    w = 2.0 * np.pi * hz / sr
    z1 = np.exp(-1j * w)
    return a / (1.0 - b * z1 - c * z1 * z1)


def _biquad(kind, f, sr, q=0.7071):
    w = 2.0 * math.pi * f / sr
    c, s = math.cos(w), math.sin(w)
    al = s / (2.0 * q)
    a0 = 1.0 + al
    if kind == "lp":
        b0 = (1.0 - c) / 2.0 / a0
        b1 = (1.0 - c) / a0
    else:
        b0 = (1.0 + c) / 2.0 / a0
        b1 = -(1.0 + c) / a0
    return b0, b1, b0, -2.0 * c / a0, (1.0 - al) / a0


def _biquad_mag(co, hz, sr):
    b0, b1, b2, a1, a2 = co
    w = 2.0 * np.pi * np.asarray(hz, dtype=float) / sr
    z1 = np.exp(-1j * w)
    z2 = z1 * z1
    return np.abs((b0 + b1 * z1 + b2 * z2) / (1.0 + a1 * z1 + a2 * z2))


def _biquad_run(x, co):
    b0, b1, b2, a1, a2 = co
    x1 = x2 = y1 = y2 = 0.0
    out = []
    ap = out.append
    for v in x:
        y = b0 * v + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
        x2 = x1
        x1 = v
        y2 = y1
        y1 = y
        ap(y)
    return out


def _branch_gains(form, bws, f0b, oqb, tiltb, amps, sr):
    """Per-hop input gains for the 5 parallel branches.

    Evaluates each branch at its formant's nearest harmonic of F0, including
    the glottal two-pole tilt, radiation first-difference, output HP/LP
    biquads, voice spectral tilt (`oq` and `tilt_db` relative to nominal
    0.50 / 6.0 dB), and two steps of complex inter-branch skirt decoupling so
    steady vowel harmonic peaks land on `amps` (`FORMANT_AMP`).
    """
    f0b = np.asarray(f0b, dtype=float)[:, None]
    hz = np.where(f0b > 0.0, np.round(form / np.maximum(f0b, 1e-9)) * f0b, form)
    hz = np.maximum(hz, 1.0)
    w = 2.0 * np.pi * hz / sr
    d = math.exp(-2.0 * math.pi * TILT_HZ / sr)
    tilt = (1.0 - d) ** 2 / np.abs(1.0 - d * np.exp(-1j * w)) ** 2
    rad = np.abs(1.0 - np.exp(-1j * w))
    hp_m = _biquad_mag(_biquad("hp", 80.0, sr), hz, sr)
    lp_m = _biquad_mag(_biquad("lp", min(4800.0, 0.45 * sr), sr), hz, sr)
    src = tilt * rad * hp_m * lp_m

    # Voice-specific spectral tilt from open quotient (oq) and tilt_db
    # relative to the nominal reference (oq = 0.50, tilt_db = 6.0).
    delta_tilt = (np.asarray(tiltb, dtype=float) - 6.0) + 12.0 * (
        np.asarray(oqb, dtype=float) - 0.50
    )
    tilt_mult = (form / np.maximum(form[:, :1], 1.0)) ** (-delta_tilt[:, None] / 18.0)
    tgt = amps[None, :] * tilt_mult

    res_diag = _frames.resonator_mag(hz, form, bws, sr)
    lvl = res_diag * src
    g = tgt * (lvl[:, :1] / np.maximum(lvl, 1e-12))

    # Complex skirt decoupling across the 5 parallel resonators:
    H = _res_complex(hz[:, :, None], form[:, None, :], bws[:, None, :], sr) * src[:, :, None]
    for _ in range(2):
        tot = np.abs(np.sum(H * g[:, None, :], axis=2))
        rel_tot = tot / np.maximum(tot[:, :1], 1e-12)
        rel_tgt = tgt / np.maximum(tgt[:, :1], 1e-12)
        corr = np.clip(rel_tgt / np.maximum(rel_tot, 1e-6), 0.5, 2.0)
        g = g * corr
    return g


def _noise(n, rng):
    u = rng.uniform(-1.0, 1.0, n).tolist()
    out = []
    ap = out.append
    p = 0.0
    for v in u:
        p = v + 0.75 * p
        ap(p)
    return np.asarray(out)


def render(frames, sr=22050, rng=None, source="klglott88"):
    if source != "klglott88":
        raise ValueError("unsupported source: %s" % source)
    if rng is None:
        rng = np.random.default_rng()
    n = int(len(frames.f0) * sr / _frames.FRAME_HZ)
    if n <= 0:
        return np.zeros(0)
    fr = _frames.interpolate(frames, n)

    f0a = np.asarray(fr.f0, dtype=float)
    voiced = f0a > 0.0
    av_a = np.where(voiced, np.asarray(fr.av, dtype=float), 0.0)
    ah_a = np.asarray(fr.ah, dtype=float)

    f0l = f0a.tolist()
    avl = av_a.tolist()
    noise = _noise(n, rng)
    noisel = noise.tolist()

    # ---- excitation: period-synchronous impulse train + spectral tilt ----
    # The glottal source is modelled as a flat impulse train shaped by two
    # one-pole tilt filters at TILT_HZ (-12 dB/oct above the corner), which the
    # radiation stage at the end partly cancels (+6 dB/oct). A flat source is
    # what lets FORMANT_AMP alone decide the balance: the KLGLOTT88 pulse shape
    # carries a 20 dB low-frequency bump (measured) that drowned F2..F5 under
    # the F1 resonance, whatever the branch gains were.
    bus = [0.0] * n
    # Sub-sample fractional Blackman-windowed-sinc glottal impulse train on a
    # continuous phase accumulator (sam-demo/methods.py `phase += f0 / sr`).
    # The old loop deposited a delta at integer samples, quantising every
    # pitch period to a whole number of samples; on steady vowels that
    # sample-grid quantisation reads as pitch grain. The 8-tap Blackman
    # windowed sinc is deposited at the exact fractional closure instant and
    # is normalised to unity sum, so its gain is flat across the speech band
    # (0..4800 Hz) and the < 1.0 dB formant-peak calibration on /AA/ is
    # untouched. The 3-sine vibrato flutter is gone and jitter/shimmer are
    # 10x smaller, so periods stay phase-coherent instead of wobbly.
    imp_buf = [0.0] * (n + 8)
    phase = 0.0
    dt = 0.0
    ar_j = ar_s = 0.0
    k_ar = math.sqrt(1.0 - 0.99 ** 2)
    t1 = t2 = 0.0
    tilt_d = math.exp(-2.0 * math.pi * TILT_HZ / sr)
    tilt_g = 1.0 - tilt_d
    for i in range(n):
        f0 = f0l[i]
        if f0 > 0.0:
            fe = max(f0, 20.0) * (1.0 + 0.0004 * ar_j)
            dt = fe / sr
            phase += dt
            if phase >= 1.0:
                phase -= 1.0
                ar_j = 0.99 * ar_j + k_ar * float(rng.standard_normal())
                ar_s = 0.99 * ar_s + k_ar * float(rng.standard_normal())
                amp = avl[i] * (1.0 + 0.001 * ar_s)
                mu = min(max(phase / max(dt, 1e-9), 0.0), 1.0)
                taps = [0.0] * 8
                tsum = 0.0
                for m in range(-3, 5):
                    x = m + mu
                    if abs(x) < 1e-9:
                        sc = 1.0
                    else:
                        px = math.pi * x
                        sc = math.sin(px) / px
                    u = (x + 3.5) / 7.0
                    if u < 0.0:
                        u = 0.0
                    elif u > 1.0:
                        u = 1.0
                    w = (0.42 - 0.5 * math.cos(2.0 * math.pi * u)
                         + 0.08 * math.cos(4.0 * math.pi * u))
                    taps[m + 3] = sc * w
                    tsum += taps[m + 3]
                if tsum != 0.0:
                    scale = amp / tsum
                    for m in range(-3, 5):
                        j = i + m
                        if 0 <= j < n + 8:
                            imp_buf[j] += taps[m + 3] * scale
        t1 = tilt_g * imp_buf[i] + tilt_d * t1
        t2 = tilt_g * t1 + tilt_d * t2
        bus[i] = t2

    # ---- aspiration: noise tilted like the glottal source and scaled to it ----
    # The parallel branch gains undo the two-pole 400 Hz tilt of `bus` (+30 dB
    # at F4/F5), so aspiration entering the parallel bank must carry the same
    # two-pole tilt or F4/F5 boost it 30-50x. When a clip has no voiced frames
    # at all (e.g. the `whisper` voice, where av=0 everywhere), fall back to
    # the nominal RMS of a unit-av 110 Hz pulse train (~0.015) so ah is not
    # multiplied by zero.
    bus = np.asarray(bus)
    asp_n = [0.0] * n
    a1_s = a2_s = 0.0
    for i in range(n):
        a1_s = tilt_g * noisel[i] + tilt_d * a1_s
        a2_s = tilt_g * a1_s + tilt_d * a2_s
        asp_n[i] = a2_s
    asp_noise = np.asarray(asp_n)
    noise_rms = float(np.sqrt(np.mean(asp_noise ** 2))) or 1.0
    voiced_bus = bus[voiced]
    voiced_rms = float(np.sqrt(np.mean(voiced_bus ** 2))) if voiced_bus.size else 0.045
    ah = np.asarray(fr.ah, dtype=float) * (voiced_rms / noise_rms) * ASP_SCALE
    bus = bus + ah * asp_noise

    # ---- parallel formant bank: five branches, explicit amplitudes ----
    # A five-stage cascade multiplies every stage's out-of-band attenuation:
    # measured on /AA/ it put F3 36 dB and F4 45 dB below F1, where natural
    # speech is about 12 and 20 dB down. Klatt's parallel synthesiser avoids
    # that by giving each formant its own resonator and its own amplitude (and
    # is the more intelligible of his two configurations), so all five branches
    # run in parallel here. The branch input gains undo each resonator's peak
    # gain and the glottal source's rolloff, so the output peaks land on
    # FORMANT_AMP and the source spectrum cannot tilt the balance.
    idx = np.arange(0, n, _HOP)
    raw_form = np.asarray(fr.formants, dtype=float)[idx]
    default_f = np.array([500.0, 1500.0, 2500.0, 3500.0, 4500.0])
    form = np.where(raw_form > 80.0, raw_form, default_f[None, :])
    bws = np.maximum(np.asarray(fr.bws, dtype=float)[idx], 40.0)
    fnp = np.asarray(fr.fnp, dtype=float)[idx]
    fnz = np.asarray(fr.fnz, dtype=float)[idx]
    nas = (fnp > 80.0) & (fnz > 80.0)

    coefs = [_res_coef(form[:, k], bws[:, k], sr) for k in range(5)]
    amps = _frames.peak_amplitudes(fr)
    raw_gain = _branch_gains(
        form,
        bws,
        np.asarray(fr.f0, dtype=float)[idx],
        np.clip(np.asarray(fr.oq, dtype=float)[idx], 0.1, 0.95),
        np.asarray(fr.tilt_db, dtype=float)[idx],
        amps,
        sr,
    )
    # Smooth branch gains across 16-sample hops (~5 ms time constant) so
    # half-harmonic crossings in np.round(form/f0) glide instead of stepping,
    # while steady vowels keep their exact constant gain.
    gain = np.empty_like(raw_gain)
    if raw_gain.shape[0] > 0:
        g_cur = raw_gain[0].copy()
        for h_i in range(raw_gain.shape[0]):
            g_cur = 0.90 * g_cur + 0.10 * raw_gain[h_i]
            gain[h_i] = g_cur

    # Keep the nasal pole-zero pair running on every sample: when non-nasal,
    # fnp_eff == fnz_eff == 270 Hz so the pole and zero cancel to an exact
    # identity filter while keeping zx1, zx2, n1, n2 warm (no switching spike).
    fnp_eff = np.where(nas, np.maximum(fnp, 200.0), 270.0)
    fnz_eff = np.where(nas, np.maximum(fnz, 200.0), 270.0)
    pa, pb, pc = _res_coef(fnp_eff, 100.0, sr)
    za, zb, zc = _res_coef(fnz_eff, 100.0, sr)
    # Balance DC vs high-frequency asymptote via geometric mean sqrt(pa/za)
    # so nasals neither boost low frequencies by +14 dB nor attenuate F2..F5
    # by -9 dB, while fnp_eff == fnz_eff still reduces to exact unity gain.
    g_nz = np.sqrt(pa / np.maximum(za, 1e-12))
    za2 = g_nz.tolist()
    zb2 = (-zb * g_nz).tolist()
    zc2 = (-zc * g_nz).tolist()
    pa = np.ones_like(pa).tolist()
    pb, pc = pb.tolist(), pc.tolist()
    (a1, b1, c1), (a2, b2, c2), (a3, b3, c3), (a4, b4, c4), (a5, b5, c5) = coefs
    a1, b1, c1 = a1.tolist(), b1.tolist(), c1.tolist()
    a2, b2, c2 = a2.tolist(), b2.tolist(), c2.tolist()
    a3, b3, c3 = a3.tolist(), b3.tolist(), c3.tolist()
    a4, b4, c4 = a4.tolist(), b4.tolist(), c4.tolist()
    a5, b5, c5 = a5.tolist(), b5.tolist(), c5.tolist()
    g1 = gain[:, 0].tolist()
    g2 = gain[:, 1].tolist()
    g3 = gain[:, 2].tolist()
    g4 = gain[:, 3].tolist()
    g5 = gain[:, 4].tolist()

    bank = [0.0] * n
    zx1 = zx2 = 0.0
    n1 = n2 = 0.0
    y1a = y1b = y2a = y2b = y3a = y3b = y4a = y4b = y5a = y5b = 0.0
    for h, s0 in enumerate(range(0, n, _HOP)):
        e = min(s0 + _HOP, n)
        A1, B1, C1 = a1[h], b1[h], c1[h]
        A2, B2, C2 = a2[h], b2[h], c2[h]
        A3, B3, C3 = a3[h], b3[h], c3[h]
        A4, B4, C4 = a4[h], b4[h], c4[h]
        A5, B5, C5 = a5[h], b5[h], c5[h]
        G1, G2, G3, G4, G5 = g1[h], g2[h], g3[h], g4[h], g5[h]
        ZA, ZB, ZC = za2[h], zb2[h], zc2[h]
        PA, PB, PC = pa[h], pb[h], pc[h]
        for i in range(s0, e):
            x = bus[i]
            y = G1 * A1 * x + B1 * y1a + C1 * y1b
            y1b = y1a
            y1a = y
            p = G2 * A2 * x + B2 * y2a + C2 * y2b
            y2b = y2a
            y2a = p
            y += p
            p = G3 * A3 * x + B3 * y3a + C3 * y3b
            y3b = y3a
            y3a = p
            y += p
            p = G4 * A4 * x + B4 * y4a + C4 * y4b
            y4b = y4a
            y4a = p
            y += p
            p = G5 * A5 * x + B5 * y5a + C5 * y5b
            y5b = y5a
            y5a = p
            y += p
            z = ZA * y + ZB * zx1 + ZC * zx2
            zx2 = zx1
            zx1 = y
            y = PA * z + PB * n1 + PC * n2
            n2 = n1
            n1 = y
            bank[i] = y

    bank = np.asarray(bank)
    out = np.empty(n)
    out[0] = bank[0]
    out[1:] = bank[1:] - bank[:-1]  # radiation: first difference

    # ---- parallel frication branch ----
    fric = np.asarray(fr.fric, dtype=float)
    af = np.asarray(fr.af, dtype=float)
    byp = np.asarray(fr.bypass, dtype=float)
    if np.any(af > 0.0) and (np.any(fric > 0.0) or np.any(byp > 0.0)):
        fsum = np.zeros(n)
        for k in range(6):
            g = fric[:, k] * af
            if not np.any(g > 0.0):
                continue
            a, b, c = (float(q) for q in _res_coef(_FRIC_F[k], _FRIC_BW[k], sr))
            # divide out the band's own peak gain, exactly as the voiced bank
            # does, so af=1 means "this band peaks at the voicing level"
            pk = float(_frames.resonator_mag(_FRIC_F[k], _FRIC_F[k], _FRIC_BW[k], sr))
            ya = yb = 0.0
            r = []
            ap = r.append
            for x in noisel:
                y = a * x + b * ya + c * yb
                yb = ya
                ya = y
                ap(y)
            r = np.asarray(r) * g / pk
            fsum += r if k < 1 else (r if (k + 1) % 2 else -r)
        bp = noise * byp * af * 0.5  # the first difference below adds ~6 dB
        bp = np.concatenate(([bp[0]], np.diff(bp)))
        fric_tot = fsum + bp
        tract_pk = float(np.max(np.abs(out)))
        fric_pk = float(np.max(np.abs(fric_tot)))
        if tract_pk > 1e-6 and fric_pk > 0.30 * tract_pk:
            fric_tot = fric_tot * (0.30 * tract_pk / fric_pk)
        out += fric_tot

    # ---- band-limit and normalise ----
    y = _biquad_run(out.tolist(), _biquad("hp", 80.0, sr))
    y = np.asarray(_biquad_run(y, _biquad("lp", min(4800.0, 0.45 * sr), sr)))
    pk = float(np.max(np.abs(y)))
    if pk > 0.0:
        y *= 0.85 / pk
    return y
