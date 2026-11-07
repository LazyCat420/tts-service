"""Objective measurement harness for the synth engines.

Five metrics (plan §7) plus the repo's own regression guards. Everything is
implemented from the published definitions in numpy; nothing is vendored.

    python3 measure.py --engine worldish --text "The quick brown fox jumps over the lazy dog."

Reference-dependent metrics (MCD, F0 RMSE, rate/nPVI) take an optional human
recording of the same sentence; without one they are reported as n/a and the
absolute metrics still run.

Metrics
-------
1. CER      : character error rate from an ASR oracle (optional; needs
              faster-whisper or openai-whisper installed).
2. MCD      : DTW-aligned mel-cepstral distortion vs a reference, dB.
3. CPP      : cepstral peak prominence, dB (crispness / breathiness).
4. F0 RMSE  : semitone RMSE of the F0 track vs a reference contour, plus the
              P95-P5 span (the "monotone collapse" guard).
5. Rate/PVI : syllables per second and nPVI/rPVI of vocalic intervals.

Guards (always): harmonicity, zero-crossings/s, sample steps over 25% of peak,
energy above 5 kHz, spectral flatness on voiced frames, HNR.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

# ── primitives ──────────────────────────────────────────────────────────────


def _frames(x: np.ndarray, win: int, hop: int) -> np.ndarray:
    if len(x) < win:
        x = np.pad(x, (0, win - len(x)))
    n = 1 + (len(x) - win) // hop
    idx = np.arange(win)[None, :] + hop * np.arange(n)[:, None]
    return x[idx]


def _hann(n: int) -> np.ndarray:
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n) / n)


def _norm_ac_peak(seg: np.ndarray, lo: int, hi: int) -> float:
    """Boersma (1993) unbiased lag correlation peak over lags [lo, hi)."""
    seg = seg - np.mean(seg)
    n = len(seg)
    hi = min(hi, n - 2)
    if hi <= lo:
        return 0.0
    ac = np.correlate(seg, seg, "full")[n - 1:]
    c2 = np.cumsum(seg * seg)
    ks = np.arange(lo, hi)
    e1 = c2[n - 1 - ks]
    e2 = c2[n - 1] - c2[ks - 1]
    denom = np.sqrt(np.maximum(e1 * e2, 1e-24))
    return float(np.max(ac[ks] / denom))


def f0_track(x: np.ndarray, sr: int, fmin: float = 60.0, fmax: float = 400.0,
             win_ms: float = 40.0, hop_ms: float = 10.0) -> np.ndarray:
    """YIN pitch track (de Cheveigne & Kawahara 2002, fixed window W); 0.0 where unvoiced."""
    win = int(sr * win_ms / 1000)
    hop = int(sr * hop_ms / 1000)
    fr = _frames(np.asarray(x, float), win, hop)
    tau_min = max(2, int(sr / fmax))
    tau_max = min(win // 2, int(sr / fmin))
    w_len = win - tau_max
    out = np.zeros(fr.shape[0])
    taus = np.arange(tau_max + 1)
    tau_idx = np.arange(1, tau_max + 1)
    for i, f in enumerate(fr):
        f = f - np.mean(f)
        if np.sqrt(np.mean(f ** 2)) < 0.01:
            continue
        a = f[:w_len]
        ea = float(np.dot(a, a))
        c2 = np.cumsum(f * f)
        eb = c2[taus + w_len - 1] - np.where(taus > 0, c2[taus - 1], 0.0)
        corr = np.correlate(f[:tau_max + w_len], a, mode="valid")[:tau_max + 1]
        d = np.maximum(ea + eb - 2.0 * corr, 0.0)
        d[0] = 0.0
        csum = np.cumsum(d[1:])
        cmnd = np.ones(tau_max + 1)
        cmnd[1:] = d[1:] * tau_idx / np.maximum(csum, 1e-12)
        cand = np.where(cmnd[tau_min:tau_max + 1] < 0.20)[0]
        if cand.size:
            tau = tau_min + int(cand[0])
            while tau + 1 <= tau_max and cmnd[tau + 1] < cmnd[tau]:
                tau += 1
            if 1 <= tau < tau_max:
                a0, b0, c0 = cmnd[tau - 1], cmnd[tau], cmnd[tau + 1]
                den = a0 - 2 * b0 + c0
                shift = 0.5 * (a0 - c0) / den if abs(den) > 1e-12 else 0.0
                out[i] = sr / (tau + shift)
    return out


def _mel_filterbank(sr: int, n_fft: int, n_bands: int = 40,
                    fmin: float = 50.0, fmax: float | None = None) -> np.ndarray:
    fmax = fmax or sr / 2.0
    def hz2mel(f):
        return 2595.0 * np.log10(1.0 + f / 700.0)
    def mel2hz(m):
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)
    mels = np.linspace(hz2mel(fmin), hz2mel(fmax), n_bands + 2)
    hz = mel2hz(mels)
    bins = np.fft.rfftfreq(n_fft, 1.0 / sr)
    fb = np.zeros((n_bands, len(bins)))
    for k in range(n_bands):
        lo, ce, hi = hz[k], hz[k + 1], hz[k + 2]
        up = (bins - lo) / max(ce - lo, 1e-9)
        down = (hi - bins) / max(hi - ce, 1e-9)
        fb[k] = np.clip(np.minimum(up, down), 0.0, None)
    return fb


def _dct_matrix(n: int, k: int) -> np.ndarray:
    m = np.arange(k)[:, None]
    j = np.arange(n)[None, :]
    return np.cos(np.pi * m * (2 * j + 1) / (2 * n)) * np.sqrt(2.0 / n)


def mel_cepstra(x: np.ndarray, sr: int, n_fft: int = 1024, hop: int = 256,
                n_bands: int = 40, order: int = 24) -> np.ndarray:
    fr = _frames(np.asarray(x, float), n_fft, hop) * _hann(n_fft)
    mag = np.abs(np.fft.rfft(fr, axis=1)) ** 2
    fb = _mel_filterbank(sr, n_fft, n_bands)
    mel = np.log(mag @ fb.T + 1e-12)
    return mel @ _dct_matrix(n_bands, order).T


def _dtw(a: np.ndarray, b: np.ndarray, band: int = 15) -> list[tuple[int, int]]:
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return []
    band_eff = max(band, abs(n - m) + 5)
    cost = np.full((n + 1, m + 1), np.inf)
    cost[0, 0] = 0.0
    dist = np.sqrt(np.sum((a[:, None, :] - b[None, :, :]) ** 2, axis=2))
    for i in range(1, n + 1):
        jc = int(round(i * m / max(n, 1)))
        lo = max(1, jc - band_eff)
        hi = min(m, jc + band_eff)
        for j in range(lo, hi + 1):
            cost[i, j] = dist[i - 1, j - 1] + min(cost[i - 1, j],
                                                 cost[i, j - 1],
                                                 cost[i - 1, j - 1])
    path = []
    i, j = n, m
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        step = np.argmin([cost[i - 1, j - 1], cost[i - 1, j], cost[i, j - 1]])
        if step == 0:
            i, j = i - 1, j - 1
        elif step == 1:
            i -= 1
        else:
            j -= 1
    return path[::-1]


def mcd(x: np.ndarray, y: np.ndarray, sr: int, order: int = 24,
        sr_y: int | None = None) -> float:
    """DTW-aligned mel-cepstral distortion, dB (c0 excluded)."""
    a = mel_cepstra(x, sr, order=order)[:, 1:]
    b = mel_cepstra(y, sr_y or sr, order=order)[:, 1:]
    path = _dtw(a, b)
    if not path:
        return 0.0
    d = np.array([np.sum((a[i] - b[j]) ** 2) for i, j in path])
    return float(10.0 / np.log(10.0) * np.sqrt(2.0 * np.mean(d)))


def cpp(x: np.ndarray, sr: int, win_ms: float = 40.0, hop_ms: float = 10.0,
        n_fft: int = 4096) -> float:
    """Cepstral peak prominence, dB, averaged over voiced frames."""
    win = int(sr * win_ms / 1000)
    hop = int(sr * hop_ms / 1000)
    raw = _frames(np.asarray(x, float), win, hop)
    fr = raw * _hann(win)
    if fr.shape[0] == 0:
        return 0.0
    lo, hi = int(sr / 400), int(sr / 60)
    vals_voiced, vals_all = [], []
    for r_f, f in zip(raw, fr):
        if np.sqrt(np.mean(f ** 2)) < 0.01:
            continue
        # Hillenbrand 1994: dB power spectrum -> dB cepstrum
        logmag = 20.0 * np.log10(np.abs(np.fft.rfft(f, n_fft)) + 1e-12)
        cep = 20.0 * np.log10(np.abs(np.fft.irfft(logmag)) + 1e-12)
        band = np.arange(lo, min(hi, len(cep)))
        if band.size < 3:
            continue
        # linear regression through the cepstrum over 1-15 ms quefrency
        fit = np.arange(int(sr * 0.001), min(int(sr * 0.015), len(cep)))
        if fit.size < 2:
            continue
        coef = np.polyfit(fit, cep[fit], 1)
        line = np.polyval(coef, band)
        prom = float(np.max(cep[band] - line))
        vals_all.append(prom)
        if _norm_ac_peak(r_f, lo, hi) >= 0.45:
            vals_voiced.append(prom)
    vals = vals_voiced if vals_voiced else vals_all
    return float(np.mean(vals)) if vals else 0.0


def hnr(x: np.ndarray, sr: int) -> float:
    """Boersma/Yumoto HNR from unbiased lag correlation of voiced frames, dB."""
    fr = _frames(np.asarray(x, float), 1024, 512)
    lo, hi = int(sr / 400), int(sr / 60)
    vals = []
    for f in fr:
        if np.sqrt(np.mean(f ** 2)) < 0.01:
            continue
        r = _norm_ac_peak(f, lo, hi)
        if r < 0.50:                         # unvoiced / consonant frame
            continue
        r = min(max(r, 1e-4), 0.9999)
        vals.append(10.0 * np.log10(r / (1.0 - r)))
    return float(np.mean(vals)) if vals else 0.0


def flatness(x: np.ndarray, sr: int) -> float:
    raw = _frames(np.asarray(x, float), 1024, 512)
    if raw.shape[0] == 0:
        return 0.0
    lo, hi = int(sr / 400), int(sr / 60)
    fr = raw * _hann(1024)
    mag = np.abs(np.fft.rfft(fr, axis=1)) + 1e-12
    sf = np.exp(np.mean(np.log(mag), axis=1)) / np.mean(mag, axis=1)
    voiced = [i for i, r_f in enumerate(raw)
              if np.sqrt(np.mean(r_f ** 2)) >= 0.01 and _norm_ac_peak(r_f, lo, hi) >= 0.45]
    return float(np.mean(sf[voiced] if voiced else sf))


def guards(x: np.ndarray, sr: int) -> dict:
    y = np.asarray(x, float)
    peak = max(1e-9, float(np.max(np.abs(y))))
    sp = np.abs(np.fft.rfft(y))
    f = np.fft.rfftfreq(len(y), 1.0 / sr)
    hi = float(np.sum(sp[f > 5000] ** 2) / max(np.sum(sp ** 2), 1e-12))
    # Median unbiased lag correlation across active 40 ms windows (60-400 Hz).
    w, hop = int(0.04 * sr), int(0.02 * sr)
    lo, hi_i = int(sr / 400), int(sr / 60)
    harms, rms = [], []
    for i in range(0, max(1, len(y) - w + 1), hop):
        seg = y[i:i + w]
        if len(seg) < hi_i + 2:
            continue
        rms.append(float(np.sqrt(np.mean(seg ** 2))))
        harms.append(_norm_ac_peak(seg, lo, hi_i))
    if harms:
        rms_a, harms_a = np.asarray(rms), np.asarray(harms)
        keep = rms_a > 0.3 * float(np.median(rms_a))
        harm = float(np.median(harms_a[keep] if keep.any() else harms_a))
    else:
        harm = 0.0
    dj = np.abs(np.diff(y))
    return {
        "peak": round(peak, 3),
        "steps25": int(np.sum(dj > 0.25 * peak)),
        "jump_pct": round(100.0 * float(np.mean(dj > 0.25 * peak)), 2),
        "max_jump_ratio": round(float(np.max(dj)) / peak, 3),
        # true click: a spike that reverses direction immediately (1-sample
        # discontinuity). A glottal pulse rises over several samples and is
        # not counted.
        "crackle": int(np.sum((dj[1:] > 0.25 * peak) & (np.abs(np.diff(y, 2)) > 0.5 * peak))),
        "hi5k_pct": round(hi * 100, 2),
        "harm": round(harm, 3),
        "zcr": round(float(np.mean(np.abs(np.diff(np.sign(y))) > 0)) * sr / 2, 0),
        "flat": round(flatness(y, sr), 3),
        "hnr_db": round(hnr(y, sr), 1),
        "cpp_db": round(cpp(y, sr), 1),
        "dur_s": round(len(y) / sr, 2),
    }


def f0_stats(track: np.ndarray, hop_s: float = 0.01) -> dict:
    v = track[track > 0]
    if v.size < 2:
        return {"f0_med": 0.0, "span_st": 0.0, "voiced_pct": 0.0}
    st = 12.0 * np.log2(v / 125.0)
    return {
        "f0_med": round(float(np.median(v)), 1),
        "span_st": round(float(np.percentile(st, 95) - np.percentile(st, 5)), 2),
        "voiced_pct": round(100.0 * v.size / track.size, 1),
    }


def f0_rmse_st(track_a: np.ndarray, track_b: np.ndarray) -> float:
    a = np.asarray(track_a, float)
    b = np.asarray(track_b, float)
    if a.size < 2 or b.size < 2:
        return float("nan")
    if a.size != b.size:
        idx = np.round(np.linspace(0, b.size - 1, a.size)).astype(int)
        b = b[idx]
    both = (a > 0) & (b > 0)
    if both.sum() < 2:
        return float("nan")
    return float(np.sqrt(np.mean((12 * np.log2(a[both] / b[both])) ** 2)))


def pvi(durations: np.ndarray, normalised: bool = True) -> float:
    if len(durations) < 2:
        return 0.0
    d = np.asarray(durations, float)
    if normalised:
        return float(100.0 * np.mean(np.abs(d[1:] - d[:-1]) / ((d[1:] + d[:-1]) / 2)))
    return float(np.mean(np.abs(d[1:] - d[:-1])))


def nuclei_durations(x: np.ndarray, sr: int, hop_ms: float = 10.0) -> np.ndarray:
    """Vocalic intervals from an energy envelope (syllable nuclei)."""
    hop = int(sr * hop_ms / 1000)
    env = np.sqrt(np.convolve(x ** 2, np.ones(hop) / hop, mode="same"))[::hop]
    if env.size < 4:
        return np.array([])
    thr = 0.35 * np.max(env)
    on = env > thr
    durs, run = [], 0
    for v in on:
        if v:
            run += 1
        elif run:
            durs.append(run * hop_ms / 1000.0)
            run = 0
    if run:
        durs.append(run * hop_ms / 1000.0)
    return np.array(durs)


def cer(y: np.ndarray, sr: int, text: str) -> float | None:
    """Character error rate via an ASR oracle, if one is installed."""
    try:
        from faster_whisper import WhisperModel          # noqa: F401
    except Exception:
        try:
            import whisper                              # noqa: F401
        except Exception:
            return None
    tmp = os.path.join("/tmp", "vs_asr.wav")
    _write_wav(tmp, y, sr)
    try:
        try:
            from faster_whisper import WhisperModel
            model = WhisperModel("tiny.en", device="cpu", compute_type="int8")
            segs, _ = model.transcribe(tmp, language="en")
            hyp = " ".join(s.text for s in segs)
        except Exception:
            import whisper
            model = whisper.load_model("tiny.en")
            hyp = model.transcribe(tmp)["text"]
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass
    return _levenshtein_cer(text, hyp)


def _norm(s: str) -> str:
    import re
    return re.sub(r"[^a-z0-9 ]+", "", s.lower()).strip()


def _levenshtein_cer(ref: str, hyp: str) -> float:
    a, b = _norm(ref), _norm(hyp)
    if not a:
        return 0.0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1] / len(a)


def _write_wav(path: str, samples: np.ndarray, sr: int) -> None:
    import wave
    y = np.asarray(samples, float)
    y = np.clip(y / max(1e-9, np.max(np.abs(y))) * 0.95, -1, 1)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((y * 32767).astype("<i2").tobytes())


# ── CLI ────────────────────────────────────────────────────────────────────


def _render(engine: str, text: str, voice: dict, voice_name: str | None = None):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from app.synth.reciter import reciter
    from app.synth import frames as F
    if engine == "sam":
        import sam
        return np.array(sam.render(text)), sam.SR, None
    if engine == "synth":
        from app import synth_manager
        from app.synth import presets
        name = voice_name or "narrator"
        v = presets.voice(name)
        fr = F.build(reciter(text), v)
        y = synth_manager.render(text, name)
        return np.asarray(y), synth_manager.SAMPLE_RATE, fr
    phones = reciter(text)
    fr = F.build(phones, voice)
    if engine == "klatt":
        from app.synth import klatt
        return np.array(klatt.render(fr)), 22050, fr
    from app.synth import worldish
    return np.array(worldish.render(fr)), 24000, fr


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="worldish",
                    choices=["klatt", "worldish", "sam", "synth"])
    ap.add_argument("--voice", default="narrator",
                    help="voice type for --engine synth")
    ap.add_argument("--text", default="The quick brown fox jumps over the lazy dog.")
    ap.add_argument("--ref", default=None, help="human reference WAV, same text")
    ap.add_argument("--pitch", type=int, default=64)
    ap.add_argument("--speed", type=int, default=72)
    ap.add_argument("--save", default=None)
    args = ap.parse_args(argv)

    voice = {"pitch": args.pitch, "speed": args.speed}
    y, sr, fr = _render(args.engine, args.text, voice, args.voice)
    if args.save:
        _write_wav(args.save, y, sr)

    row = guards(y, sr)
    track = f0_track(y, sr)
    row.update(f0_stats(track))
    durs = nuclei_durations(y, sr)
    row["rate_syl_s"] = round(len(durs) / max(row["dur_s"], 1e-6), 2)
    row["npvi"] = round(pvi(durs), 1)
    row["rpvi"] = round(pvi(durs, normalised=False), 3)
    row["cer"] = cer(y, sr, args.text)
    if fr is not None:
        row["f0_rmse_vs_plan_st"] = round(f0_rmse_st(track, fr.f0), 2)

    if args.ref:
        import wave
        with wave.open(args.ref, "rb") as w:
            sr_ref = w.getframerate()
            ref = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(float) / 32768.0
        row["mcd_db"] = round(mcd(y, ref, sr, sr_y=sr_ref), 2)
        row["f0_rmse_vs_ref_st"] = round(f0_rmse_st(track, f0_track(ref, sr_ref)), 2)

    print(f"engine={args.engine} sr={sr} text={args.text!r}")
    for k, v in row.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
