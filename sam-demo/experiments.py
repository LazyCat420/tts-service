"""How speech is made from waves: nine renderings of one sentence.

    python3 experiments.py

Builds `audio/x-*.wav` + `spec/x-*.png`, then `experiments.html`. The point is
to hear the same phoneme stream produced by every family of technique:

  cascade   saw -> F1 -> F2 -> F3, fixed bandwidths      (what sam.py did first)
  parallel  three resonators side by side, explicit gains (Klatt's parallel branch)
  additive  sum of harmonics shaped by the formant response
  pulse     parallel branch driven by a Rosenberg glottal pulse
  lpc       LPC-10 style: all-pole filter + pulse/noise excitation
  psola-len PSOLA time-stretch 1.6x (length changes, formants do not)
  psola-pit PSOLA pitch-shift 1.3x (pitch changes, formants do not)
  hnm       sinusoidal modelling + noise residual
  sam       the ported 1982 engine, for reference
"""

from __future__ import annotations

import html
import math
import os
import shutil
import subprocess

import hnm
import lpc
import methods
import psola
import sam
import sam_faithful

HERE = os.path.dirname(os.path.abspath(__file__))
AUDIO = os.path.join(HERE, "audio")
SPEC = os.path.join(HERE, "spec")

TEXT = "I am a little robot. I come from 1982."
KNOBS = dict(pitch=64, speed=72, throat=128, mouth=128)

# slug -> (title, explanation, formula)
METHODS = {
    "cascade": (
        "Cascade resonators",
        "A sawtooth through F1 → F2 → F3 in series. Each resonator is normalised to "
        "unity gain at DC, so the product peaks as (F/bw)³: push the formants up and "
        "F3 takes over and the voice screams. This is what the synth did first.",
        "y[n] = A·x[n] + B·y[n−1] + C·y[n−2]\n"
        "C = −e^(−2π·bw·T)   B = 2e^(−π·bw·T)·cos(2π·f·T)   A = 1 − B − C"),
    "parallel": (
        "Parallel resonators (default now)",
        "The same three resonators side by side, each with its own gain (1.0, 0.55, "
        "0.30), summed. The formants keep their intended balance at any knob setting, "
        "and the bandwidths now scale with frequency (6% + 50 Hz) instead of being "
        "fixed at 70/110/180 Hz.",
        "y[n] = Σ_k g_k · H_k(x[n])      H_k = 2-pole resonator at f_k, bw_k\n"
        "bw_k = 0.06·f_k + 50 Hz"),
    "additive": (
        "Additive harmonics",
        "No filters at all: sum every harmonic of f0, with each harmonic's amplitude "
        "read off the formant response at that frequency. Phases restart at every "
        "glottal pulse, so the wave is exactly periodic and does not acquire the "
        "hollow 'phasiness' of free-running oscillators.",
        "x(t) = Σ_(k=1..K) A(k·f0)·sin(2π·k·f0·t)\n"
        "A(f) = Σ_k g_k·f_k² / √((f_k² − f²)² + (f·bw_k)²)"),
    "pulse": (
        "Glottal pulse source",
        "The Rosenberg pulse instead of a sawtooth: 40% open phase, 16% return. Its "
        "spectrum rolls off faster than a saw's, which is the spectral tilt real voices "
        "have — less buzzy, more like a larynx.",
        "g(t) = ½(1 − cos(π·t/Tp))          for 0 ≤ t < Tp\n"
        "g(t) = cos(π·(t−Tp)/(2·Tn))        for Tp ≤ t < Tp+Tn\n"
        "Tp = 0.4·T, Tn = 0.16·T"),
    "lpc": (
        "LPC resynthesis",
        "Analyse the waveform into an all-pole filter (order 10) per 18 ms frame, then "
        "re-excite that filter with a pulse train plus noise. This is LPC-10, the 2.4 "
        "kbit/s vocoder: the formants survive because the filter carries them, the "
        "excitation does not.",
        "x[n] ≈ Σ_(k=1..p) a_k·x[n−k]        →      1/A(z) driven by pulses + noise"),
    "psola-len": (
        "PSOLA, length ×1.6",
        "Same waveform, 60% longer, same pitch and the same formants. Pitch marks are "
        "found by autocorrelation; windows of two periods are re-laid at scaled "
        "positions and overlap-added.",
        "y = Σ_m w(t − m·Ts')·x(t − m·Ts)      Ts' = Ts·α"),
    "psola-pit": (
        "PSOLA, pitch ×1.3",
        "Same length, f0 up 30%, formants unmoved — the thing naive resampling gets "
        "wrong (it moves formants by the same ratio, the chipmunk effect).",
        "f0' = f0·β,   formant peaks unchanged"),
    "hnm": (
        "Sinusoidal + noise (HNM)",
        "Track every spectral peak frame by frame, give each partial a continuous "
        "phase, and keep the high band as noise. This is how a vocoder models sound: "
        "partials for the voice, noise for the breath.",
        "x(t) = Σ_k A_k(t)·cos(φ_k(t)) + n(t)\n"
        "φ_k(t) = φ_k(t−1) + 2π·f_k(t)/sr"),
    "sam": (
        "The 1982 engine (reference)",
        "The ported SAM renderer: three direct oscillators, phase reset every glottal "
        "pulse, 4-bit output. What the target sound actually is.",
        "mux = (sin(F1)·A1 + sin(F2)·A2 + rect(F3)·A3)/32 + 128"),
}


def spectrogram(wav: str, png: str) -> bool:
    if not shutil.which("ffmpeg"):
        return False
    r = subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", wav,
         "-lavfi", "showspectrumpic=s=760x260:legend=1:scale=log:fscale=log:color=intensity",
         png], capture_output=True, text=True)
    return r.returncode == 0


def probe(samples, f0_hint: float) -> dict:
    x = [v / 32768.0 if isinstance(v, int) else float(v) for v in samples]
    seg = x[len(x) // 3: len(x) // 3 + 4096]
    mags = []
    for f in range(60, 10000, 20):
        re = sum(v * math.cos(2 * math.pi * f * i / sam.SR) for i, v in enumerate(seg))
        im = sum(v * math.sin(2 * math.pi * f * i / sam.SR) for i, v in enumerate(seg))
        mags.append((f, math.hypot(re, im)))
    tot = sum(v * v for _, v in mags) or 1e-9
    harm = sum(v * v for f, v in mags if min(abs(f - k * f0_hint) for k in range(1, 25)) < 40)
    zc = sum(1 for i in range(len(seg) - 1) if (seg[i] < 0) != (seg[i + 1] < 0)) / (len(seg) / sam.SR)
    band = sorted((v for f, v in mags if 200 <= f <= 3500), reverse=True)
    return {"dur": len(x) / sam.SR, "harm": harm / tot, "zcr": zc,
            "contrast": band[0] / max(band[len(band) // 2], 1e-9)}


def main() -> int:
    os.makedirs(AUDIO, exist_ok=True)
    os.makedirs(SPEC, exist_ok=True)
    phon = sam.reciter(TEXT)
    f0 = 20.0 + KNOBS["pitch"] * 1.15

    base = methods.render(phon, method="parallel", source="saw", **KNOBS)
    clips = {
        "cascade": methods.render(phon, method="cascade", source="saw",
                                  bw_mode="fixed", **KNOBS),
        "parallel": base,
        "additive": methods.render(phon, method="additive", source="saw", **KNOBS),
        "pulse": methods.render(phon, method="pulse", source="pulse", **KNOBS),
        "lpc": lpc.lpc_resynth(base),
        "psola-len": psola.td_psola(base, time_factor=1.6),
        "psola-pit": psola.td_psola(base, pitch_factor=1.3),
        "hnm": hnm.hnm_resynth(base),
        "sam": sam_faithful.faithful_samples(TEXT, KNOBS["pitch"], KNOBS["speed"],
                                             KNOBS["mouth"], KNOBS["throat"]),
    }

    # chipmunk baseline: what plain resampling does to the same sentence
    resampled = [base[int(i / 1.6)] for i in range(int(len(base) * 1.6))
                 if int(i / 1.6) < len(base)]
    clips["resample"] = resampled

    files, stats = {}, {}
    for slug, samples in clips.items():
        wav = os.path.join(AUDIO, f"x-{slug}.wav")
        sam.write_wav(wav, [v / 32768.0 if isinstance(v, int) else v for v in samples])
        png = os.path.join(SPEC, f"x-{slug}.png")
        spectrogram(wav, png)
        files[slug] = (wav, png)
        stats[slug] = probe(samples, f0)

    write_page(files, stats)
    import json
    with open(os.path.join(HERE, "experiments.json"), "w") as f:
        json.dump({k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in stats.items()},
                  f, indent=1)
    print(f"experiments: {len(files)} clips, "
          f"{sum(stats[s]['dur'] for s in stats):.1f}s total")
    return 0


def _rel(path):
    return os.path.relpath(path, HERE).replace(os.sep, "/")


def write_page(files, stats) -> None:
    order = ["cascade", "parallel", "pulse", "additive", "lpc",
             "psola-len", "psola-pit", "hnm", "resample", "sam"]
    titles = dict((k, v[0]) for k, v in METHODS.items())
    titles["resample"] = "Naive resampling (the wrong way)"
    cards = []
    for slug in order:
        wav, png = files[slug]
        if slug in METHODS:
            title, why, formula = METHODS[slug]
        else:
            title = titles[slug]
            why = ("Playing the same samples 1.6× faster lengthens nothing: it raises "
                   "pitch and formants together. Compare its spectrogram with psola-len.")
            formula = "y(t) = x(t/α)  →  every frequency ×α"
        s = stats[slug]
        cards.append(f"""    <section class="card">
      <h2>{html.escape(title)}</h2>
      <p class="says">{html.escape(why)}</p>
      <pre>{html.escape(formula)}</pre>
      <audio controls preload="none" src="{_rel(wav)}"></audio>
      <img class="spec" src="{_rel(png)}" alt="spectrogram">
      <p class="knobs">duration <b>{s['dur']:.2f} s</b> · harmonicity <b>{s['harm'] * 100:.1f}%</b>
      · zero-crossings <b>{s['zcr']:.0f}/s</b> · formant contrast <b>{s['contrast']:.0f}×</b></p>
    </section>""")

    rows = "\n".join(
        f"<tr><td>{html.escape(titles[s])}</td><td>{stats[s]['dur']:.2f}</td>"
        f"<td>{stats[s]['harm'] * 100:.1f}%</td><td>{stats[s]['zcr']:.0f}</td>"
        f"<td>{stats[s]['contrast']:.0f}×</td></tr>" for s in order)

    doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Making speech out of waves — ten renderings of one sentence</title>
<style>
  :root {{ color-scheme: dark; }}
  body {{ margin: 0 auto; max-width: 900px; padding: 32px 20px 80px;
         background: #101014; color: #e6e6ea;
         font: 15px/1.55 ui-sans-serif, system-ui, -apple-system, sans-serif; }}
  h1 {{ font-size: 26px; margin: 0 0 6px; }}
  h2 {{ font-size: 15px; margin: 0 0 4px; font-weight: 600; }}
  .lede {{ color: #9a9aa6; margin: 0 0 24px; }}
  section.card {{ border: 1px solid #26262e; border-radius: 10px; padding: 14px 16px;
                  margin: 0 0 14px; background: #16161c; }}
  .says {{ margin: 0 0 8px; color: #c7c7d2; font-size: 13.5px; }}
  .knobs {{ margin: 6px 0 0; color: #8f8f9c; font-size: 12.5px; }}
  .knobs b {{ color: #d8d8e2; }}
  audio {{ width: 100%; height: 34px; margin: 6px 0 8px; }}
  img.spec {{ width: 100%; border-radius: 6px; display: block;
              border: 1px solid #26262e; background: #000; }}
  pre {{ background: #101014; border: 1px solid #26262e; border-radius: 8px;
         padding: 10px 12px; overflow-x: auto; font-size: 12.5px; color: #c7c7d2;
         font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; margin: 8px 0 0; }}
  th, td {{ text-align: left; padding: 6px 8px; border-bottom: 1px solid #26262e; }}
  th {{ color: #7c7c8a; font-weight: 600; }}
</style></head><body>
<h1>Making speech out of waves</h1>
<p class="lede">One sentence — “{html.escape(TEXT)}” — rendered ten ways, all at the
same knobs (pitch 64, speed 72, throat 128, mouth 128). Every clip is generated by
Python in this folder; nothing is sampled except the SAM reference. Listen for which
ones still sound like speech and which turn into noise, then read the formula under each.</p>

<h2 style="margin-bottom:8px">The numbers</h2>
<table>
<tr><th>Rendering</th><th>duration</th><th>harmonicity</th><th>zero-crossings/s</th><th>formant contrast</th></tr>
{rows}
</table>
<p class="lede">Harmonicity = share of 0–10 kHz energy sitting within ±40 Hz of a
harmonic of f0. Higher is more voice-like. Zero-crossings per second above ~2× f0 means
high-frequency hash is dominating. Formant contrast = loudest spectral peak over the
median level in 200–3500 Hz: low values mean the formants have merged into mush.</p>

<h2 style="margin-bottom:8px">Listen</h2>
{chr(10).join(cards)}
</body></html>
"""
    with open(os.path.join(HERE, "experiments.html"), "w") as f:
        f.write(doc)


if __name__ == "__main__":
    raise SystemExit(main())
