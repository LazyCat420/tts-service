"""Build a single self-contained HTML artifact: every clip embedded as base64.

    python3 make_artifact.py [output.html]

Default output: /tmp/sam-demo.html. Requires `audio/` to exist (run demo.py).
"""

from __future__ import annotations

import base64
import html
import os
import sys

import computer_voices as cv
import demo
import sam

HERE = os.path.dirname(os.path.abspath(__file__))
AUDIO = os.path.join(HERE, "audio")

CV_DESC = {
    "chip": "square source through the formant cascade",
    "vocoder": "16-band envelopes onto a 110 Hz square carrier",
    "ring": "ring modulation, 45 Hz, 35% wet",
    "crush": "8-bit, sample-and-hold 2, TPDF dither",
    "blips": "one square blip per letter, 20/s",
}

CSS = """
 :root { color-scheme: dark; }
 body { margin:0 auto; max-width:900px; padding:32px 20px 80px; background:#101014; color:#e6e6ea;
        font:15px/1.55 ui-sans-serif,system-ui,-apple-system,sans-serif; }
 h1 { font-size:26px; margin:0 0 6px; letter-spacing:-0.01em; }
 h2 { font-size:15px; margin:0 0 4px; font-weight:600; }
 .lede { color:#9a9aa6; margin:0 0 24px; }
 section.card { border:1px solid #26262e; border-radius:10px; padding:14px 16px; margin:0 0 14px; background:#16161c; }
 .knobs { margin:0 0 6px; color:#8f8f9c; font-size:13px; }
 .knobs b { color:#d8d8e2; }
 .says { margin:0 0 10px; color:#c7c7d2; font-size:13.5px; }
 audio { width:100%; height:34px; }
 h3 { font-size:13px; text-transform:uppercase; letter-spacing:0.08em; color:#7c7c8a;
      margin:34px 0 10px; font-weight:600; }
 pre { background:#16161c; border:1px solid #26262e; border-radius:8px; padding:12px 14px;
       overflow-x:auto; font-size:12.5px; color:#c7c7d2; }
 table { width:100%; border-collapse:collapse; font-size:13px; }
 th,td { text-align:left; padding:6px 8px; border-bottom:1px solid #26262e; }
 th { color:#7c7c8a; }
 .good { color:#7fd6a0; }
 .bad { color:#ff8f8f; }
"""


def player(name: str) -> str:
    path = os.path.join(AUDIO, name)
    data = base64.b64encode(open(path, "rb").read()).decode()
    return f'<audio controls preload="none" src="data:audio/wav;base64,{data}"></audio>'


def card(title: str, sub: str, wav: str) -> str:
    return (f'<section class="card"><h2>{html.escape(title)}</h2>'
            f'<p class="says">{html.escape(sub)}</p>{player(wav)}</section>')


def main(argv: list[str]) -> int:
    out_path = argv[0] if argv else "/tmp/sam-demo.html"
    line = demo.MONTAGE_LINE

    preset_cards = []
    for name, (speed, pitch, throat, mouth) in sam.PRESETS.items():
        slug = name.lower().replace(" ", "-")
        preset_cards.append(
            f'<section class="card"><h2>{html.escape(name)}</h2>'
            f'<p class="knobs">speed <b>{speed}</b> · pitch <b>{pitch}</b> · '
            f'throat <b>{throat}</b> · mouth <b>{mouth}</b></p>'
            f'<p class="says">“{html.escape(line)}”</p>'
            f'{player(f"preset-{slug}.wav")}</section>')

    phrase_cards = [card(title, f"“{text}”", f"phrase-{slug}.wav")
                    for slug, title, text in demo.PHRASES]

    faithful_cards = []
    for name, (speed, pitch, throat, mouth) in sam.PRESETS.items():
        slug = name.lower().replace(" ", "-")
        faithful_cards.append(
            f'<section class="card"><h2>{html.escape(name)}</h2>'
            f'<p class="knobs">speed <b>{speed}</b> · pitch <b>{pitch}</b> · '
            f'throat <b>{throat}</b> · mouth <b>{mouth}</b></p>'
            f'<p class="says">“{html.escape(line)}”</p>'
            f'{player(f"faithful-{slug}.wav")}</section>')

    cv_cards = [card(k, v, f"cv-{k}.wav") for k, v in CV_DESC.items()]

    preset_rows = "".join(
        f"<tr><td>{html.escape(n)}</td><td>{p[0]}</td><td>{p[1]}</td>"
        f"<td>{p[2]}</td><td>{p[3]}</td></tr>" for n, p in sam.PRESETS.items())

    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SAM demo — real engine + clean-room synth</title>
<style>{CSS}</style></head><body>
<h1>SAM demo — the real engine and a clean-room synth</h1>
<p class="lede">Two implementations side by side. <code>sam.py</code> is clean-room:
text → phonemes → a buzzy glottal pulse train through three cascaded formant resonators.
<code>sam_engine/</code> is a faithful port of the actual 1982 engine — its tables, three
direct oscillators, 4-bit output — verified byte-identical to the reference. Audio is
embedded, so this page plays on its own.</p>

<h3>Real SAM vs the clean-room synth</h3>
<p class="lede">Identical text and knobs (72 / 64 / 128 / 128). The port sums three
oscillators directly — sin(F1)·A1 + sin(F2)·A2 + rect(F3)·A3 — and resets their phase at
every glottal pulse; the clean-room one filters a sawtooth through resonators. That hard
sync is why SAM's formants cut through. The port also takes 4.87 s where the synth takes
2.39 s: SAM's own phoneme durations and stress, not a tuning choice.</p>
{card("sam.py — clean-room cascade synth", "sawtooth → three resonators", "ab-ours.wav")}
{card("sam_engine — faithful port of the 1982 renderer", "three direct oscillators, phase reset per glottal pulse", "ab-faithful.wav")}

<h3>Faithful engine, six presets</h3>
{card("Faithful montage", "Each preset announces its name, then speaks one line.", "montage-faithful.wav")}
{''.join(faithful_cards)}

<h3>The whole montage (clean-room)</h3>
{card("Six presets", "Each announces its name, then speaks one line.", "montage.wav")}

<h3>Clear vs fuzzy — what changed</h3>
<p class="lede">Same text, same synth. “Before” has no output band-limiting and the raw
white-noise path (gain 1.4, bandwidth 1200 Hz). Measured on that clip:
<span class="bad">7.56% of energy above 5 kHz and 2143 sample steps over 25% of peak</span>
— the distorted hiss. “After” band-limits at 80 Hz–4.8 kHz and tightens the noise
(gain 0.35, bandwidth 700 Hz): <span class="good">0.14% and 0 steps</span>.</p>
{card("Before", "no band-limiting, raw noise path", "clarity-before.wav")}
{card("After", "80 Hz–4.8 kHz band-limit, tighter noise", "clarity-after.wav")}

<h3>Same sentence, six presets</h3>
<p class="lede">Only the four 1982-manual parameters change between clips.</p>
{''.join(preset_cards)}

<h3>Different sentences, default SAM voice</h3>
{''.join(phrase_cards)}

<h3>Voices made of computer sounds</h3>
<p class="lede">Five DSP-only techniques driven by the same synth. See
<code>RESEARCH-COMPUTER-VOICES.md</code> for the math and sources.</p>
{''.join(cv_cards)}

<h3>Presets</h3>
<table><tr><th>Preset</th><th>Speed</th><th>Pitch</th><th>Throat</th><th>Mouth</th></tr>
{preset_rows}</table>

<h3>What each knob does</h3>
<pre>pitch   0-255  base f0 = 20 + pitch * 1.15 Hz      (64 → ~94 Hz)
speed   0-255  phoneme duration scale = 72/speed
throat  0-255  formant scale = (throat/128)^0.75   (all three formants)
mouth   0-255  F1 scale      = (mouth/128)^0.75    (apparent mouth size)</pre>

<h3>Signal chain</h3>
<pre>text ──reciter──→ phonemes ──render──→ wav

buzz = polyBLEP sawtooth at f0, 18% declination across the utterance
out  = buzz·voiced ─→ F1 ─→ F2 ─→ F3    (2-pole resonators, bw 70/110/180 Hz)
     + noise·fricative ─→ resonator at the fricative band
     ─→ high-pass 80 Hz ─→ low-pass 4.8 kHz</pre>

<h3>Computer-sound techniques</h3>
<pre>chip     square source (odd harmonics, 1/n) through the same cascade
vocoder  e_k = LP40(|BP_k(x)|);  y = Σ e_k · BP_k(square@110 Hz)   16 bands 120–5000 Hz
ring     y = x·((1−w) + w·sin(2π·45·t)), w = 0.35   → sidebands at k·f0 ± 45 Hz
crush    8-bit quantise + hold 2 + TPDF dither       SNR ≈ 6.02·N + 1.76 dB
blips    one square blip per letter at 20/s          prosody only, not words</pre>
</body></html>"""

    with open(out_path, "w") as f:
        f.write(doc)
    print(f"{out_path}  {len(doc)/1e6:.2f} MB  {doc.count('<audio')} clips")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
