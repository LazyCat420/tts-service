"""Builds the from-scratch synthesiser demo: clips, spectrograms, montage, page.

    python3 synth_demo.py            # writes audio/, spec/ and synth-demo.html

Every clip is produced by `app.synth_manager` (reciter -> prosody -> frames ->
klatt -> post), the same path the HTTP service uses for `engine="synth"`.
Metrics come from `measure.py` so they are comparable with `experiments.json`.
"""

from __future__ import annotations

import html
import os
import sys
import wave

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import demo  # noqa: E402  (spectrogram/_player/_rel helpers)
import measure  # noqa: E402

from app import synth_manager  # noqa: E402
from app.synth import presets  # noqa: E402

AUDIO = os.path.join(HERE, "audio")
SPEC = os.path.join(HERE, "spec")
SR = synth_manager.SAMPLE_RATE

# (slug, voice, line) — the announcement is prepended to the shared line
VOICE_LINES = [
    ("narrator", "narrator", "Narrator"),
    ("female", "female", "Female"),
    ("child", "child", "Child"),
    ("giant", "giant", "Giant"),
    ("whisper", "whisper", "Whisper"),
    ("aged", "aged", "Aged"),
    ("droid", "droid", "Droid"),
    ("sam", "SAM", "SAM"),
    ("little-robot", "Little Robot", "Little Robot"),
    ("elf", "Elf", "Elf"),
    ("stuffy-guy", "Stuffy Guy", "Stuffy Guy"),
    ("little-old-lady", "Little Old Lady", "Little Old Lady"),
    ("extra-terrestrial", "Extra-Terrestrial", "Extra Terrestrial"),
]
MONTAGE_LINE = "I am a little robot. I come from 1982."
PHRASES = [
    ("pangram", "The quick brown fox jumps over the lazy dog."),
    ("intro", "Hello. I am SAM. Software Automatic Mouth."),
    ("drift", "Oh my god. Whoa. Watch out for that car."),
]


def write_wav(path: str, x: np.ndarray, sr: int = SR) -> None:
    pcm = (np.clip(x, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def row(x: np.ndarray, sr: int = SR) -> dict:
    g = measure.guards(x, sr)
    g.update(measure.f0_stats(measure.f0_track(x, sr)))
    durs = measure.nuclei_durations(x, sr)
    g["rate_syl_s"] = round(len(durs) / max(g["dur_s"], 1e-6), 2)
    return g


def main() -> int:
    os.makedirs(AUDIO, exist_ok=True)
    os.makedirs(SPEC, exist_ok=True)

    clips: list[tuple[str, str, str, str, dict]] = []
    montage: list[np.ndarray] = [np.zeros(int(0.2 * SR))]
    for slug, voice, label in VOICE_LINES:
        text = f"{label}. {MONTAGE_LINE}"
        x = synth_manager.render(text, voice)
        wav = os.path.join(AUDIO, f"synth-{slug}.wav")
        png = os.path.join(SPEC, f"synth-{slug}.png")
        write_wav(wav, x)
        demo.spectrogram(wav, png)
        clips.append((slug, voice, label, wav, row(x)))
        montage += [x, np.zeros(int(0.25 * SR))]
        print(f"{voice:18s} {len(x)/SR:5.2f}s peak={np.max(np.abs(x)):.2f}")

    montage_wav = os.path.join(AUDIO, "synth-montage.wav")
    write_wav(montage_wav, np.concatenate(montage))

    phrase_clips = []
    for slug, text in PHRASES:
        x = synth_manager.render(text, "narrator")
        wav = os.path.join(AUDIO, f"synth-phrase-{slug}.wav")
        png = os.path.join(SPEC, f"synth-phrase-{slug}.png")
        write_wav(wav, x)
        demo.spectrogram(wav, png)
        phrase_clips.append((slug, text, wav, png, row(x)))

    # legacy 1982 engine on the same sentence, for the comparison row
    import sam  # noqa: E402
    legacy = np.asarray(sam.render(PHRASES[0][1]), dtype=float)
    legacy_wav = os.path.join(AUDIO, "synth-legacy-sam.wav")
    write_wav(legacy_wav, legacy, sam.SR)
    legacy_row = row(legacy, sam.SR)

    write_html(clips, montage_wav, phrase_clips, legacy_wav, legacy_row)
    return 0


def _metrics_table(clips, legacy_row) -> str:
    head = ("voice", "dur s", "F0 med", "span st", "harm", "HNR dB", "CPP dB",
            ">5 kHz %", "crackle", "syl/s")
    rows = []
    for slug, voice, label, wav, m in clips:
        rows.append(
            f"<tr><td>{html.escape(label)}</td>"
            f"<td>{m['dur_s']}</td><td>{m['f0_med']}</td><td>{m['span_st']}</td>"
            f"<td>{m['harm']}</td><td>{m['hnr_db']}</td><td>{m['cpp_db']}</td>"
            f"<td>{m['hi5k_pct']}</td><td>{m['crackle']}</td>"
            f"<td>{m['rate_syl_s']}</td></tr>")
    rows.append(
        f"<tr class=old><td>1982 SAM (legacy)</td>"
        f"<td>{legacy_row['dur_s']}</td><td>{legacy_row['f0_med']}</td>"
        f"<td>{legacy_row['span_st']}</td><td>{legacy_row['harm']}</td>"
        f"<td>{legacy_row['hnr_db']}</td><td>{legacy_row['cpp_db']}</td>"
        f"<td>{legacy_row['hi5k_pct']}</td><td>{legacy_row['crackle']}</td>"
        f"<td>{legacy_row['rate_syl_s']}</td></tr>")
    return ("<table><thead><tr>"
            + "".join(f"<th>{h}</th>" for h in head)
            + "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>")


def _params_table() -> str:
    keys = ("pitch", "speed", "vtl", "oq", "tilt_db", "breath", "nasality",
            "f4_gain", "f5_gain")
    rows = []
    for name in presets.names():
        v = presets.voice(name)
        rows.append(f"<tr><td>{html.escape(name)}</td>"
                    + "".join(f"<td>{v.get(k, '—')}</td>" for k in keys) + "</tr>")
    return ("<table><thead><tr><th>voice</th>"
            + "".join(f"<th>{k}</th>" for k in keys)
            + "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>")


def write_html(clips, montage_wav, phrase_clips, legacy_wav, legacy_row) -> None:
    blocks = []
    for slug, voice, label, wav, m in clips:
        png = os.path.join(SPEC, f"synth-{slug}.png")
        blocks.append(
            f"<section><h2>{html.escape(label)}</h2>"
            f"<p class=says>“{html.escape(MONTAGE_LINE)}” · {m['dur_s']:.2f} s</p>"
            f"{demo._player(wav, png)}</section>")
    phrase_blocks = []
    for slug, text, wav, png, m in phrase_clips:
        phrase_blocks.append(
            f"<section><h2>{html.escape(slug)}</h2>"
            f"<p class=says>“{html.escape(text)}” · {m['dur_s']:.2f} s</p>"
            f"{demo._player(wav, png)}</section>")

    doc = f"""<!doctype html>
<html lang="en"><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>From-scratch voice synthesiser</title>
<style>
:root{{--fg:#e8e6e3;--bg:#12131a;--muted:#9aa0b0;--line:#2a2d3a;--accent:#7fd1ff}}
*{{box-sizing:border-box}}
body{{margin:0;padding:32px;background:var(--bg);color:var(--fg);
font:15px/1.5 ui-sans-serif,system-ui,sans-serif;max-width:1100px;margin:auto}}
h1{{font-size:24px;margin:0 0 4px}}h2{{font-size:15px;margin:0 0 2px}}
p{{color:var(--muted)}}.says{{margin:0 0 8px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:20px}}
section{{border:1px solid var(--line);border-radius:10px;padding:12px}}
audio{{width:100%}}img.spec{{width:100%;border-radius:6px;margin-top:8px}}
table{{border-collapse:collapse;width:100%;font-size:13px;margin:12px 0}}
th,td{{padding:5px 8px;border-bottom:1px solid var(--line);text-align:right}}
th:first-child,td:first-child{{text-align:left}}
thead th{{color:var(--accent);font-weight:600}}
tr.old td{{color:var(--muted);font-style:italic}}
code{{background:#1c1f2b;padding:1px 5px;border-radius:4px}}
</style></head><body>
<h1>From-scratch voice synthesiser</h1>
<p>Formant synthesis written from the literature: a rule-based reciter, a
Fujisaki-style prosody model, a Klatt parallel formant bank excited by a
period-synchronous impulse train, then de-essing, LTAS tilt, CELP postfilter
and BS.1770 loudness. numpy only, no model weights.</p>

<h2>Montage</h2>
<p class=says>Every voice type, each announcing itself.</p>
<audio controls src="{demo._rel(montage_wav)}"></audio>

<h2>Voice types</h2>
<div class=grid>{''.join(blocks)}</div>

<h2>Sentences (narrator)</h2>
<div class=grid>{''.join(phrase_blocks)}</div>

<h2>Comparison with the 1982 engine</h2>
<audio controls src="{demo._rel(legacy_wav)}"></audio>
<p class=says>The same pangram through the legacy SAM port: harm
{legacy_row['harm']}, {legacy_row['hi5k_pct']}% above 5 kHz.</p>

<h2>Metrics</h2>
{_metrics_table(clips, legacy_row)}
<p><code>harm</code> autocorrelation peak in 60-400 Hz · <code>crackle</code>
one-sample discontinuities · <code>&gt;5 kHz</code> share of spectral energy.</p>

<h2>Voice parameters</h2>
{_params_table()}
<p>Pitch and speed keep SAM's 0-255 scale; <code>vtl</code> scales the vocal
tract, <code>oq</code> the glottal open quotient, <code>tilt_db</code> the
source slope, <code>breath</code> aspiration, <code>nasality</code> nasal
colour.</p>

<h2>Service</h2>
<pre>POST /api/v1/tts/synthesize
{{"text": "Hello there.", "engine": "synth", "voice_accent": "narrator"}}</pre>
<p>Piper stays the default (<code>engine="piper"</code>); the synth engine is
selected per request and needs no model files.</p>
</body></html>
"""
    with open(os.path.join(HERE, "synth-demo.html"), "w") as f:
        f.write(doc)


if __name__ == "__main__":
    raise SystemExit(main())
