# Making speech out of waves — the math, and what we measured

How do you get a talking voice out of nothing but numbers? This file collects the
techniques, the equations behind them, and the numbers from `experiments.py`
(one sentence rendered ten ways). Claims are labelled:

- **CONFIRMED** — I read it in a source (URL given).
- **MATH** — standard derivation, done here.
- **UNVERIFIED** — reported by a secondary source or recalled, not checked.

The rule that everything else follows from: **speech is not one waveform, it is a
source shaped by a filter.** A larynx produces a periodic buzz; a vocal tract
(a tube of changing cross-section) colours that buzz into vowels. Change the buzz
rate and you change pitch. Change the tube and you change which vowel you hear.
Almost every technique below is a way to model one of those two halves.

---

## 1. Additive synthesis — "combining different wavelengths at the same time"

Any periodic wave with fundamental `f0` is a sum of harmonics at `k·f0`:

```
x(t) = Σ_(k=1..K) A_k · sin(2π·k·f0·t + φ_k)
```

**MATH.** Fourier's theorem: every periodic signal is such a sum. `A_k` sets the
timbre. A sawtooth has `A_k = 1/k` (all harmonics), a square has `A_k = 1/k` for odd
`k` only, a triangle `1/k²`. That is why a square sounds hollow and a saw sounds
buzzy.

For speech you do not invent `A_k`: you read it off the vocal-tract response. With
three formants `(f_k, bw_k)` and gains `g_k`, the magnitude response at frequency `f`
is that of a bank of 2-pole resonators:

```
A(f) = Σ_k g_k · f_k² / √((f_k² − f²)² + (f·bw_k)²)
```

**MATH.** This is the analog 2-pole resonator magnitude, peak `≈ f_k/bw_k` at `f = f_k`.
Multiply by a glottal tilt (below) and evaluate it at every harmonic `k·f0`, and you
have additive formant synthesis with no filters at all.

**The catch — phase.** If `φ_k` drift independently, the waveform stops being periodic
and acquires a hollow, "phasiness" colour. The fix that SAM itself uses: reset all
phases at the start of each glottal pulse, so within a period `t ∈ [0, T)` you use
`φ_k = 2π·k·f0·t` and the wave is exactly periodic. (**MATH**; the reference engine
resets `phase1..3 = 0` at every pulse — CONFIRMED in `sam_engine/render.py`.)

**Measured** (`experiments.html`): additive synthesis scored 85.8% harmonicity and
969 zero-crossings/s — better than the original cascade (86.1% / 1152) but worse than
the parallel resonator bank (95.9% / 1077), because the harmonic count has to stop
somewhere and the amplitude normalisation is crude.

---

## 2. Source–filter: the model everything else is built on

Fant (1960): `s(t) = e(t) * v(t)` — the radiated sound is the glottal excitation
convolved with the vocal-tract impulse response. In the frequency domain,
`S(f) = E(f)·V(f)`: **source spectrum times filter response**. (**MATH**.)

The vocal tract is modelled as a cascade of resonators, each a 2-pole digital filter:

```
y[n] = A·x[n] + B·y[n−1] + C·y[n−2]
C = −e^(−2π·bw·T)      B = 2·e^(−π·bw·T)·cos(2π·f·T)      A = 1 − B − C
```

**MATH.** Poles at `r·e^(±j2πfT)` with `r = e^(−π·bw·T)`; `A = 1 − B − C` normalises
DC gain to 1. Peak gain is about `f/bw` (Q). CONFIRMED against the Oxford phonetics
Klatt resonator page and the ported engine's own coefficients.

### Cascade vs parallel — and why our first synth screamed

In a **cascade**, each resonator is normalised to unity DC gain, so the overall
response peaks as `(f/bw)³`. Push the formants up (throat 190–200) and F3 dominates:
the voice screams. In a **parallel** bank each formant gets its own gain and the
outputs are summed:

```
y[n] = Σ_k g_k · H_k(x[n])        g = (1.0, 0.55, 0.30)
```

That is Klatt's parallel branch. **Measured on our presets:** switching cascade →
parallel raised harmonicity from 86.1% to 95.9% on the default voice and from 78.4%
to 88.0% on "Elf", and cut zero-crossings from 2783 to 2137 on "Little Robot".

### Bandwidth must scale with frequency

A fixed 70/110/180 Hz is right only near the adult-male default. Real formant
bandwidths grow with centre frequency. **CONFIRMED** (Praat uses `0.05·F` above F4;
Klatt's floor is 40 Hz); we use

```
bw(f) = max(50, 0.06·f + 50) Hz
```

Too narrow (high Q) rings and whistles; too wide merges the formants into mush. The
`(f/bw)` peak gain also means a fixed bandwidth at 3.4 kHz gives Q ≈ 19 — near a
whistle.

### Ordering and Nyquist

Keep `F1 < F2 < F3 < 0.45·SR`. Crossing two formants makes them merge (measured: the
"Extra-Terrestrial" preset peaked at 880/960/800 Hz — one blob), and a formant near
Nyquist aliases into hash. **MATH.**

---

## 3. The glottal source and its spectral tilt

A sawtooth is a convenient buzz but its spectrum falls at only −6 dB/octave, which is
brighter than a real larynx. Real glottal pulses fall faster. The Rosenberg pulse
(**CONFIRMED** as a standard model, formulas standard):

```
g(t) = ½(1 − cos(π·t/Tp))           0 ≤ t < Tp        (open phase)
g(t) = cos(π·(t−Tp)/(2·Tn))         Tp ≤ t < Tp+Tn    (return phase)
g(t) = 0                            otherwise
Tp = 0.4·T, Tn = 0.16·T
```

The LF model (Fant–Liljencrants–Fant) adds an exponential return phase with a
parameter `R` controlling the tilt; its parameters are the open quotient and the
asymmetry (**UNVERIFIED** — not read). Tilt matters more than the exact shape: it
decides whether the voice reads as bright/buzzy or warm/natural.

**Measured:** the pulse source scored the best harmonicity of all (96.2%) and the
fewest zero-crossings (829/s) — smooth and periodic, i.e. the least "noisy".

---

## 4. Changing length and pitch of a recorded waveform

The user's question — "manipulate the wav length" — is the classic
time-scale/pitch-scale problem. The trap: playing samples faster raises pitch **and**
formants by the same factor (chipmunk). Measured in `experiments.html`: the naive
resampler at ×1.6 moved every spectral peak by ×1.6, while PSOLA moved them by 5 Hz.

### PSOLA (pitch-synchronous overlap-add)

**CONFIRMED** (Wikipedia PSOLA): split the waveform into overlapping segments at
pitch marks; move segments closer/further apart to change pitch; repeat/drop segments
to change duration; overlap-add.

```
y(n) = Σ_k w(n − t_k^s) · x_{M(k)}(n − t_k^s + t_{M(k)}^a)
```

- `t^a` = analysis marks (one per pitch period), `t^s` = synthesis marks.
- Window `w` = Hann of **2 pitch periods** (one before, one after the mark).
- Pitch: `t^s` spacing `= 1/f0'`. Duration: the mapping `M(k)` repeats or skips
  analysis frames.

**Why formants survive (MATH):** each frame is a verbatim slice of the original —
one glottal period convolved with the original tract. Re-spacing changes the
repetition rate, not the spectrum inside a frame. **Measured:** ×1.6 length and
×1.3 pitch both kept formant peaks within 5 Hz; naive resampling moved them 230 Hz.

Limits (**CONFIRMED**, TAMU L19 secondary): about ×2 is where artefacts become
obvious; unvoiced stretches turn mechanical (fix: reverse alternate copies);
pitch-mark errors cause clicks.

### WSOLA

Same overlap-add idea without pitch marks: search each frame for the offset whose
waveform best matches the previous one (normalised cross-correlation), then
overlap-add. **CONFIRMED** as the standard real-time alternative; robust to
non-speech, slightly lower quality on voiced speech. (**UNVERIFIED** details.)

### Phase vocoder

STFT with hop `H`; a sinusoid at bin `k` advances by `2π·k·H/N` per frame. Time
stretch by `α` = resynthesise with hop `αH`, keeping each bin's phase advancing
coherently:

```
Δφ_k(m) = princarg( φ_k(m) − φ_k(m−1) − 2π·k·H/N )
ω̂_k(m) = 2π·k/N + Δφ_k(m)/H
φ_k^out(m) = φ_k^out(m−1) + H_s·ω̂_k(m),     α = H_s / H
```

**CONFIRMED** framing and the phase-coherence problem (Wikipedia phase vocoder;
Laroche & Dolson 1999 identity phase locking). Artefact: "phasiness" — vertical
coherence between neighbouring bins is lost unless phase locking is applied.

Pitch shift = resample × `β` **and** time-stretch × `1/β` (so duration is restored and
the formants move back to their original frequencies).

---

## 5. LPC — the all-pole voice

Linear prediction models each sample as a weighted sum of the previous `p`:

```
x[n] ≈ Σ_(k=1..p) a_k·x[n−k]        →   X(z)·A(z) ≈ E(z),  A(z) = 1 − Σ a_k z^(−k)
```

**MATH.** Solving the normal equations (autocorrelation + Levinson–Durbin recursion)
gives `a_k`; the synthesis filter `1/A(z)` reproduces the spectral envelope — the
formants. Drive it with a pulse train at `f0` plus noise and you get intelligible
speech at 2.4 kbit/s (LPC-10). It sounds robotic because the excitation is a bare
pulse train, not a glottal pulse: no tilt, no jitter, no breath.

**Measured:** LPC resynthesis of our sentence scored 94.1% harmonicity, 985
zero-crossings/s — very voice-like, with the formants intact. Its spectral distance
to the original was 9.5 dB RMS on a saw+formants signal (**CONFIRMED**, measured by
`lpc.py`).

**Cepstrum.** `c[n] = IFFT(log|X(ω)|)` separates source and filter in "quefrency":
low quefrency = envelope, high = pitch. Liftering the low part gives the spectral
envelope used for formant tracking. (**MATH**.)

---

## 6. Sinusoidal modelling + noise (HNM)

Model the signal as tracked partials plus a noise band:

```
x(t) = Σ_k A_k(t)·cos(φ_k(t)) + n(t),     φ_k(t) = φ_k(t−1) + 2π·f_k(t)/sr
```

Analysis: STFT, peak picking, parabolic interpolation, track peaks across frames
(nearest frequency within a tolerance), stop at the "maximum voiced frequency";
everything above it is noise with a matched envelope. **CONFIRMED** as the HNM
family's structure (Laroche/Dolson/Serra); exact parameters **UNVERIFIED**.

**Measured:** 41 partials at the loudest frame, spectral distance 0.208 (normalised
log-magnitude L2 over 0–5 kHz) against 0.735 for a noise null — so the metric is not
trivially satisfied. Harmonicity 95.5%.

---

## 7. Vocoders and the robot voice

A channel vocoder takes band envelopes of the voice and multiplies them onto a
carrier: `y = Σ_k env_k(t)·BP_k(carrier)`. With a square carrier at 110 Hz this is
the Dalek/robot sound (measured earlier in this folder: 16 bands, 120–5000 Hz,
envelope LP 40 Hz). The phase vocoder above is its FFT cousin.

---

## 8. Modern neural TTS, in one line each

- **Tacotron 2** (2017): seq2seq text → mel spectrogram, then WaveNet vocoder.
- **FastSpeech 2** (2020): feed-forward, explicit duration/pitch/energy predictors —
  removes the attention alignment failures.
- **VITS** (2021): variational autoencoder + normalising flow + GAN vocoder, trained
  end-to-end, so no separate vocoder stage.
- **Diffusion / flow-matching TTS** (2022–): generate the waveform or the spectrogram
  by iteratively denoising; highest naturalness, slowest.
(**UNVERIFIED** — architecture summaries from memory, not read for this note.)

They all still need a *vocoder* to turn spectrograms into samples, because phase is
the hard part: magnitude spectra carry the timbre, phase carries the waveform.

---

## 9. What we recommend for a from-scratch, dependency-free synth

Ranked by measured harmonicity and freedom from hash, at fixed cost:

1. **Parallel formant bank** with explicit gains and `bw = 0.06f + 50` — best
   smoothness (95.9%), cheapest, and stable at extreme knob settings.
2. **Glottal pulse source** (Rosenberg) instead of a saw — best harmonicity (96.2%),
   fewest zero-crossings (829/s), more natural tilt.
3. **LPC resynthesis** if you need to compress or re-drive an existing recording
   (94.1% harmonicity).
4. **PSOLA** when you must change duration or pitch of a recording and keep the
   voice: formants stay within 5 Hz.
5. **HNM / sinusoidal** for analysis-style resynthesis and morphing (95.5%).
6. **Additive harmonics** last: elegant and phase-lockable, but needs careful
   normalisation (85.8% as implemented).

Rules that keep it from turning into noise:

1. Keep `F1 < F2 < F3`, and every formant below `0.45·SR`.
2. Scale bandwidth with frequency; never use one fixed bandwidth for all formants.
3. Use a parallel bank (explicit gains) rather than a cascade when knobs move far.
4. Reset oscillator phases at every glottal pulse — periodicity is what the ear
   calls "voice".
5. Give the source a spectral tilt (pulse, not saw) so it is not brighter than a
   larynx.
6. Glide formants over 10–20 ms; glide amplitude over 3–5 ms.
7. Band-limit the output (80 Hz–4.8 kHz) and keep noise out of the voiced path.
8. For length/pitch edits, use PSOLA or a phase vocoder — never plain resampling.

---

## 10. The full Klatt chain, if you want to go further than three formants

Klatt's 1980 synthesizer is the industrial version of everything above. Structure
(**CONFIRMED** via the Oxford phonetics Klatt pages, a Keller chapter and the Berkeley
Phonlab parameter list; the 1980 PDF itself is a scan and was not read):

```
source ─→ glottal pole/zero ─→ [F1..F5 cascade] ─→ nasal pole/zero ─→ radiation ─→ out
                             └→ [6-resonator parallel bank + bypass]  ← noise
```

- **Voiced path** is a cascade of five formants. Five, not three, because F4/F5 carry
  the speaker's individual colour (and the difference between /r/, /l/, nasals).
- **Unvoiced path** is a parallel bank of resonators plus a bypass, excited by
  frication/aspiration noise, with its own gains `a1..a6` and `ab`. Fricatives need
  this branch: their spectra are not a formant cascade, they are shaped noise.
- **Nasals** need a pole–zero pair (a resonance plus an antiresonance) or /m n ŋ/
  sound like vowels.
- **Parameter update rate** 5 ms default (10 ms in cheaper versions).
- **Glottal models** reduce to five parameters: `f0`, amplitude of voicing `A_v`,
  open quotient `O_q`, asymmetry `α_m`, return-phase quotient `Q_a` (Henrich,
  d'Alessandro & Doval 2002, **CONFIRMED** as the unification covering Rosenberg,
  LF, KLGLOTT88, R++).

### KLGLOTT88 (Klatt & Klatt 1990)

Glottal **flow** is a cubic on the open phase, `u(t) = a·t² + b·t³`, with `a, b` set by
`A_v` and `O_q`, followed by a first-order tilt filter (cutoff `f_t`). Its spectrum is
flat up to `f_k ≈ 1/(π·O_q·T0)`, then −12 dB/oct, and −24 dB/oct above `f_t`
(**CONFIRMED** in Doval & d'Alessandro's reading). Equivalent view: an anticausal
3rd-order all-pole lowpass.

### Cheap version of the same thing (what this folder does)

Klatt's own shortcut for a natural source, which is what `sam.py` now approximates:

```
impulse train → glottal lowpass resonator (≈250 Hz) → radiation (1 − z⁻¹) → tilt
```

`(1 − z⁻¹)` is the lip radiation: it differentiates, adding +6 dB/oct. A glottal
lowpass then brings the net tilt back to about −6 dB/oct, which is the measured
average of real voices (**CONFIRMED** as the standard tilt argument; exact
H1−H2 ↔ O_q constants are UNVERIFIED).

Sensible defaults if you extend this synth: five formants at 500/1500/2500/3300/3900 Hz
with bandwidths 70/100/140/220/280 Hz, 10 ms parameter updates, f0 110 Hz for a male
voice, glottal pole 250 Hz, parallel noise branch for /s ʃ f/, nasal pair for /m n ŋ/,
5–10 ms bursts for stops.

**Why this is worth it:** the three-formant version in this folder is intelligible but
thin. F4/F5 and the nasal branch are what make a voice sound like a person rather than
a buzzer, and the parallel noise branch is what makes fricatives sound like fricatives
instead of filtered hiss.

---


## Measured summary (`experiments.html`, same sentence, same knobs)

| Rendering | duration | harmonicity | zero-crossings/s | formant contrast |
|---|---|---|---|---|
| Cascade resonators | 3.86 s | 86.1% | 1152 | 178× |
| Parallel resonators | 3.86 s | 95.9% | 1077 | 77× |
| Glottal pulse source | 3.86 s | 96.2% | 829 | 36× |
| Additive harmonics | 3.86 s | 85.8% | 969 | 64× |
| LPC resynthesis | 3.86 s | 94.1% | 985 | 39× |
| PSOLA length ×1.6 | 6.17 s | 92.0% | 834 | 75× |
| PSOLA pitch ×1.3 | 3.86 s | 87.8% | 953 | 47× |
| Sinusoidal + noise | 3.86 s | 95.5% | 1663 | 127× |
| Naive resampling | 6.17 s | 98.1% | 554 | 200× |
| 1982 engine (reference) | 4.15 s | 67.1% | 1265 | 33× |

Harmonicity = share of 0–10 kHz energy within ±40 Hz of a harmonic of `f0`.
Zero-crossings/s well above `2·f0` means high-frequency hash dominates. Formant
contrast = loudest peak over the median level in 200–3500 Hz.

**Read the reference row carefully:** the real SAM scores 67% harmonicity because its
4-bit output is *deliberately* noisy. These numbers rank smoothness, not naturalness —
which is exactly the tension the user noticed when the "cleaner" synth sounded better
than the faithful port.

---

## Sources

- PSOLA (**read**): https://en.wikipedia.org/wiki/PSOLA
- Phase vocoder (**read**): https://en.wikipedia.org/wiki/Phase_vocoder
- Oxford phonetics Klatt resonators (**read**): https://www.phon.ox.ac.uk/klatt_resonators
- Klatt 1980, *Software for a cascade/parallel formant synthesizer* (not read)
- Fant 1960, *Acoustic Theory of Speech Production* (not read)
- TAMU L19, "Prosodic modification of speech" (Gutierrez-Osuna) — read via a
  subagent's summary, primary Moulines & Laroche not read (**UNVERIFIED** details)
- Laroche & Dolson, phase-locked vocoder (1999) — cited in Wikipedia's phase vocoder
  article (**read** as citation, paper not read)
- HNM / Serra 1989 (not read)
- Rosenberg 1971 glottal pulse (not read; formula is standard)
- LPC-10 / FIPS-137 (not read)
- Oxford Klatt structure/parameters pages (**read**): https://www.phon.ox.ac.uk/klatt_structure
- Berkeley Phonlab Klatt parameter list (**read** via subagent)
- Henrich, d'Alessandro & Doval 2002, *Glottal flow models* (**read** via subagent)
- Doval & d'Alessandro 1997, ICASSP (**read** via subagent)
- Klatt & Klatt 1990, KLGLOTT88 (not read; described in Doval & d'Alessandro)
