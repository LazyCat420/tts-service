# PLAN: our own voice synthesis — crisp, human-cadenced, many voice types

**Date:** 2026-11-06
**Repo:** `agents/tts-service` (NAS CPU Docker, Python 3.11, numpy 1.26.4, onnxruntime 1.18.1, espeak-ng installed)
**Current engine:** `sam-demo/` — clean-room SAM formant synth (`sam.py`) + byte-exact 1982 SAM port (`sam_engine/`) + DSP "computer voice" effects (`computer_voices.py`) + analysis/resynthesis tools (`lpc.py`, `psola.py`, `hnm.py`). Piper VITS is the service's default human voice.
**Constraint from the user:** build our own synthesis from scratch. No vendored repos, no shipped third-party weights. Published algorithms and equations are fair game (math is not copyrightable); code and model weights are not, so every technique below is reimplemented in numpy from its paper, and any neural model is trained by us.

**Method:** every claim is labelled **CONFIRMED** (primary source read this session, URL in Sources), **MATH** (derived), **MEASURED** (run in this repo, number from `sam-demo/experiments.json` / README), or **UNVERIFIED** (secondary/recalled, needs checking before coding). Research ran as five read-only subagents (prosody math, full Klatt chain, hybrid vocoders, evaluation metrics, CPU TTS landscape). Their full reports are session artifacts (`local://ProsodyMath.md`, `local://KlattChain.md`, `local://HybridVocoders.md`, `local://EvalMetrics.md`, `local://NeuralLandscape.md`); every load-bearing equation and number is reproduced inline below, so this plan stands alone.

---

## 0. The one-paragraph answer

What we have is a **1982 three-oscillator formant synth**: a buzz shaped by three resonators, 8-bit output, flat pitch, one fixed 15 ms smoothing constant for every phoneme boundary, no VOT, no fricative spectra, no nasal zeros, no prosody. It is intelligible but machine-flat. Human-like cadence and crispness are not one problem — they decompose into five layers, and we already have measured tools for four of them:

1. **Glottal source** (voice quality: buzzy vs warm vs breathy) — replace the saw/pulse with KLGLOTT88/LF + open quotient + tilt + per-period jitter/shimmer + aspiration.
2. **Vocal tract** (which word was said) — extend 3 formants to a Klatt cascade F1–F5 with nasal pole/zero, and replace the one-pole 15 ms smear with **locus-anchored, manner-timed transitions**, VOT machinery and per-manner fricative/burst spectra. This is the single largest intelligibility lever.
3. **Prosody** (human cadence) — a Fujisaki command-response F0 model plus a duration rule stack (stress, pre-boundary lengthening, cluster shortening) plus ToBI accents from punctuation. Published evidence: rule-based prosody at this level is heard as "just as natural as natural prosody" for English (CONFIRMED).
4. **Excitation and resynthesis** (crispness, no buzz, no ripple) — a WORLD-style front end we write ourselves: analytic formant envelope → mel-cepstrum smoothing → minimum-phase FIR → mixed excitation (F0 pulses + 5-band modified velvet noise) → phase-locked overlap-add. Pure numpy, measured 35–70 ms per 3 s sentence locally, estimated 0.1–0.4 s on the NAS.
5. **Post** (polish) — LTAS matching to a human long-term average spectrum, de-esser, loudness normalisation, and optional formant sharpening.

Then voice types are just parameter vectors over layers 1–3 (vocal-tract length scale, F0 baseline, open quotient/tilt, rate, nasality, whisper, fry), which is exactly how SAM already does presets — only with more physically meaningful knobs. A neural vocoder trained by us on *our own* parameters→real-audio pairs is an optional later phase; it is the only path to "sounds like a person", and it costs GPU training time plus a data licence decision.

---

## 1. What we have, measured

From `sam-demo/experiments.json` — one sentence, ten renderers, same formant targets:

| Renderer | Harmonicity | Zero-crossings/s | What it shows |
|---|---|---|---|
| `pulse` (Rosenberg glottal pulse, parallel bank) | **0.9621** | 829 | best smoothness of the DSP set |
| `parallel` (3-formant parallel bank, current default) | 0.9591 | 1077 | what `sam.py` ships |
| `hnm` (sinusoidal + noise) | 0.9551 | 1663 | analysis/resynthesis, not a synth |
| `lpc` (LPC resynthesis) | 0.9406 | 985 | envelope-faithful |
| `psola-len` / `psola-pit` | 0.9197 / 0.8781 | 834 / 953 | formants survive time/pitch edits (within 5 Hz) |
| `cascade` (saw → F1→F2→F3) | 0.8611 | 1152 | the original, screams at extreme knobs |
| `sam` (real 1982 engine) | 0.6708 | 1265 | 4-bit output grit is the sound, not a defect |
| `resample` (naive pitch shift) | 0.9805 | 554 | formants moved 230 Hz — a trap, not a technique |

Clarity fix already landed (README): output band-limit 80 Hz–4.8 kHz plus noise gain/bandwidth fix took high-band energy from **7.56% → 0.14%** and sample steps over 25% of peak from **2143 → 0**. The faithful port still measures **4.72% >5 kHz, 4144 steps** — that grit is SAM's identity and stays as a preset.

**Known gaps, stated by the repo itself:** no stress or intonation model (pitch falls 18% across the utterance, one 5.2 Hz vibrato, 0.3% white jitter); reciter is a small rule set plus lexicon; no `+whisper`/`+croak` modifiers; three formants only; one 15 ms smoothing constant for every boundary.

**Where the code hooks are:** `sam.py:synth()` is a per-sample loop over `(f1,f2,f3,voiced,noise,noise_hz,dur)` segments; f0 comes from `f0_base*(1-0.18*prog)` + vibrato + white jitter; formants glide with a one-pole at 15 ms, amplitudes at 5 ms. The SAM port builds per-frame `pitches[]`, `frequency[3][]`, `amplitude[3][]`, `flags[]` at 125 frames/s (`sam_engine/render.py:prepare_frames`), and `process_frames()` renders 5 samples per frame. Both are natural insertion points: prosody writes per-frame F0/amplitude, the Klatt chain replaces the oscillator sum.

---

## 2. Why the current voice sounds like a machine (five separate causes)

| # | Cause | Where | Fix layer |
|---|---|---|---|
| 1 | Flat pitch, no accents, no phrase structure, no final lengthening | `sam.py:synth` (`prog`-linear declination) | prosody (C) |
| 2 | One 15 ms one-pole on F1–F3 across every boundary: low-passes away the F2 transition slope that *is* the place cue, and blurs burst/VOT edges | `sam.py:_alpha(15.0)` | tract (B) |
| 3 | No VOT, no locus anchor, one generic closure+noise burst | `STOPS` branch in `synth()` | tract (B) |
| 4 | One noise band per phoneme (`noise_hz`, `noise_bw`) — /s/, /ʃ/, /f/, /θ/ collapse into the same hiss | `V` table | tract (B) |
| 5 | No nasal pole/zero pair, so /m/, /n/, /ŋ/ are just voiced vowels with low F1 | `V` table | tract (B) |
| 6 | Source is a saw/square/Rosenberg pulse with no open quotient, no tilt control, no aspiration, and jitter drawn per **sample** instead of per **period** (which is what local jitter actually measures) | `synth()` source branch | source (A) |
| 7 | Harmonic ripple in the envelope and a bare pulse train: "buzzy", and any STFT-domain processing acquires phasiness | whole chain | excitation (D) |
| 8 | No post chain: no LTAS match to a human voice, no de-esser, no loudness target | none | post (E) |

---

## 3. The math, layer by layer

### A. Glottal source — voice quality

**KLGLOTT88 flow pulse** (CONFIRMED via Voutetakis 2024 thesis): for `0 ≤ t ≤ Oq·T0`, else 0:

```
U_g(t) = a·t² − b·t³ ,   a = 27·AV/(4·Oq²·T0),   b = 27·AV/(4·Oq³·T0²)
```

Excite the tract with the derivative `U_g'(t) = 2a·t − 3b·t²`, or sample `U_g` and first-difference it. `Oq` (open quotient) ≈ 0.5 modal, 0.65–0.8 breathy, 0.3–0.4 pressed (UNVERIFIED ranges, standard).

**LF model** (CONFIRMED via the same thesis, eq. 4.1/4.6) — the differentiated flow, two segments:

```
U_g'(t) = E0·e^(a·t)·sin(π·t/Tp)                        0 ≤ t ≤ Te
U_g'(t) = −Ee/(ε·Ta)·[e^(−ε(t−Te)) − e^(−ε(T0−Te))]     Te ≤ t ≤ T0
E0 = −Ee·e^(−a·Te)/sin(π·Te/Tp) ,  Ta = (1 − e^(−ε(T0−Te)))/ε
```

Four parameters `{Ee, Tp, a, Ta}`; **Ta is the perceptually dominant one** — it sets the source spectral tilt (smaller Ta = sharper closure = brighter). Voice fitting procedure: LPC-inverse-filter one held vowel of the target voice, take the residual's cycles, grid-search (Ee,Tp,a,Ta) or (Oq,TL) to minimise spectral distance.

**Radiation** (CONFIRMED): the lip is a differentiator, `p[n] = u[n] − u[n−1]` (+6 dB/oct). One line; fixes the muffled 8-bit output.

**Spectral tilt** (CONFIRMED, espeak `pitch_synch_par_reset`): one-pole low-pass on the voice, `decay = 0.033·TLTdb`, `onemd = 1−decay`, TLTdb 0–24 dB (dB down at 3 kHz).

**Jitter / shimmer / flutter** (CONFIRMED definitions, Praat; CONFIRMED norms, 40 subjects):
- `jitter_local = mean|T_i − T_{i+1}| / mean(T_i)`; normal male 0.49–0.50%, female 0.59–0.62%.
- `shimmer_local = mean|A_i − A_{i+1}| / mean(A_i)`; normal ≈ 0.2 dB.
- Flutter (CONFIRMED, espeak): `ΔF0 = (flutter/50)·(F0/100)·[sin(π·12.7t)+sin(π·7.1t)+sin(π·4.7t)]·10`.
- Implementation: redraw the perturbation **once per glottal period** from an AR(1) process (`ρ=0.99`), `σ_j ≈ 0.004`, `σ_s ≈ 0.01`. Per-sample white noise (what we have) is the wrong statistic and reads as roughness.

**Aspiration / breathiness** (CONFIRMED, espeak `gen_noise`, Klatt 1980):
```
noise[n] = uniform(−1,1) + 0.75·noise[n−1]      (one-pole, ≈ −6 dB/oct)
voiced: noise[n] *= 0.5 over the second half of each period
frication: UFRIC = AF·noise        aspiration: added to the glottal source: +AH·noise
breathiness: voice += Aturb·nrand  only while n < nopen (open phase only)
```
Breathiness therefore **scales with Oq** — bigger open quotient means a longer noise window. Whisper = AV=0, F0=0, AH>0 through the formant network.

**Diplophonia / alternate-period skew**: ±2–5% amplitude on alternate periods (Klatt `skew`), removes the machine-gun regularity.

### B. Vocal tract — intelligibility and crispness

**Resonator / antiresonator** (CONFIRMED):
```
r = exp(−π·BW·T),  c = −r²,  b = 2·r·cos(2π·F·T),  a = 1 − b − c
y[n] = a·x[n] + b·y[n−1] + c·y[n−2]                     (unity DC gain)
antiresonator: a' = 1/a, b' = −b/a, c' = −c/a, delay line holds INPUTS
```
Coefficients are recomputed at least twice per pitch period; espeak interpolates a,b,c over 64-sample chunks (≈2.9 ms at 22 kHz) — **linear per-frame interpolation, never exponential smoothing across phoneme boundaries**.

**Full chain** (CONFIRMED): cascade branch = `Rnz(nasal zero) → Rnpc(nasal pole) → F6 → F5 → F4 → F3 → F2 → F1`; parallel branch = F1 and FNP driven by voicing+aspiration, F2–F6 driven by frication plus d/dt(voicing), outputs summed with alternating sign, plus bypass `AB·(frics + Δvoicing)`. Voicing feeds the cascade, frication the parallel branch (mutual exclusion).

**Nasal pole/zero** (CONFIRMED, Klatt Table III): FNP 270 Hz / BNP 100; FNZ 450 / BNZ 100 for /m/ and /n/; set FNZ = FNP for oral sounds (cancellation). /m/: F1 480, F2 1270, F3 2130. /n/: 480, 1340, 2470. /ŋ/: ≈480, 2200–2400, 2700–2800 (UNVERIFIED). Nasalised vowels: F1 +100 Hz, FNZ = (F1_new + FNP)/2.

**Locus-anchored transitions** (CONFIRMED cue literature): stop place is carried by the F2 transition *from a place-specific locus*, not by the burst alone — bilabial ≈720 Hz, alveolar ≈1800 Hz, velar ≈3000 Hz (Delattre 1955). Implement as straight-line ramps from the locus to the vowel target over **30–60 ms**, with manner-dependent time constants (stop 35 ms, sonorant 80 ms, fricative direct), and **no cross-boundary one-pole**.

**VOT and bursts** (CONFIRMED values): aspiration phase = `F0=0, AV=0, AH=60, B1=400`; release = `AH=0, B1→60, F0=target, AV=60`. English word-initial aspirated stops 60–100 ms; measured tables: p 28, t 39, k 43 ms simple initial; 47/65/70 ms pre-stressed; after /s/ 12/23/30 ms. A burst is an **instantaneous** AF step >50 dB (Klatt `PLSTEP`), 5–10 ms, never ramped.

**Fricative / burst spectra per manner** (CONFIRMED, Klatt Table III): parallel amplitudes A1–A6 + bypass AB. /s/: A6=52 at F6≈4.9 kHz, BW 1000. /ʃ/: A2=57, A3–A5≈46–48 (broad 1.8–4 kHz). /f/,/θ/: AB 48–57, weak. /p,b/ burst: wideband flat, AB=63. /t,d/: peak 2.5–4 kHz. /k,g/: peak ≈2–2.9 kHz. This replaces the single `noise_hz` per phoneme.

**Proximity corrections** (CONFIRMED): when two formants come within 50 Hz the parallel-branch amplitudes need correction, `A2COR=(F1/500)²/(F2/1500)`, up to +10 dB.

### C. Prosody — cadence

**Fujisaki command-response model** (CONFIRMED, Fujisaki 2004; α=3.0, β=20.0, γ=0.9 from Bu 2003):

```
log F0(t) = log Fb + Σ_i Ap_i·Gp(t−T0_i) + Σ_j Aa_j·[Ga(t−T1_j) − Ga(t−T2_j)]
Gp(t) = α²·t·e^(−αt)                       (impulse response, critically damped 2nd order)
Ga(t) = min(γ, 1 − (1+βt)e^(−βt))          (step response, clipped)
```

Phrase commands ≈200 ms before utterance start and major boundaries; accent commands start 40–50 ms before the accented vowel and end 40–50 ms before its end. Because every command shares α (resp. β), all commands collapse into **4 global one-pole state variables** — an O(1) per-frame recursion at 100 Hz, replacing our linear −18% declination. Closed-form kernel evaluation on the frame grid is exact and drift-free; the recursive form matches to O(α·Ts) ≈ 3% at 10 ms frames.

**PENTA / qTA syllable targets** (CONFIRMED, Xu et al. 2015) — the stronger option for English cadence:

```
f0(t) = (m·t + b) + (c1 + c2·t + c3·t²)·e^(−λt)
c1 = f0(0) − b ,  c2 = f0'(0) + c1·λ − m ,  c3 = (f0''(0) + 2c2λ − c1λ²)/2
```

Three parameters per syllable (target height `b`, target slope `m`, approximation strength `λ`), with the `(f0, f0', f0'')` state handed from syllable to syllable — carryover and delayed peak alignment emerge for free. Published result: predictive synthesis from ~78 learned parameters per speaker produced English prosody heard as **"just as natural as natural prosody"** (CONFIRMED). That is the direct evidence that the no-neural-net goal is reachable.

**ToBI accent → F0 target map** (CONFIRMED-mirror, citing Pierrehumbert 1980 / Ladd 2008), male base 110 Hz, range 80 Hz: `H*` = 0.85·range, `L*` = 0.15·range, `L+H*` low then high on the starred syllable, `L*+H` low then rise into the next; downstep `k=0.6` after each `!H*` with floor 0.25; `L-L%` falls to baseline, `H-L%` falls from ~0.8, `L-H%` rises to 0.8 with upstep 0.2; continuation rise +8 Hz at a comma, +5 Hz at a semicolon. Microprosody: after a voiceless onset ×1.08, after a voiced onset ×0.95 (Hombert).

**Duration rules** (CONFIRMED-mirror, Klatt 1976 / Allen 1987 / Wightman 1992 lineage): stress ×1.3 / ×0.8; polysyllabic shortening ×0.78; non-word-initial consonant ×0.7; cluster first member ×0.88, later ×0.78; pre-boundary lengthening by break index BI4 ×1.5, BI3 ×1.3, BI2 ×1.15; accent lengthening nuclear ×1.25, prenuclear ×1.15; final open-syllable vowel ×1.2; before voiced fricative ×1.6; before voiced stop ×1.2; before nasal ×0.85; pauses comma 150 ms, phrase 200 ms, terminal 300 ms. Rate compression with incompressibility floors (vowels 0.42, consonants 0.6), total cap 2.0×.

**Microprosody to add on top** (CONFIRMED): per-period jitter ≈0.4%, shimmer ≈1%, phrase-final breathiness (ASP −18 dB final vs −25 dB modal), Oq 0.6 phrase-finally vs 0.5 modal, ±3% alternate-period skew.

### D. Excitation and resynthesis — crispness without buzz

Three sub-techniques, all numpy, all reimplemented from published math:

1. **Analytic envelope → mel-cepstrum.** Sample our formant targets at 5 ms, build `|H(f)|` from the resonator magnitudes, fit `J=34` mel-cepstra through an 80-band mel filterbank (`c = DCT-II(log(mel energies))`). This is the CheapTrick smoothing effect (CONFIRMED: CheapTrick is an F0-adaptive envelope estimator) without needing analysis at all — we know the envelope analytically.
2. **Minimum-phase FIR from a magnitude** (CONFIRMED, Oppenheim 1969 homomorphic lifting):
```
c[n] = IFFT{log|H|} ; c_min = 2c[n] (0<n<N/2), c[0], 0 (N/2<n<N)
h_min = IFFT{ exp( FFT{c_min} ) }
```
Minimum phase puts all energy after the onset; a zero-phase filter puts energy before it, which is audible **pre-echo** on fricatives and bursts. WORLD deliberately uses min-phase for its OLA envelope filters (CONFIRMED).
3. **Mixed excitation + phase-locked OLA** (CONFIRMED, Morise & Shono APSIPA 2019): excitation = F0 pulses plus 5-band modified velvet noise (`k(m)=⌊m·Td + r1(m)⌋`, Td=4, balanced ±1 amplitudes, flat ±0.6 dB 100 Hz–Nyquist), each band scaled by our analytic aperiodicity (from the phone table: noise_amp/noise_hz → which bands). Overlap-add at 5 ms hop with the min-phase envelope FIR.
   For any STFT-domain work, use **Laroche–Dolson identity phase locking** with **75% overlap**: worst-case sidebands −51 dB at 75% Hann overlap vs **−21 dB at 50%** (CONFIRMED numbers) — 50% overlap with fractional shifts is the audible "phasiness".

**Measured cost** (CONFIRMED, run locally on 3 s @22.05 kHz): framed STFT 3.2 ms, 80-mel matmul 17 ms, batched ISTFT+OLA 20 ms, min-phase IR 1.5 ms, 5-band FFT filtering 15 ms, mel-cep reconstruction 0.24 ms, envelope 0.07 ms, pulse train 0.5 ms → **35–70 ms per sentence locally**; NAS cores are 3–6× slower single-core → **0.1–0.4 s**, which meets <200 ms for typical ≤2 s lines.

### E. Post chain — polish

All implementable as biquads/FFT (MATH, RBJ cookbook): high-pass 80–100 Hz; presence +3 dB @3 kHz Q 0.8; warmth +2 dB @200 Hz; **LTAS matching** — compute the long-term average spectrum of a human reference voice, cepstrally lifter the log ratio, apply as a minimum-phase EQ; de-esser (5–9 kHz split-band compressor, threshold −28 dBFS, ratio 3:1, attack 1 ms, release 40 ms); loudness to −16 LUFS via ITU-R BS.1770 K-weighting; optional CELP-style formant post-filter `H(z)=A(z/γ1)/A(z/γ2)` with γ1 0.9–0.95, γ2 0.6–0.8 for sharper formants.

---

## 4. Hybridization options, ranked

The user's question — "can we hybridize techniques for better synthesis?" — has five honest answers, ordered by cost and by what each buys:

### H0 — pure DSP: Klatt chain + prosody (recommended first)
numpy only, no training, no data. Layers A+B+C+E, with the existing parallel-bank renderer. Buys: intelligibility (consonant place, nasals, fricatives), cadence (Fujisaki/qTA + duration rules), voice quality (OQ/tilt/jitter/aspiration). Cost: ~1–2 weeks of work, <10 ms compute per 10 s of speech. Risk: none technical; the risk is that "formant speech" has a ceiling no DSP trick crosses — it will sound like a very good 1990s Klatt synthesizer, not a person.

### H1 — WORLD-style resynthesis driven by our parameters (recommended second)
Layer D on top of H0. Buys: crispness — ripple-free envelope, minimum-phase transients, mixed excitation instead of a bare pulse train. This is where the current "buzzy" character dies. Cost: ~1 week; measured 35–70 ms/sentence locally. Still no training, still fully ours. Note: this is *synthesis-driven* WORLD, not analysis-driven — we never need D4C/CheapTrick analysis because the front end already knows F0, envelope and aperiodicity. That is a genuine simplification nobody in the literature needed to make, because they were all resynthesising real recordings.

### H2 — our own small NSF/SiFi-GAN vocoder (the "sounds like a person" path)
Train a source-filter GAN ourselves, conditioned on **our** parameters (`cf0`, `vuv`, 40-coefficient mel-cepstrum, 5-band aperiodicity, `sin(F0)`), generator ≈1–2 M params (HiFi-GAN V2 class; stride product 240 = 5·4·3·4 at 24 kHz/10 ms), losses = LS-GAN + 45·L1(mel) + 1·L_reg(LPC residual) (CONFIRMED, SiFi-GAN recipe). Published precedent: SiFi-GAN conditioned purely on WORLD parameters reaches MOS **3.90** vs 2.98 for plain WORLD (CONFIRMED). Deploy generator-only ONNX on the NAS: RTF ≈0.2–0.6 → 0.6–1.5 s per 3 s sentence, i.e. slower than Piper but acceptable, or pre-render.
Critical risk (CONFIRMED): GAN vocoders degrade badly on synthetic front-end features and **must** be fine-tuned on features produced by our own front end — feeding them real-speech mels and then our envelopes is the classic domain-gap failure. Mitigation: train on pairs built by running real recordings through our own analysis (LPC inverse filter → our envelope/aperiodicity representation), so the vocoder learns "render these parameters", not "invert a real mel".

### H3 — differentiable formant filters (HiFi-Glot class)
The 2024–2026 research answer to exactly this question: a neural excitation (NSF-HiFiGAN) driving **differentiable all-pole filters** whose coefficients come from interpretable formant parameters, filtered in the STFT domain `X = H(z)·E(z)`, with a log-envelope loss plus adversarial losses (CONFIRMED, HiFi-Glot 2024/2026). It achieves better formant control and naturalness than Praat-style LPC manipulation. This is the technically strongest option and the most expensive: GPU training on 1664 h of speech in the paper's setup, four discriminators, and a 50-coefficient LAR parameterisation. For us it is a research branch to consider only after H2 works, and only if we want formant-accurate voice morphing rather than "good voices".

### H4 — voice conversion (kNN-VC / OpenVoice / RVC) — rejected
Taking a SAM waveform and converting its timbre to a human voice keeps timing 1:1 but throws away our formant precision, needs a HuBERT-class content encoder (30–90 M params, out-of-domain on buzzy SAM audio), and is a heavy dependency. The same timbre transfer is achieved by H2 without a content encoder. Keep VC only as an offline experiment if H2 fails.

**Summary ranking:** H0 → H1 → H2 → (H3 research) — H4 rejected. H0+H1 are ours, numpy, deterministic, <200 ms, and fix the two complaints that matter (cadence, crispness). H2 is the only route to genuinely human timbre and it is still "ours" because we train it.

---

## 5. Voice types: how many voices from parameters

Voice identity = a parameter vector over layers A–C. This is cheap, deterministic, and gives more distinct voices than Piper's model zoo:

| Axis | Parameter | Range | Perceived effect |
|---|---|---|---|
| Body size | vocal-tract length scale on F1–F5 | 0.85–1.20 | child ↔ large adult |
| Pitch | F0 base (`pitch` knob → Fb) | 60–300 Hz | gender/age character |
| Voice quality | Oq (open quotient) | 0.3 pressed → 0.8 breathy | tense ↔ soft/breathy |
| Brightness | Ta / TLTdb tilt | 0–24 dB | warm ↔ sharp |
| Breath | AH aspiration level | −30…−12 dB | clean ↔ whispered |
| Regularity | jitter/shimmer/flutter | 0–1.5% | steady ↔ aged/unstable |
| Rate & rhythm | speed + duration-rule scale | 0.7–1.5× | calm ↔ manic |
| Nasality | FNP/FNZ engagement | 0–100% | oral ↔ nasal |
| Character FX | existing `computer_voices.py` (ring mod, vocoder, crush, blips) | unchanged | robot/chip/Dalek |

Named presets map to vectors: `narrator` (Oq 0.5, tilt 8 dB, jitter 0.3%), `child` (VTL 1.12, F0 250), `giant` (VTL 0.85, F0 75, tilt 14 dB), `whisper` (AV 0, AH −12), `aged` (jitter 1.2%, shimmer 2%, flutter 1%), `droid` (SAM 1982 port + crush), etc. No new models, no downloads.

---

## 6. Data worth researching and collecting (what we have not looked at)

We do not need a corpus to *train* anything in H0/H1 — but we need statistics and reference voices, and we need one licence-clean corpus if we go to H2.

| Dataset | Size / licence | What to extract for us |
|---|---|---|
| **LJSpeech** | 13,100 clips, 24 h, one speaker, 22.05 kHz, **public domain** (CONFIRMED) — a 500-clip slice is ≈100 MB | per-phone duration statistics, F0/CPP reference targets, LTAS target, H2 training pairs. **Study first** |
| **LibriTTS** | 585 h multi-speaker, 24 kHz, **CC BY 4.0** (CONFIRMED); dev-other is 924 MB | second reference set, multi-accent phone statistics |
| **VCTK 0.92** | 110 accented speakers, 10.94 GB, **CC BY 4.0** (attribution; licence text flagged UNVERIFIED) | per-speaker F0/VTL/voice-quality distributions → voice-type presets grounded in real speakers |
| **CMU ARCTIC** | ~2–3 h per voice, 16 kHz, page carries no explicit licence (CONFIRMED absence) — research use | clean single-speaker references for LTAS matching |
| **CMUdict** | 134k words → ARPAbet + stress, 3 MB, **"research or commercial, unrestricted"** (CONFIRMED) | syllable counts for the rate/nPVI metric and a licence-clean lexicon for a shipped game |
| **Common Voice** | **CC0** (CONFIRMED), per-language GBs | only if we need accent coverage |
| **Mocha-TIMIT / Boston University Radio News** | research licence, application/gated (UNVERIFIED terms) | ToBI label statistics, break-index → lengthening calibration, accent shapes for qTA fitting |
| **espeak-ng data** | **GPL-3.0** (CONFIRMED) | study `.phsrc`/intonation files for parameter values only; never vendor into our tree |

What we extract, per corpus: (1) per-phone duration means/percentiles by context → replace hand-set multipliers; (2) F0 contours fitted with the Fujisaki model → realistic `Ap`/`Aa` magnitudes; (3) LTAS → post-EQ target; (4) formant statistics by speaker → VTL/F0 preset ranges; (5) for H2, aligned (parameters → audio) pairs. **No audio is shipped or redistributed** — only statistics and self-trained weights, which keeps the licence surface small.

Also worth researching that we have not: Klatt 1980 §IV ΔF0 rule list and Klatt 1987 intelligibility percentages (both PDFs are image scans — OCR them); locus-equation coefficient tables (Childers & Hanna 1986, Johnson 1989); exact CheapTrick/D4C internals (paywalled — we avoid needing them in H1); the /ŋ/ Klatt Table III row.

---

## 7. Measurement harness (so improvements are ranked, not guessed)

One script, six fixed sentences, one human reference recording per sentence (same text). Five metrics, all reimplementable from the equations (no licensed tools), plus the repo's own regression guards:

| # | Metric | What it answers | How to compute | Pass target |
|---|---|---|---|---|
| 1 | **ASR CER** — Whisper `tiny.en`/`base.en` as a black-box oracle (MIT code and weights; used only as a measuring instrument, never part of the synth) | Is it intelligible? | transcribe → normalise → Levenshtein / len(target) | ≤5% pass, 5–15% marginal, >15% fail. Compare against the reference recording's CER as the floor (Whisper's own floor is ~3–5% on clean read speech) |
| 2 | **DTW-MCD** — K=24 mel-cepstra, c0 excluded, 25 ms/10 ms frames, 50 Hz–7.8 kHz, pre-emphasis 0.97 | Are the formants/envelope right? | `MCD = (10/ln10)·√(2·Σ(Δc_k)²)` averaged over DTW-aligned frames (Sakoe-Chiba ±15) | ≤4 dB excellent, ≤6 dB good, >8 dB mush (common TTS practice, UNVERIFIED thresholds) |
| 3 | **CPP** (cepstral peak prominence) | Crisp voice, or breathy hiss? | 40 ms Hann frames → 4k log-magnitude FFT → cepstrum → peak in the 1/400–1/60 s quefrency band above a regression line; voiced frames only | ≥0.75× the human reference's CPP on the same sentence; absolute <6 dB (Praat scale) flags audible breathiness |
| 4 | **F0 semitone RMSE + span** | Cadence and pitch contour vs a human | YIN both sides → DTW-align voiced frames → RMSE in semitones (`12·log2(f/125)`); span = P95−P5 | RMSE ≤2 st pass, ≤3.5 st marginal; `|span_synth − span_ref| ≤ 3 st` (monotone-collapse guard — the "robotic" signature) |
| 5 | **Rate + nPVI/rPVI** | Human-like timing? | syllable count from CMUdict / (voiced + voiced-gap duration, pauses >150 ms excluded); vocalic intervals → `nPVI = mean_k |d_k − d_{k+1}|/((d_k+d_{k+1})/2)·100`; rPVI = mean absolute successive differences; same code path on the reference | rate within ±15%, `|ΔnPVI| ≤ 10 pts`, `|ΔrPVI| ≤ 15%` of reference |

**Regression guards** (already in the repo, keep them at zero cost): energy >5 kHz <1% (faithful 8-bit path exempt — its grit is the identity), sample steps over 25% of peak = 0, ZCR within 1.5× the preset's historical value, harmonicity ≥ preset baseline, HNR in the 10–15 dB normal-voice band (7.4 dB is the 95% lower CL for normal voices, CONFIRMED), spectral flatness <0.3 on voiced frames.

**Deliberately excluded:** PESQ/POLQA (intrusive, licensed, trained on codec artifacts not synth buzz), SI-SDR (separation metric, alignment-fragile), UTMOS/NISQA/DNSMOS as *installed* judges (they need human-MOS labels and SSL backbones — their *feature recipes* are worth studying, NISQA's four diagnosis heads map exactly onto our defect vocabulary: hiss→Noisiness, band-limiting→Coloration, clicks→Discontinuity).

**Honest baseline statement:** no published MOS/STOI exists for SAM or eSpeak-NG (searched, absent), and no rigorous MOS for Piper/Kokoro was read this pass. Our baseline is therefore the repo's own measured table plus the five numbers above; the realistic ceiling for a formant synth before any learned model is eSpeak-class intelligibility (CER ≤5%) and MOS ≈2.5–3.5 (INFERENCE).

---

## 8. Phased roadmap with acceptance checks

Every phase lands on its own branch, measured before/after, and the previous phase's numbers must not regress.

**P0 — harness first (½ day).** Add `sam-demo/measure.py` computing the five metrics of §7 over a fixed sentence set (pangram, the Drift King line, the office line, a 12-word paragraph). Baseline table committed. No synth change ships without a before/after row.
Acceptance: metrics reproducible run-to-run (same seed → identical bytes).

**P1 — Klatt source and tract (3–5 days).** New module `sam-demo/klatt.py`, `engine="klatt"` alongside `engine="sam"`: radiation differentiator, cascade F1–F5, KLGLOTT88 source with Oq, tilt one-pole, per-period jitter/shimmer/flutter, aspiration bus, nasal pole/zero, per-manner fricative/burst tables, VOT machinery, locus-anchored transitions with manner-dependent time constants, and removal of the single 15 ms constant.
Acceptance: vowel harmonicity ≥ 0.96 (matches our best measured 0.9621); nasal minimal pairs /m n ŋ/ distinguishable in an ABX by us; stop place identification ≥ 90% on p/t/k + b/d/g in a 3-listener informal test; Whisper WER on the pangram no worse than the current engine; high-band energy ≤ 1% on voiced material.

**P2 — prosody (2–3 days).** `sam-demo/prosody.py`: Fujisaki F0 (α=3, β=20, γ=0.9) with phrase/accent commands derived from punctuation and a stress assignment, duration rule stack, ToBI accents, pauses. Per-frame output feeds either engine.
Acceptance: F0 RMSE against a hand-fitted Fujisaki target < 1 semitone; measured durations within 10% of the rule stack; F0 range and pause distribution match the human reference (LJSpeech sentence) within 15%; blind A/B — listeners prefer prosody-on over prosody-off.

**P3 — excitation/resynthesis (3–5 days).** `sam-demo/worldish.py`: mel-cep envelope fit, min-phase FIR, mixed excitation with 5-band MVN, phase-locked 75%-overlap OLA. Selectable as a render path.
Acceptance: HNR in the 10–15 dB band (normal-voice norm), no pre-echo measurable on bursts (energy before onset < −40 dB relative), harmonicity ≥ 0.96, total synthesis ≤ 200 ms for a 2 s line on the NAS-class core, and A/B preference over P2 output.

**P4 — voice types + post chain (2 days).** Parameter presets, LTAS match, de-esser, loudness; expose through the service as `engine="synth"` with `voice_accent` presets (keeping every existing alias working — the resolver falls back to Piper for unknown names).
Acceptance: 8 presets measurably distinct (F0 median, spectral centroid, Oq-derived HNR differ by set margins); service smoke: `/api/v1/tts/synthesize` returns a valid WAV in < 300 ms for each preset; `/health` unchanged; no existing alias changes output.

**P5 — optional H2 vocoder (weeks, GPU).** Only after P1–P4 are measured. Train on licence-clean data, ship generator ONNX, A/B against P3.
Acceptance: UTMOS and Whisper WER both improve over P3; latency stated honestly; licence audit of training data and weights.

---

## 9. Risks

| Risk | Mitigation |
|---|---|
| Formant synthesis has a naturalness ceiling no DSP crosses | Measure against the P1/P2/P3 numbers; if P3 still sounds synthetic to us, that is the signal to spend on P5, not to keep polishing DSP |
| GAN vocoder domain gap on our synthetic envelopes (CONFIRMED failure mode in the literature) | Train on pairs generated by our own analysis path; condition on analytic parameters, not mel alone |
| The NAS core is ~3–6× slower than this workstation (INFERENCE from Piper 1.1 s vs 0.18 s) | Keep the DSP path numpy-only and batched; measure on the NAS before promising latency; pre-render game lines |
| Removing the 15 ms smoothing may make some transitions click | Manner-dependent time constants plus per-frame linear coefficient interpolation, and the discontinuity counter in the harness catches it |
| Licensing of training data / voices | Statistics only from PD/CC-BY corpora; train H2 ourselves; never ship third-party weights |
| We accidentally re-derive SAM's 8-bit grit as a "defect" | Keep `engine="sam"` untouched as the retro preset; new engine is a separate path |

---

## 10. Decisions needed

1. **Scope of "our own":** is H2 (train our own vocoder) in scope now, or after H0/H1 are measured? *Recommended default: H0/H1 now, decide H2 after P3 numbers.*
2. **Where the new engine ships:** service-side `engine="synth"` aliases (works for office/trading/HTML-Notes immediately) versus pre-rendered game lines only. *Recommended: service-side, with unknown aliases still falling back to Piper.*
3. **Piper's role:** keep it as the default human voice and use our engine for character voices, or eventually replace it? *Recommended: keep Piper; our engine is the character/cadence engine, not a Piper replacement.*
4. **Data:** do we accept CC-BY-4.0 / PD corpora (LJSpeech, VCTK) for statistics and later training? *Recommended: yes; statistics now, training only at P5.*
5. **NAS latency budget:** is 0.1–0.4 s/sentence acceptable for the new engine (vs Piper's 1.1–1.3 s)? *Recommended: yes, it is faster than the current default.*

---

## 11. Sources

Primary, read this session (see the five `local://*.md` reports for the full list):
- Klatt 1980 cascade/parallel synth: `fon.hum.uva.nl/david/ma_ssp/doc/Klatt-1980-JAS000971.pdf` (scan), content CONFIRMED via `github.com/ctoth/Qlatt` notes.
- Klatt & Klatt 1990 KLGLOTT88, LF model (Fant–Liljencrants–Fant): CONFIRMED via Voutetakis 2024, "Aliasing-free Models of the Glottal Source".
- espeak-ng `klatt.c`/`klatt.h` — algorithm reference only, nothing copied.
- Oxford Phonetics (Coleman): Cooper 1952 bursts/transitions, Delattre 1955 loci 720/1800/3000 Hz, Lisker & Abramson categorical VOT.
- Fujisaki 2004 (Speech Prosody), Bu et al. 2003 (α=3, β=20, γ=0.9).
- Xu et al. 2015, "Explaining the PENTA model" — qTA naturalness result.
- Praat jitter/shimmer definitions; Felippe et al. 2006 normal-voice norms (jitter 0.49–0.62%, HNR 9.6–11 dB).
- WORLD: Morise 2016; Morise & Shono APSIPA 2019 (min-phase OLA, modified velvet noise, 5-band aperiodicity).
- Oppenheim 1969 homomorphic minimum-phase; Laroche & Dolson 1999 phase locking (−51 dB @75% vs −21 dB @50%).
- HiFi-Glot 2024/2026 (differentiable resonant filters + neural excitation) — the H3 reference.
- SiFi-GAN 2022 (source-filter HiFi-GAN on WORLD features, MOS 3.90 vs 2.98).
- Shannon et al. 1995 (envelope-only intelligibility: 8 channels ≈ ceiling).
- Qlatt `frontend.yaml` (CONFIRMED-mirror of Klatt 1975/1976, Allen 1987, Wightman 1992, Pierrehumbert 1980, Hombert 1979, Fant 1997 constants).
