"""End-to-end and unit tests for the from-scratch formant synthesis engine."""
from __future__ import annotations

import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "sam-demo"))

from app import synth_manager
from app.synth import frames, klatt, post, presets, prosody, reciter, worldish
import measure


class TestSynthEngine(unittest.TestCase):
    def test_reciter_and_prosody_edge_words(self) -> None:
        for text in (
            "To be or not to be.",
            "He and she said we can.",
            "Dr Smith watched TV.",
            "Hmm, shh!",
            "The quick brown fox jumps over the lazy dog.",
        ):
            phs = reciter.reciter(text)
            self.assertGreater(len(phs), 0)
            durs, f0 = prosody.plan(phs, presets.voice("narrator"))
            self.assertEqual(len(durs), len(phs))
            self.assertGreater(f0.size, 10)
            self.assertTrue(np.all(np.isfinite(f0)))

    def test_connected_speech_cadence_and_flapping(self) -> None:
        # 1. Connected speech: running sentences carry no inter-word "_"
        #    gap phones; words link across boundaries like human speech.
        phs = reciter.reciter("I am a little robot.")
        self.assertTrue(all(p.name != "_" for p in phs))

        # 2. Intervocalic flapping: "little" -> L IH DX AX L and
        #    "automatic" -> AO DX AX M AE DX IH K surface alveolar taps.
        self.assertIn("DX", [p.name for p in reciter.reciter("A little automatic robot.")])

        # 3. Function words reduce to stress 0; content words keep a
        #    stressed vowel.
        text = "I come from the little robot."
        words = text.strip(".").split()
        function_words = {"a", "the", "to", "for", "from"}
        content_words = {"I", "come", "little", "robot"}
        by_word: dict[int, list[reciter.Phone]] = {}
        for p in reciter.reciter(text):
            by_word.setdefault(p.word, []).append(p)
        for idx, word_phones in by_word.items():
            word = words[idx]
            if word in function_words:
                self.assertTrue(all(p.stress == 0 for p in word_phones),
                                f"function word {word!r} carries stress")
            elif word in content_words:
                self.assertTrue(any(p.stress > 0 for p in word_phones
                                    if p.name in reciter.VOWELS),
                                f"content word {word!r} has no stressed vowel")

        # 4. Unstressed schwa shortens below the old flat 60 ms robotic
        #    floor: Klatt phone-specific d_min durations are active.
        durs, _f0 = prosody.plan(reciter.reciter("I am a little robot."),
                                 presets.voice("narrator"))
        unstressed_ax = [d for p, d in zip(phs, durs)
                         if p.name == "AX" and p.stress == 0]
        self.assertTrue(unstressed_ax)
        self.assertTrue(all(d < 0.060 for d in unstressed_ax),
                        f"unstressed AX durations not below 60 ms: {unstressed_ax}")

    def test_frames_invariants_and_whisper_gating(self) -> None:
        phs = reciter.reciter("Watch out for that dog.")
        fr = frames.build(phs, presets.voice("narrator"))
        self.assertTrue(np.all(fr.formants > 0.0))
        self.assertTrue(np.all(fr.bws > 0.0))
        self.assertTrue(np.all(np.diff(fr.formants, axis=1) >= 119.9))
        self.assertTrue(np.array_equal(fr.av > 1e-4, fr.f0 > 0.0))

        fr_w = frames.build(phs, presets.voice("whisper"))
        self.assertTrue(np.all(fr_w.av == 0.0))
        self.assertTrue(np.any(fr_w.ah == 0.0))
        self.assertTrue(np.any(fr_w.ah > 0.5))

    def test_klatt_and_worldish_voice_controls_and_steady_vowel(self) -> None:
        phs = [reciter.Phone("AA", stress=1, syl=0, word=0, boundary=4)]
        stub = type("S", (), {"plan": staticmethod(lambda p, v: ([1.0], np.full(200, 120.0)))})
        fr = frames.build(phs, {"pitch": 88, "speed": 72}, prosody=stub)
        y = klatt.render(fr, sr=22050, rng=np.random.default_rng(0))
        win = int(round(0.046 * 22050))
        seg = y[len(y) // 2 : len(y) // 2 + win] * np.hanning(win)
        spec = np.abs(np.fft.rfft(seg))
        freqs = np.fft.rfftfreq(win, 1.0 / 22050)
        pks = np.array([spec[(freqs >= fk - 150) & (freqs <= fk + 150)].max()
                        for fk in (730, 1090, 2440, 3400, 4200)])
        err_db = np.abs(20.0 * np.log10((pks / pks[0]) / np.array(frames.FORMANT_AMP)))
        self.assertTrue(np.all(err_db < 1.0), f"steady /AA/ peak error dB: {err_db}")

        sent = reciter.reciter("Hello world.")
        base = presets.voice("narrator")
        yk0 = klatt.render(frames.build(sent, base), rng=np.random.default_rng(0))
        yw0 = worldish.render(frames.build(sent, base), rng=np.random.default_rng(0))
        for key, val in (("oq", 0.75), ("tilt_db", 12.0), ("f4_gain", 0.4), ("f5_gain", 0.3)):
            v2 = presets.override(base, **{key: val})
            yk = klatt.render(frames.build(sent, v2), rng=np.random.default_rng(0))
            yw = worldish.render(frames.build(sent, v2), rng=np.random.default_rng(0))
            self.assertGreater(float(np.max(np.abs(yk - yk0))), 1e-3)
            self.assertGreater(float(np.max(np.abs(yw - yw0))), 1e-3)

    def test_post_conditioning_and_all_presets_clean(self) -> None:
        sr = 22050
        x = np.ones(4096)
        y = post._stft_apply(x, sr, np.ones(2049))
        self.assertLess(float(np.max(np.abs(y - x))), 1e-12)

        for name in presets.names():
            clip = synth_manager.render("Hello world.", name)
            g = measure.guards(clip, synth_manager.SAMPLE_RATE)
            self.assertEqual(g["crackle"], 0, f"{name} had crackle={g['crackle']}")
            self.assertLessEqual(g["peak"], 0.981)

        sam_audio = synth_manager.render(
            "SAM. I am a little robot. I come from 1982.", "SAM")
        g_sam = measure.guards(sam_audio, synth_manager.SAMPLE_RATE)
        self.assertGreaterEqual(g_sam["harm"], 0.85)
        self.assertGreaterEqual(g_sam["hnr_db"], 8.0)
        self.assertEqual(g_sam["crackle"], 0)
        if hasattr(measure, "calibrate"):
            cal = measure.calibrate(sam_audio, synth_manager.SAMPLE_RATE)
            self.assertGreaterEqual(cal["spec_hnr_db"], 14.0)
            self.assertGreaterEqual(cal["comb_lo_db"], 30.0)
            self.assertLessEqual(cal["valley_floor_db"], -18.0)

    def test_http_endpoints(self) -> None:
        from fastapi.testclient import TestClient
        import main

        client = TestClient(main.app)
        r_v = client.get("/api/v1/tts/synth-voices")
        self.assertEqual(r_v.status_code, 200)
        self.assertEqual(len(r_v.json()["voices"]), 13)

        r_s = client.post(
            "/api/v1/tts/synthesize",
            json={"text": "To be or not to be.", "engine": "synth", "voice_accent": "narrator"},
        )
        self.assertEqual(r_s.status_code, 200)
        self.assertTrue(r_s.content.startswith(b"RIFF"))


if __name__ == "__main__":
    unittest.main()
