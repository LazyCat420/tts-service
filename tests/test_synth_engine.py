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
