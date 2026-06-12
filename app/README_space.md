---
title: tiny-kws — edge keyword spotting
emoji: 🎙️
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 6.18.0
app_file: app.py
pinned: false
license: mit
---

# tiny-kws — live keyword spotting demo

Say **yes / no / up / down / left / right / on / off / stop / go** into your
microphone; a 119K-parameter (~0.5 MB) depthwise-separable CNN classifies the
clip on free CPU and shows the log-mel spectrogram it "sees". Anything else
should come out as *unknown*; saying nothing should come out as *silence*.

Microphone audio (48 kHz, often stereo) is converted to mono, resampled to
16 kHz, and cropped to its highest-energy 1-second window before
classification — skipping any of those steps silently breaks accuracy.

- Code + training pipeline: https://github.com/priyadeepjaiswal9c/tiny-kws
- Trained on Google Speech Commands v2 (Warden 2018, arXiv:1804.03209,
  CC-BY-4.0), official 12-class benchmark splits.
- If you deny mic access, use the bundled example clips instead.
