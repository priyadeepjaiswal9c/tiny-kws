# LEARNING.md — every concept in tiny-kws, explained

This is the interview-prep companion to the code. It follows the data:
microphone → waveform → spectrogram → CNN → probabilities → metrics.
Sections marked **[S&S]** connect directly to Signals & Systems coursework.

---

## 1. The task: keyword spotting (KWS)

Keyword spotting is the "small ears" problem: detect a handful of spoken
commands ("yes", "stop", "go"…) cheaply enough to run **always-on, on-device**
— think "Hey Siri" wake-word chips, not data-center speech recognition. The
constraints that make it interesting are *edge* constraints: tiny memory
(our model: ~0.48 MB), low latency, low power. This is why we report
parameter count and CPU latency, not just accuracy.

**The 12-class benchmark**: Google Speech Commands v2 contains 105,829
one-second clips of 35 words. The standard benchmark (Warden 2018,
arXiv:1804.03209) uses 10 keywords as classes, plus:
- **unknown** — clips of the *other* 25 words. A real system mostly hears
  words it should ignore; without this class the model would map every sound
  to its nearest keyword.
- **silence** — random 1-second crops of background-noise recordings. The
  model must know when *nothing* was said.

So the model answers: *"which keyword, or was it some other word, or nothing?"*

## 2. Digital audio: sampling and Nyquist **[S&S]**

A microphone produces a continuous voltage x(t); an ADC samples it every
T = 1/16000 s. The **Nyquist–Shannon theorem** says a 16 kHz sample rate can
only represent frequencies below **8 kHz**. That's fine for speech: the
phonetic information lives mostly under ~8 kHz (telephone audio survives at
4 kHz). This is also why our mel filterbank stops at 7600 Hz — there is
literally nothing above 8 kHz in the signal, and content near the edge is
distorted by the anti-aliasing filter.

**Where this bit us in practice**: browser microphones record at 48 kHz.
Feeding 48 kHz samples into a model trained on 16 kHz audio means every
"frame" covers 1/3 of the time the model expects — features are garbage and
accuracy silently collapses. The demo app resamples 48 kHz → 16 kHz first
(a proper polyphase resampler, which low-pass filters before decimating —
exactly the anti-aliasing requirement from S&S; naive decimation by taking
every 3rd sample would alias 8–24 kHz content down into the speech band).

## 3. From waveform to spectrogram: the STFT **[S&S]**

A 1-second clip is 16,000 numbers. Raw waveforms are a hard input for small
models: the same word shifted by 1 ms is a completely different vector. The
fix is the **Short-Time Fourier Transform**: slide a 25 ms window (400
samples) along the signal in 10 ms hops (160 samples) and take an FFT of
each window. Result: a 2-D image, frequency × time — a **spectrogram**
(we use power, |X|²).

- Why 25 ms? Speech is only *locally* stationary: within ~25 ms a vowel's
  pitch and formants are stable, so the Fourier spectrum is meaningful.
- Why hop 10 ms? Overlapping windows so we don't miss transients (e.g. the
  burst of a "t").
- 1 s / 10 ms ≈ 100 → our clips become 101 frames (the extra frame comes
  from torch's centered padding).
- **[S&S]** This is the classic time–frequency resolution trade-off (the
  uncertainty principle): longer windows → finer frequency resolution but
  blurrier timing, and vice versa. 25 ms is the speech-processing sweet spot.

## 4. Mel filterbanks and log compression

The FFT gives 201 linearly-spaced frequency bins, but **human hearing is not
linear in Hz** — we resolve 100 Hz vs 200 Hz easily, 7000 vs 7100 Hz not at
all. The **mel scale** is a perceptual frequency axis (roughly logarithmic
above 1 kHz). A **mel filterbank** is a bank of 64 triangular bandpass
filters, narrow at low frequencies and wide at high ones; multiplying the
power spectrum by it pools 201 bins down to 64 perceptually-spaced bands.

Then we take the **log**: `log(mel + 1e-6)`. Two reasons:
1. Loudness perception is logarithmic (decibels!). A whisper and a shout
   differ by ~10⁴ in power but should not differ by 10⁴ in feature values.
2. It compresses dynamic range so the network sees well-conditioned inputs.

Final feature: a **64 × 101 log-mel spectrogram** — effectively a small
grayscale image of the word. (Classical pipelines add a DCT step to get
MFCCs; CNNs work as well or better on log-mels directly, so we stop here.)

We normalize features with the **training set's** global mean/std. Using
test-set statistics would leak information; using per-clip stats would make
quiet/loud clips inconsistent. The mean/std are stored inside the checkpoint
so the demo applies *exactly* the same normalization.

## 5. Why a CNN on a spectrogram?

A spoken keyword is a local pattern in time–frequency: formant ridges,
energy bursts, transitions. Two properties make convolutions the right tool:
- **Locality**: a 3×3 kernel sees a small time–frequency patch; phonetic
  events are local.
- **Translation equivariance**: a conv filter detects its pattern wherever
  it occurs. A word said 100 ms later (or by a higher-pitched speaker,
  shifted slightly along the mel axis) still triggers the same filters.
  This is also why our random time-shift augmentation is consistent with
  the architecture.

**[S&S]** A conv layer literally computes 2-D discrete convolutions
(cross-correlations, technically — flipped kernel) of learned FIR filters
with the input, followed by a pointwise nonlinearity. "Filter banks with
learned coefficients, stacked" is an accurate one-line description of a CNN.

## 6. Depthwise-separable convolutions — the "tiny" in tiny-kws

A standard conv layer mapping C_in=128 → C_out=128 channels with 3×3 kernels
costs 128·128·3·3 ≈ **147k parameters per layer**. Our whole model is 119k.

The depthwise-separable factorization (MobileNet; DS-CNN for KWS, Zhang et
al. 2017, arXiv:1711.07128) splits this into:
1. **Depthwise 3×3**: one 3×3 filter *per channel*, no channel mixing —
   C·3·3 = 1,440 params (C=160).
2. **Pointwise 1×1**: a 1×1 conv that mixes channels, no spatial extent —
   C·C = 25,600 params.

Cost ratio vs standard conv: (3·3·C + C²) / (3·3·C²) = 1/9 + 1/C ≈ **~8.4×
fewer parameters** at C=160, with a small accuracy cost. Intuition: spatial
filtering and channel mixing are *separate jobs*; doing them jointly (a full
3-D kernel per output channel) is mostly redundant.

Our DSCNN: a 10×4 standard-conv stem (stride 2) → 4 DS blocks (one with
stride 2) → global average pooling → dropout → a single 160→12 linear layer.
Total: **119,372 parameters, 0.48 MB in fp32**.

Other pieces, one line each:
- **BatchNorm** standardizes each channel over the batch, then rescales with
  learned parameters — stabilizes/accelerates training; at inference it
  collapses into a fixed affine transform (zero extra cost).
- **ReLU** max(0,x): the nonlinearity; without it stacked convs would
  collapse into one linear filter **[S&S: LTI systems compose into one LTI
  system — nonlinearity is what breaks that]**.
- **Global average pooling** averages each channel's feature map to one
  number → the classifier sees a 160-dim "which patterns occurred" summary,
  independent of exactly where, and the model has no giant dense layer.
- **Dropout (0.2)** randomly zeros features during training so the
  classifier can't over-rely on any single one — regularization.

## 7. Train / validation / test discipline

- **Train** (~38k clips): gradients computed here, and *only* here.
- **Validation** (~3.7k): no gradients; used to pick hyperparameters and to
  select the best epoch's checkpoint. Because we make choices based on it,
  it is mildly "spent" — its accuracy is an optimistic estimate.
- **Test** (4,890): touched exactly once per final model, by `evaluate.py`.
  Every number we publish comes from this split.

**Why the official split matters — two reasons:**
1. **Speaker disjointness.** The split files assign by *speaker hash*, so no
   speaker appears in both train and test. A random shuffle of clips would
   put the same person's "yes" in both — the model could partly memorize
   voices, inflating accuracy. Our test number answers the question that
   matters: does this generalize to *people it has never heard*?
2. **Comparability.** Everyone who reports "12-class Speech Commands v2"
   accuracy uses these exact files (we evaluate on the official curated
   test archive, 4,890 WAVs). Our number can sit honestly in the same table
   as Warden's 88.2% CNN baseline or the 98.6% Keyword Transformer SOTA.

## 8. Data augmentation

We train on randomly perturbed copies of each clip — the label is unchanged,
so we're teaching invariances instead of collecting more data:
- **Random time shift ±100 ms**: words aren't centered in real recordings;
  also matches the CNN's translation equivariance (§5).
- **Background-noise mixing**: with probability 0.8, add a random crop of a
  real noise recording (dishes, bike, white/pink noise) at volume U(0, 0.1).
  Real microphones are never in silence.

Augmentation happens on the **waveform**, before the spectrogram — you can't
realistically "add a dishwasher" to a log-mel image (log of a sum ≠ sum of
logs). It applies **only to training data**: augmenting val/test would
change the question we're measuring.

The "silence" class is *generated* the same way: noise crops at random gain
(including near-zero = true silence).

## 9. Loss and optimization (the one-paragraph version)

The model outputs 12 logits → softmax → probabilities. We minimize
**cross-entropy** (the negative log-probability of the correct class) with
**label smoothing 0.1**: the target is 0.9-ish for the true class instead of
1.0, which stops the model from pushing logits to extremes and improves
calibration — useful since the demo shows probabilities. Optimizer is
**AdamW** (per-parameter adaptive step sizes + decoupled weight decay) with
a **cosine learning-rate schedule** from 3e-3 down to ~0: large steps early
to explore, tiny steps late to settle. Batch size 128.

## 10. Metrics: accuracy vs macro-F1, confusion matrix

- **Accuracy** = fraction correct. Fine headline number, but it weights
  classes by frequency.
- **Per-class F1** = harmonic mean of precision (when we say "stop", was it
  "stop"?) and recall (when it *was* "stop", did we say so?).
- **Macro-F1** = unweighted mean of per-class F1 — every class counts
  equally regardless of size, so a model that quietly fails on "silence"
  can't hide behind the keyword classes. Reporting both is the honest move.
- The **confusion matrix** (true class × predicted class) shows *which*
  mistakes happen. **What our final model actually does** (official test
  set, 164 total errors out of 4,890): the single biggest confusion is
  off→up (15), then on→off (8), go→no (7), up→on (6), down→no (6) — all
  acoustically sensible (shared vowels/short plosive onsets). The hardest
  class is "unknown" (F1 0.921): 39% of all errors involve it, because by
  construction it overlaps every keyword (it's sampled from 25 other words,
  some near-homophones — "tree"/"three", "forward"/"four"). "silence" is
  near-perfect (F1 0.998). This is the honest story to tell in an interview:
  the errors are where phonetics predicts they'd be, not random.

## 11. The edge/TinyML framing

"Small" is a feature, not a limitation, and it's measured:
- **119,372 parameters / 0.48 MB fp32** — fits in the SRAM of a Cortex-M7
  class microcontroller; quantized to int8 it would be ~0.12 MB.
- **CPU latency, batch=1, single thread: ~1.9 ms** (measured on Apple M2;
  see assets/metrics.json) — the realistic deployment setting: a stream of
  single clips on a small CPU, no batching, no GPU. That's ~500 inferences
  per second on one core; a wake-word model only needs ~10/s.
- For context, MobileNet-style factorization is exactly how production
  wake-word models are built; our DS-CNN is the canonical small-footprint
  KWS architecture (arXiv:1711.07128).

## 12. The live demo: where theory meets the browser

Pipeline in `app/app.py`, and why each step exists:
1. Browser records **48 kHz, possibly stereo** → average to mono, resample
   to 16 kHz (Nyquist + anti-aliasing, §2). Skip this and the demo fails
   *silently* — the #1 bug in deployed audio demos.
2. Recordings aren't exactly 1 s → pick the **highest-energy 1-second
   window** (slide a window, sum of squares, take the max) so we crop to
   the word, not to leading silence.
3. Same LogMel transform + same normalization stats as training (loaded
   from the checkpoint — single source of truth).
4. Softmax → top-3 probabilities + the spectrogram itself, so you can *see*
   what the model sees.

## 13. Likely interview questions (answer sketches)

- *Why not raw waveforms?* Could work (e.g. wav2vec-style), but needs more
  capacity/data; log-mels bake in 60 years of speech DSP for free —
  perfect for a 0.5 MB model.
- *Why not an RNN/transformer?* At this budget DS-CNNs are the
  accuracy-per-parameter sweet spot (Hello Edge compared exactly this);
  attention models win at larger budgets (KWT, 98.6%).
- *What would you do with more time?* Streaming detection (sliding window +
  posterior smoothing instead of isolated 1 s clips), int8 quantization
  (PTQ → ~4× smaller, faster), SpecAugment, mixup, knowledge distillation
  from a larger teacher.
- *Where does your model fail?* Read it off the confusion matrix — and say
  so concretely (e.g. which pair confuses most, with counts).
- *Gap between val and test accuracy?* Val is "spent" on model selection
  (§7); also val/test populations differ slightly (val unknowns are a
  sample; test is the official curated set).

*(Numbers cited above — param count, sizes — are computed by the code;
final accuracy/F1/latency live in assets/metrics.json, produced by
`evaluate.py` on the official test set.)*
