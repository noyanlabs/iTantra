# Technical Reference — STT/TTS Models for the App

## STT — AI4Bharat IndicConformer (Hindi), via OpenVoiceOS ONNX conversion

**Source (do NOT use raw AI4Bharat `.nemo`, too heavy — use this instead):**
`huggingface.co/OpenVoiceOS/ai4bharat-indicconformer-hi-onnx`
(Marathi: same pattern, `ai4bharat-indicconformer-mr-onnx`)

**Files needed** (all three, in the same folder):
- `model.int8.onnx` — 138 MB, INT8-quantized weights
- `vocab.txt` — token vocabulary for decoding
- `config.json` — **required**, we initially missed this and it caused a load failure

**Architecture**: Hybrid CTC-RNNT Conformer-Large, 120M params (encoder), 17 conformer blocks, 512-dim. We are using the **CTC decoding head** (`nemo-conformer-ctc` model type), not RNNT — CTC exports more cleanly to ONNX and is lighter for mobile. **Unconfirmed**: whether OpenVoiceOS's conversion is definitely CTC-only — a community member asked this exact question on a sibling language page and it wasn't resolved in what we found. Verify by checking the actual output/behavior, which we've now done empirically (it works as CTC via `onnx_asr`).

**Input**: 16kHz, mono-channel WAV. **This is non-negotiable** — mismatched sample rate/channels will silently degrade or break accuracy. Always resample with `ffmpeg -ar 16000 -ac 1` before feeding audio in.

**Loading library used for verification**: `onnx_asr` Python package.
```python
import onnx_asr
model = onnx_asr.load_model("nemo-conformer-ctc", "/path/to/stt/", quantization="int8")
# or rename model.int8.onnx -> model.onnx and drop the quantization arg
print(model.recognize("audio.wav"))
```
Key gotcha: `load_model()`'s first argument is an **architecture-type string** from a fixed list (e.g. `"nemo-conformer-ctc"`), **not** a Hugging Face repo ID — the second positional argument is the local path. Passing a local path as the first argument fails with a confusing `HFValidationError`.

**Known issue — truncation on longer audio**: transcriptions were getting cut short in testing. Root cause **not fully isolated yet** — leading hypothesis is the test recordings themselves were cut by a fixed-duration `ffmpeg -t 5` capture before the speaker finished talking, not a model defect. Other possible contributors flagged but unconfirmed: CTC decoders can have edge artifacts at utterance end (architecture-inherent quirk, distinct from RNNT); possible max-input-length truncation in preprocessing. **Action item**: test with untruncated recordings (`ffmpeg` without `-t`, manual stop) to confirm this is a recording-length artifact, not a model bug, before building pause-detection logic in the app.

**What still needs solving for the app** (not yet done, don't assume `onnx_asr`'s Python-side logic ports for free):
- Audio preprocessing (PCM → mel-spectrogram) — the exact math `onnx_asr` does internally on the PC needs to be replicated in Dart/Kotlin for Flutter; not automatically portable
- CTC decoding (logits → text) — same issue, this logic lives in `onnx_asr` on Python, must be reimplemented for Flutter
- No true streaming/incremental transcription — this model does full-utterance transcription only, which matches the PS's "pause-triggered, transcribe full sentence" design, so this is fine, not a gap

---

## TTS — Piper (Hindi voice: "rohan")

**Source**: `huggingface.co/rhasspy/piper-voices/tree/main/hi/hi_IN/rohan/medium` (official Piper voices repo, not a reposted mirror)

**Files needed** (both, same folder, matching filename stem):
- `hi_IN-rohan-medium.onnx`
- `hi_IN-rohan-medium.onnx.json` — config (phoneme/espeak mapping, sample rate). **Both files required together** — the `.onnx` alone is not usable.

**Architecture**: VITS-based, single-stage (unlike AI4Bharat's two-stage FastPitch+HiFiGAN) — text goes in, waveform comes out directly. Medium quality tier ≈ 60MB, ~15M params, 22050 Hz output.

**Verified working** via:
```bash
pip install piper-tts
echo "<hindi text>" | piper --model hi_IN-rohan-medium.onnx --output_file out.wav
```
Confirmed producing intelligible Hindi speech in testing.

**License caveat, unresolved**: `piper-tts` (the Python library) is GPL; individual voice model files may carry separate, more permissive terms, but this was **never explicitly confirmed per-voice**. Given the PS's open-source-only requirement, check this before final submission — don't assume it's clear.

**Note on why we're not using AI4Bharat's own TTS**: the official `AI4Bharat/Indic-TTS` GitHub release zips are ~1.4GB **per language** (research-release artifact, likely bundling training checkpoints/optimizer states, not deployment-trimmed) — infeasible for the Efficiency metric and impractical to download reliably. Piper was chosen as the practical substitute. Trade-off to keep in mind: Piper is lighter/faster but likely less natural/expressive than AI4Bharat's TTS would have been — relevant to the "human legibility and flow" accuracy sub-metric.

---

## Environment/tooling notes worth remembering

- **Python 3.14 breaks NeMo entirely** — `triton` has no macOS build at any Python version, and parts of the toolchain assume Linux/CUDA. If any further NeMo work is needed, use a `conda` env pinned to Python 3.10, and install via `pip install "nemo_toolkit[asr]"` directly rather than `bash reinstall.sh` (which pulls in the full, heavier dependency tree including triton).
- **We ended up NOT needing NeMo at all** for deployment — OpenVoiceOS's pre-converted ONNX saved that entire step. Worth remembering if teammates start the NeMo path independently without knowing this shortcut exists.
- **Verify models on PC (Python) before touching Flutter.** Both STT and TTS were sanity-checked this way — this is what caught the missing `config.json` and the truncation issue early, before they'd have been much harder to debug inside a Flutter/Android build.

---

## Immediate next steps, in order

1. Resolve the STT truncation question definitively (untruncated recording test)
2. Repeat the full PC-verification process for **Marathi** (STT + TTS) — not yet done, only Hindi confirmed
3. Begin Flutter integration via `flutter_onnxruntime`, TTS first (simpler pipeline, no custom preprocessing/decoding needed on-device beyond what Piper's format requires), then STT
4. Write the audio preprocessing (mel-spectrogram) and CTC decoding logic natively for Flutter — this is unverified/unbuilt and is likely the single biggest remaining technical risk, since it's the one piece with no ready-made library doing it for you in Dart