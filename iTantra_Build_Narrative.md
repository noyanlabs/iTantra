# iTantra — Build Narrative for Implementing Agent

This document is the complete context an agent needs to build this app. Read it fully before writing code. It covers what already exists, what is verified working, what is unverified/risky, and the exact architecture to build. Do not re-derive decisions already made here — follow them, and flag explicitly if something here turns out to be wrong once you're in the code.

---

## 0. What this app is (context, not to be re-explained to the user)

This is a Smart India Hackathon (SIH) submission. Problem statement: build an Android app for alert/distress scenarios that converts speech to text locally, transmits the text over WiFi/Bluetooth (not raw audio, because audio is too data-heavy for low-bandwidth/disaster-zone links), and converts the received text back to speech on the receiving phone. Two phones running the same app should work like a walkie-talkie. Everything must run fully offline, on-device, on low/mid-range Android phones, using only open-source models — no cloud APIs.

Evaluation weighting the team is optimizing for: Efficiency 20% (model size, RAM/flash, idle CPU), Accuracy 40% (STT WER, TTS naturalness/"legibility and flow"), Latency 20% (STT completion time, TTS synthesis time, RTF, and critically the full loop time — sentence spoken on phone A to sentence starting to play on phone B).

Every architectural choice below has been made with this weighting in mind. Do not add features that trade against these metrics without flagging it back to the user — e.g., do not bundle all 10 languages' models into the APK (kills Efficiency), do not add cross-language translation (out of scope, adds latency), do not add speech emotion detection (not in requirements, unverified for Hindi, adds risk with no scoring benefit).

---

## 1. What already exists — DO NOT re-download, re-research, or re-derive these

The user has already done significant work to get here. Two full days were spent solving environment issues, format mismatches, and finding the right model sources. Do not send the user back through this. Assume the following is done and verified:

### STT model — Hindi, verified working on the user's Mac
- Source: `huggingface.co/OpenVoiceOS/ai4bharat-indicconformer-hi-onnx` — a **verified, credible conversion** (OpenVoiceOS is an established open-source project, not a random reupload) of AI4Bharat's IndicConformer to ONNX, done for their own `onnx-asr` library and STT plugin.
- Local files the user has, exactly as downloaded, sitting in a local folder (e.g. `~/Downloads/stt/`):
  - `model.int8.onnx` (138 MB) — **this is the correct, quantized file to use**. Do NOT use `model.onnx` + `model.onnx_data` (that pairing is the unquantized FP32 version, ~481MB+, and was explicitly rejected during model selection for being too heavy).
  - `vocab.txt` — token vocabulary, required for decoding.
  - `config.json` — required; loading failed without it initially, must be present alongside the other two files.
- **Underlying architecture**: Hybrid CTC-RNNT Conformer-Large (120M param encoder, 17 conformer blocks, 512-dim), converted using the **CTC decoding head** specifically (not RNNT — CTC is lighter and exports more cleanly for mobile; this was a deliberate choice made earlier in the process).
- **Verified working** on the user's machine via Python (`onnx_asr` library, NOT relevant to Flutter build but proves the model itself is sound): fed a 16kHz mono WAV recording, produced correct Hindi text output (`मैं ये एक टेस्ट`). This confirms the ONNX file, vocab, and config are all valid and functional — any future STT problems in the Flutter build are integration bugs, not bad model files.
- **Input requirement, confirmed and non-negotiable**: 16kHz, mono-channel PCM audio. Any mismatch here (wrong sample rate, stereo instead of mono) will silently degrade or break accuracy without throwing an error. All audio capture code MUST resample/force to 16kHz mono before inference.
- **Known open issue, unresolved**: transcription was observed to truncate on longer test recordings. Leading hypothesis is that this was an artifact of a fixed-duration test recording (`ffmpeg -t 5` cutting off mid-sentence) rather than a genuine model defect — this was NOT fully isolated before the user moved to app-building. **The implementing agent should treat this as an open risk**: if transcriptions come out truncated in the app once pause-detection is implemented properly (see Section 4), this needs dedicated debugging — check (a) whether the full utterance audio is actually being passed to the model, (b) whether there's an unintended max-length truncation somewhere in preprocessing, (c) whether it's a CTC-decoding edge artifact at utterance boundaries (a known characteristic of CTC-style decoders in general, separate from RNNT). Do not assume this is fixed just because it wasn't blocking.
- **Marathi and all other languages besides Hindi are OUT OF SCOPE for this build** — see Section 3 for the revised frontend treatment (placeholders only, no functional pipeline, no visible distinction from Hindi in the UI). Do not spend implementation time downloading, verifying, or integrating Marathi or any other language's models unless explicitly asked. The `OpenVoiceOS/ai4bharat-indicconformer-mr-onnx` source exists and follows the same pattern as Hindi if this changes later, but it is not part of the current build.

### TTS model — Hindi, verified working on the user's Mac
- Source: `huggingface.co/rhasspy/piper-voices/tree/main/hi/hi_IN/rohan/medium` — this is Piper's **official** voices repository (not a reposted mirror), confirmed via an independent citation.
- Local files: `hi_IN-rohan-medium.onnx` + `hi_IN-rohan-medium.onnx.json` (config — phoneme/espeak mapping, sample rate). **Both files are required together**, matched by filename stem; the `.onnx` alone will not work.
- **Architecture**: Piper uses VITS (single end-to-end model — text in, waveform out directly). This is architecturally different and simpler than AI4Bharat's own TTS (which is a two-stage FastPitch acoustic model + HiFi-GAN vocoder pipeline) — Piper was deliberately chosen over AI4Bharat's own TTS.
- **Why Piper instead of AI4Bharat's own Indic-TTS**: AI4Bharat's official TTS release (`github.com/AI4Bharat/Indic-TTS`, checkpoints via GitHub Releases) is ~1.4GB compressed **per language** — this is a research-release artifact (likely bundling training checkpoints/optimizer states, not a deployment-trimmed export) and was explicitly rejected as infeasible for the Efficiency metric and impractical to reliably download. Piper's medium-quality voice is ~60MB and is built for edge deployment from the start.
- **Trade-off the user should be aware of, and the agent should not "fix" unprompted**: Piper is lighter and faster but likely less natural/expressive than AI4Bharat's TTS would have been. This affects the "human legibility and flow" component of the Accuracy metric (40% weighted). This was a deliberate, informed trade-off favoring Efficiency/Latency over maximum naturalness — do not silently swap in a heavier TTS model to "improve quality" without flagging the size/latency cost back to the user first.
- **Verified working**: `piper --model hi_IN-rohan-medium.onnx --output_file out.wav` on Hindi text produced intelligible spoken Hindi output on the user's Mac.
- **License note, UNRESOLVED — flag before final submission**: `piper-tts` (the Python CLI/library) is GPL-licensed. Whether the voice model files themselves carry separate, more permissive terms was never confirmed. The competition requires open-source only. This must be checked (read the actual LICENSE file / MODEL_CARD in the `rhasspy/piper-voices` repo for the Hindi voice specifically) before the app is considered submission-ready. Do not assume this is fine.
- **Marathi and all other languages besides Hindi are OUT OF SCOPE for this build** — same as the STT note above. No TTS work needed for any language besides Hindi.

### What this means for the agent building the Flutter app
- **Do not re-search for or suggest alternative STT/TTS models.** The selection process (AI4Bharat IndicConformer for STT via OpenVoiceOS's ONNX conversion, Piper for TTS) is final for the hackathon timeline, made after ruling out several alternatives (raw AI4Bharat NeMo checkpoints — too heavy and required a broken macOS toolchain to export; AI4Bharat's own TTS — 1.4GB per language, infeasible).
- **Do not attempt to convert or quantize models yourself as a "first step."** That work is done for Hindi, which is the only language in scope. Don't default to "let's export from NeMo" for anything — that was a multi-day dead-end for this team already (Python 3.14 / triton / macOS incompatibility, resolved only after switching to conda + Python 3.10, and even then the export was never completed — the OpenVoiceOS pre-converted files were used instead).

---

## 2. Runtime stack for Flutter

- **ONNX Runtime plugin**: `flutter_onnxruntime` (pub.dev). This is the confirmed, actively maintained package for running both the STT and TTS ONNX models on-device in Flutter/Android.
  - Android requires a proguard rule: create/edit `android/app/proguard-rules.pro` with:
    ```
    -keep class ai.onnxruntime.** { *; }
    ```
    Without this, release builds will strip needed classes and crash at runtime — debug builds may work fine and mask this until a release build is tested, so test a release build explicitly, don't only test debug.
  - For large model files (138MB STT, ~60MB TTS), do NOT bundle as Flutter assets in the APK for all 10 languages — this directly damages the Efficiency (app size) metric. Load the initially-selected language's model files from app-local storage (downloaded at runtime or bundled only for the single default/available language), using `createSessionFromFile`, not `createSessionFromAsset`, once the app scales beyond one language.
  - Given only Hindi is actually functional (see Section 3, point 1 — all other languages are frontend-only placeholders with no backing pipeline), it's reasonable to bundle just the Hindi model files as Flutter assets for this build. No need to architect for runtime per-language download or multi-model switching logic — that complexity isn't needed until/unless more languages are actually added, which is not part of the current scope.

- **What ONNX Runtime does NOT give you for free** — the agent must build these, there is no shortcut library for either:
  1. **Audio preprocessing for STT**: raw mic PCM → the exact mel-spectrogram/feature representation IndicConformer expects. The Python `onnx_asr` library did this automatically during PC verification, but that logic does not port to Flutter — it must be reimplemented in Dart or (better, for performance) in native Kotlin exposed via a platform channel. Get the exact preprocessing parameters (sample rate — confirmed 16kHz — window size, hop length, number of mel bins, normalization) from the model's `config.json` or from NeMo's own model config if `config.json` doesn't fully specify it. Mismatches here degrade accuracy silently.
  2. **CTC decoding for STT**: the raw model output is per-timestep logits, not text. A greedy (or beam) CTC decode routine must be written, using `vocab.txt` to map token IDs to characters. This is standard CTC decoding (collapse repeated tokens, remove blank tokens) but is not automatic — it must be implemented.
  3. **Text tokenization + phonemization for TTS**: Piper expects input processed through `espeak-ng` phonemization before being fed to the model (confirmed from Piper's own standalone-ONNX usage pattern). This is a real dependency, not optional — check whether `espeak-ng` can be bundled/run on Android (it's a native C library; there may be an existing Android port or Flutter plugin — search for this specifically before assuming it needs to be built from scratch) or whether Piper's `.onnx.json` config exposes a simpler phoneme mapping that avoids needing full espeak-ng on-device.
  4. **Waveform → playable audio**: after HiFi-GAN-equivalent (or in Piper's case, direct VITS decoder) output, you have a raw float32 waveform array. This needs to be encoded into a playable format (WAV header + PCM, or fed directly to a raw-PCM-capable audio player) and played via `just_audio`, `audioplayers`, or similar.

---

## 3. UI Requirements (from user's spec, verbatim intent preserved)

1. **Main screen**: large mic button, bottom center. Language dropdown, center, all 10 languages listed (Hindi, Gujarati, Marathi, Kannada, Malayalam, Tamil, Telugu, Odia, Bengali, English) — Hindi selected by default.

   **Revised scope, supersedes any earlier "unavailable" popup requirement**: only Hindi is actually functional. The other 9 languages are **visual placeholders only** — they exist in the dropdown so the UI looks and feels like the full 10-language product, but selecting one should NOT surface any "model unavailable" popup, error dialog, or other explicit signal that they're non-functional. Do not draw attention to the gap. Pick one of these two behaviors (agent should choose based on what's simplest to implement correctly, and confirm with user if genuinely unsure):
   - Selecting a non-Hindi language silently keeps the app on the Hindi pipeline underneath (i.e., the dropdown visually updates to show the selected language, but STT/TTS continues running against the Hindi models regardless of what's shown) — riskiest if a judge tests in a language that clearly isn't Hindi and notices the mismatch, so only sensible if there's no live multi-language demo planned.
   - Selecting a non-Hindi language keeps the UI otherwise fully normal (mic button enabled, no error, no popup) but that language's talk flow is simply not exercised in the live demo — the dropdown's job is to show breadth of intended scope, not to be interacted with beyond Hindi during any actual demonstration.

   In either case: **no error states, no "coming soon" badges, no popups, no visibly greyed-out/disabled entries** for the other 9 languages. They should read as complete and present in the dropdown, same as Hindi, with no frontend signal distinguishing them.
2. **Chat-style transcript view**: STT output text and TTS-received text displayed in a chat-bubble-style scrolling view (like a messaging app), not just a single line that gets overwritten. This is both a UX nicety and useful for demoing to judges — they should be able to see the conversation history, not just hear it.
3. **Live metrics display**: Latency (of the last STT/TTS/round-trip cycle), RAM usage, CPU usage — visible on screen, updating live or per-message. This directly demonstrates the Efficiency and Latency metrics to judges during a live demo, which is valuable for scoring, not just a debug feature — treat this as a first-class UI element, not an afterthought debug overlay.
4. **Two talk modes, user-switchable**: "Live mode" and "Walkie-Talkie mode" (see Section 4 for behavior).
5. **Device connection UI**: a way to discover and connect to the other phone over WiFi (WiFi Direct) — separate from the mic/talk UI, but accessible from the main screen (e.g., a connection status indicator + a "connect" flow).
6. **Emergency button**: large, visually distinct (should not be confusable with the normal mic button), triggers the emergency flow described in Section 6.

---

## 4. Talk Modes — exact behavior

### Live Mode
- User taps mic once to start. Mic stays open/listening continuously.
- The app must detect **pauses in speech** — specifically, a pause of approximately 2 seconds of no speech after the user has started speaking — and treat that as the end of one "utterance."
- That utterance's audio (from start of speech to the detected pause) is sent to the STT pipeline, the resulting text is sent over the WiFi connection to the other device, and the mic continues listening for the next utterance (does not require the user to tap again) — this is the "live," continuous conversational mode.
- **This requires real voice activity detection (VAD)** — silence/pause detection — not a fixed-duration recording window. This directly relates to the truncation issue flagged in Section 1: earlier testing used a fixed 5-second window, which is NOT what the final app should do. The agent must implement (or find a suitable lightweight VAD approach — even a simple energy-threshold-based silence detector may be sufficient given the constrained scenario, doesn't need to be a full ML-based VAD model) a proper pause-detector before this mode can work correctly. Budget real implementation time for this — it's a core piece of the "detects pauses and stoppages" requirement from the original problem statement, not a minor detail.

### Walkie-Talkie Mode
- Push-to-talk. User holds the mic button, speaks, releases.
- The entire audio from press to release is sent as one block to STT (no internal pause-splitting within a single press) — simpler than Live Mode's continuous pause-detection, since the "utterance boundary" is just the button press/release.
- After STT completes, resulting text goes through the **smart compression/extraction engine** (see Section 5) before transmission, then is decompressed/extracted on the receiving end and played via TTS.
- **Note the asymmetry with Live Mode**: the user's spec only mentions the smart compression engine for Walkie-Talkie mode, not Live mode. Preserve this distinction unless the user says otherwise — don't assume it should apply uniformly to both modes without checking.

### Mode switching
- Should be a simple, clearly visible toggle — the user did not specify exact UI treatment, use judgment (e.g., a segmented control/tab near the mic button) but confirm with the user if it's ambiguous during implementation rather than guessing silently.

---

## 5. "Smart Compression and Extraction Engine" — Walkie-Talkie mode only

The user's spec: "For walkie talkie mode use smart compression and extraction engine made unique for compressing the same message and extracting at listener side in exact same format."

**This requirement is underspecified and the agent should NOT invent a complex custom protocol without clarifying intent with the user first.** Reasonable interpretations, roughly in order of likely intent:

1. **Standard text compression** (e.g., gzip/deflate on the UTF-8 text string before sending over WiFi, decompress on receipt) — text is already tiny compared to audio, so this may be more about minimizing latency/packet size on a potentially poor connection than achieving dramatic compression ratios. This is the simplest, safest interpretation and is likely sufficient — text payloads for a spoken sentence are on the order of tens to a couple hundred bytes; compression gains here are marginal but not harmful, and demonstrates engineering rigor to judges ("we compress every transmission to minimize latency on poor links").
2. **A custom dictionary/shorthand encoding** specific to the disaster/alert domain (e.g., common distress phrases mapped to short codes) — this would give bigger size reduction but requires building and maintaining a phrase dictionary, and risks mistranslation if the phrase-matching isn't robust. Higher risk, higher reward, and NOT explicitly requested detail — only pursue if the user confirms this is what they meant, given hackathon time constraints.
3. Something else the user has in mind that isn't fully captured above.

**Agent action**: implement interpretation #1 (standard compression, e.g. Dart's built-in `gzip` via `dart:io` or `archive` package) as the default/MVP path, since it satisfies the literal requirement ("compressing the message and extracting in exact same format" — compress → transmit → decompress losslessly) with low risk and low implementation cost. Flag to the user that a more elaborate domain-specific compression scheme (interpretation #2) is possible as a stretch goal if time permits, but don't build it speculatively without confirmation.

---

## 6. Emergency Button

- Large, distinct button (separate from normal mic/talk controls).
- On press: immediately sends a special "emergency" tagged message over the WiFi connection (distinct from a normal transcribed-text message — needs its own message type/flag in whatever data format is used for transmission, not just special text content, so the receiving app can reliably distinguish it).
- On the **receiving** device: screen color changes (to something high-visibility, e.g. red/flashing) AND a loud, attention-grabbing beep/alarm sound plays.
- **"Loud sound" implementation note**: standard audio playback respects the user's media volume, which may be low or muted. For a true emergency alert to be effective, this likely needs to force/override system volume and/or request audio focus aggressively (this same requirement — "highest volume non-interruptible" — appears in the original problem statement for TTS alert messages generally, not just this button, so the underlying native-audio-focus mechanism built for this should likely be shared/reused for both). This requires native Android code (via platform channel) using `AudioManager` — not achievable through a pure-Dart audio plugin alone. Budget this as real native development work, not a simple `audioplayers.play()` call.
- Should work in both Live and Walkie-Talkie mode without needing a mode switch — it's an override/interrupt, not a third mode.

---

## 7. Networking — WiFi Direct + connection resilience

- **Mechanism**: Android's native `WifiP2pManager` API, via a Kotlin platform channel (this was established earlier in the project's research — no mature Flutter package exists for this exact phone-to-phone P2P use case, so native implementation is required, not optional). Bluetooth Classic (`BluetoothSocket`) was discussed as an alternative/fallback transport — the user has not finalized which to use as primary; WiFi Direct was the lead candidate. Confirm with user if unclear, or implement WiFi Direct first as primary given it was the more-discussed option.
- **Connection discovery/pairing UI**: needed as part of the main UI (Section 3, point 5) — peer discovery, connection request/accept flow.
- **Graceful retry on connection loss** (explicit requirement #8): the app must detect a dropped connection (WiFi Direct connections can be flaky, particularly cross-OEM — this was flagged as a known real-world risk during earlier research, not a hypothetical) and attempt automatic reconnection without requiring the user to manually redo the full pairing flow, where possible. At minimum: clear UI indication of connection state (connected/disconnected/reconnecting), and an automatic retry loop with reasonable backoff rather than a single silent failure. Messages composed/spoken while disconnected should be queued (not silently dropped) and sent once reconnected, or the user should be clearly informed the message wasn't sent — do not fail silently.

---

## 8. Suggested build order (do not build everything simultaneously)

Given the scope here, sequence matters more than usual. Recommended order, each step should be a working, testable checkpoint before moving to the next:

1. **Flutter project scaffold + `flutter_onnxruntime` wired up**, proguard rule in place. Load the Hindi TTS model, hardcode a Hindi sentence, get it to produce and play audio on a real Android device (not emulator — audio/ONNX behavior can differ). This is the fastest path to confirming the whole toolchain works, since TTS has fewer moving parts than STT (no live mic capture, no custom CTC decode logic needed beyond what Piper's format requires).
2. **STT pipeline**: mic capture (16kHz mono) → preprocessing → ONNX inference → CTC decode → text. Test with a fixed-duration recording first (simplest), get correct Hindi text output on-device, matching what was already verified on the PC.
3. **Basic UI**: mic button, language dropdown (all 10 languages listed, only Hindi functional underneath — see Section 3, point 1 for exact behavior), chat-style transcript display — wire up to the working STT/TTS pipeline from steps 1-2, single device, no networking yet.
4. **WiFi Direct networking**: peer discovery, connect, send/receive plain text between two devices. Prove this works with a trivial "hello" message before wiring it to real STT output.
5. **Wire STT → network → TTS end-to-end** across two devices — this is the core walkie-talkie loop.
6. **Walkie-Talkie mode formalized** (push-to-talk boundary, compression/extraction engine).
7. **Live mode** (continuous listening + real pause/VAD detection) — this is harder than Walkie-Talkie mode and depends on solving proper pause-detection (see Section 4), which the team has not yet validated. Do not attempt this before Walkie-Talkie mode works, since Walkie-Talkie sidesteps the hardest part (VAD) via manual button press/release.
8. **Emergency button** (native audio-focus/volume override, cross-device alert tag + UI color change).
9. **Metrics display** (latency/RAM/CPU) — can be built incrementally alongside earlier steps rather than saved entirely for last, since it's useful for debugging performance throughout, not just a final polish item.
10. **Connection-loss handling and retry logic** — layer onto the networking built in step 4, test by deliberately killing WiFi mid-conversation.
11. ~~Marathi language support~~ — **out of scope for this build.** Marathi and all other non-Hindi languages remain frontend-only placeholders in the dropdown (Section 3, point 1). Do not schedule or attempt this step unless the user explicitly asks to expand language support later.

---

## 9. Things the agent must NOT do

- Do not re-suggest alternative STT/TTS models or re-litigate the model selection — it's final for this build, made after real investigation.
- Do not attempt NeMo model export/conversion as a default troubleshooting step — this was a multi-day dead end already; the ONNX files in hand are the way forward.
- Do not bundle all 10 languages' models in the APK by default — violates the Efficiency metric the team is being scored on.
- Do not add features not requested here (translation, emotion detection, etc.) — these were explicitly considered and deprioritized earlier for being out of scope and/or unreliable.
- Do not invent an elaborate custom compression protocol for Walkie-Talkie mode without confirming interpretation with the user first (Section 5).
- Do not assume the STT truncation issue is resolved — treat it as an open risk to actively verify once real pause-detection (not fixed-duration test recording) is implemented.
- Do not build out Marathi or any other non-Hindi language pipeline, and do not add any UI element (popup, badge, disabled state, tooltip) that reveals to the user/demo audience that only Hindi is functional — the other 9 languages must appear indistinguishable from Hindi in the dropdown, with no functional backing.
