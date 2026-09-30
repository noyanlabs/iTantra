# Future improvements (theory, not implemented)

None of this is built yet. These are directions, with notes on how they fit the current code. Claims about specific model behavior are untested and need verifying on your model.

---

## 1. Streaming instead of waiting for the full utterance

**Today:** record the whole utterance (push-to-talk release, or a 2 s pause in Live mode), then run STT once (`SttService.transcribe`), then send the text. Latency = speech length + silence wait + STT time.

**Idea:** decode while the person is still talking, and send partial text as it appears.

- **Needs a streaming STT model.** The current IndicConformer CTC model is *offline*. It needs the whole clip, and it can't be made to stream just by feeding it chunks. sherpa-onnx has an *online* recognizer (`OnlineRecognizer`, streaming transducer or streaming CTC models). You'd need a streaming Hindi model, which has to be found and checked. The model swap steps are in `SWAPPING_MODELS.md`.
- **Flutter side:** `Recorder` already receives audio in small chunks. Instead of buffering everything, each chunk goes into an online stream (`acceptWaveform`), then `decode` runs while `isReady`, and `getResult` gives the text so far.
- **Endpointing replaces the 2 s energy timer.** The online recognizer has built-in endpoint rules (trailing silence), which are better than a fixed threshold. This would also shorten Live mode's pause.
- **Sending:**
  - Send partial text as a new message type (for example `{"t":"partial","id":N,"d":"..."}`), then a final one with the same `id` that replaces it. The receiver updates one chat bubble.
  - Speak on the receiver only at sentence or phrase boundaries, or TTS will stutter. TTS (Piper) is not streaming, so chunk the text per phrase and synthesize each chunk as it arrives.
- **Cost:** streaming models are usually a bit less accurate than offline ones, and partial text can change. Since accuracy is 40% of the score, measure WER before switching.
- **CPU work in Dart:** decoding on the UI isolate would make the screen stutter. Move STT into a background isolate before doing this.

---

## 2. Showing bandwidth

What to show: bytes per second sent and received, total bytes, and the compression ratio.

- **Where to count:** `Transport.send` knows each outgoing line, and `incoming` sees each incoming one. Count the UTF-8 byte length of the JSON line (plus 1 for the newline) in both places, and keep running totals in `Transport`.
- **Rate:** sample the totals once a second, like `Metrics` does for CPU, and show the difference as B/s.
- **Compression ratio:** compare raw text bytes with the gzip+base64 size. For a short sentence, gzip plus base64 can be *larger* than the raw text. Show the real numbers, and consider only compressing when it actually saves bytes.
- **Comparison for the demo:** show "raw audio equivalent" next to the text bytes. PCM at 16 kHz, 16-bit, mono is 32,000 bytes per second of speech. That is the whole argument for sending text.
- **UI:** add a line to `_metricsBar()` in `main.dart`.
- **Real link speed:** that's a different number and harder to get. The TCP socket gives you bytes sent, not the radio rate.

---

## 3. "What if STT mis-hears a word in a life-critical alert?"

The app doesn't handle this today. It sends whatever text the model returns, with no confidence information.

**Ideas, cheapest first:**

1. **Surface confidence.** sherpa-onnx's offline result may expose per-token log-probabilities (`tokens`, and in some versions `ysScores` or similar, which needs checking in the Dart result class for this model). If available, compute an utterance score (mean or minimum token probability).
2. **Low-confidence fallback:**
   - Mark low-confidence words in the sender's bubble and on the receiver, for example with a "?" or a different color, so a human knows to double check.
   - Ask the sender to confirm or repeat ("Did you say ...?") before sending when the score is below a threshold.
   - For alerts, send a short flag with the text (`"low_conf": true`) and have the receiver show it.
3. **Never drop the audio evidence.** Optionally keep the raw recording of the utterance and let the receiver request it over the link when the text looks doubtful. This costs bandwidth, so make it an option, not the default.
4. **Critical-word checks:** run a small keyword list (SOS words, numbers, medical terms, see section 4) over the output and require higher confidence for those words, because a wrong digit in a location or a wrong drug name does the most harm.
5. **N-best:** beam search can return alternative hypotheses. If the top two disagree on a critical word, flag it.
6. **Emergency button is the safe path.** The SOS button doesn't go through STT at all, which is the reason to keep it separate.

Be honest in a demo: there is no guarantee here. Confidence scores from CTC models are often over-confident, so calibrate the threshold on real recordings before trusting it.

---

## 4. Adapting STT to the distress vocabulary

**Why:** disaster speech uses a narrow vocabulary: SOS phrases, place and location words, medical terms, numbers. A general model has no reason to favor those words, so biasing it toward them can lower errors on the real use case.

**Options, lightest first. All need testing on your model.**

1. **Hotword / contextual biasing.** sherpa-onnx supports hotwords for some *transducer* models (`hotwordsFile`, `hotwordsScore` in the recognizer config). It does **not** apply to the current CTC model, so this route depends on switching model family.
2. **Post-correction lexicon (works with any model).** After STT, match the output against a list of critical phrases using fuzzy matching (edit distance, ideally on phonetic or character level for Hindi) and snap close matches to the known word. Risk: it can wrongly "correct" a real different word into an alert word. Keep the list small and set a strict threshold, and mark corrected words in the UI.
3. **Number handling.** Normalize spoken numbers ("पाँच", "five") to a consistent form with rules, and read digits back through TTS, since numbers carry location and quantity.
4. **Language model rescoring.** A small n-gram LM trained on disaster text, used to rescore beam hypotheses. sherpa-onnx supports external LMs for some decoders, but check support for your exact model, and getting Hindi disaster text for training is the main work.
5. **Fine-tuning the acoustic model.** Best results, but needs recorded Hindi speech of those phrases, GPU training, and re-exporting to ONNX. The narrative already records that NeMo export was a multi-day dead end on your setup, so treat this as a later project.

**Measure it.** Record a test set of real distress sentences, compute WER before and after each change, and only keep changes that help. Without that test set, you can't claim an improvement.

---

## Suggested order if you only have a little time

1. Bandwidth display (small, visible in a demo).
2. Critical-word lexicon correction plus marking corrected words (option 2 of section 4).
3. Confidence display, if the result exposes scores (section 3).
4. Streaming (section 1), last, because it needs a new model and a background isolate.
