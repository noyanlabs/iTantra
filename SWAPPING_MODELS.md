# Swapping the STT / TTS model

Yes, you can use a different ONNX model. The catch is that the app runs models through **sherpa-onnx**, not raw ONNX Runtime. A model works only if sherpa-onnx supports its architecture. Checking that comes first.

- STT code: `lib/services/stt_service.dart`
- TTS code: `lib/services/tts_service.dart`
- Assets are copied to app storage once, by `lib/services/asset_copy.dart`.

---

## Step 0: Check the model is supported

Open the sherpa-onnx pretrained model lists:
- TTS: https://k2-fsa.github.io/sherpa/onnx/tts/pretrained_models/index.html
- STT: https://k2-fsa.github.io/sherpa/onnx/pretrained_models/index.html

Supported families include:
- **TTS:** Piper/VITS, Matcha, Kokoro, Kitten and others.
- **STT:** NeMo CTC, transducer, Whisper, Paraformer, SenseVoice and others.

If the model's family is not there, sherpa-onnx can't run it. You would need a different runtime, which means bigger changes. Note that `flutter_onnxruntime` can't be added back, because it clashes with sherpa-onnx's own `libonnxruntime.so` and breaks the build.

**Prefer a model already packaged for sherpa-onnx.** Those tarballs include the files and metadata sherpa needs.

---

## Swapping the TTS model

1. Download and extract the sherpa-onnx package for the voice. It usually has: the `.onnx`, `tokens.txt`, and `espeak-ng-data/` (Piper) or other extra files.
2. Delete everything in `assets/tts/`, then copy the new files in.
3. Regenerate the asset list in `pubspec.yaml`. Flutter needs every subfolder listed, because it doesn't recurse. From the project folder:
   ```
   python3 - <<'E'
   import os,re
   dirs=sorted({r for r,_,f in os.walk('assets/tts') if f})
   s=open('pubspec.yaml').read()
   s=re.sub(r"(    - assets/tts/.*\n)+","".join(f"    - {d}/\n" for d in dirs),s)
   open('pubspec.yaml','w').write(s)
   E
   ```
4. In `lib/services/tts_service.dart`, change the model filename (`hi_IN-rohan-medium.onnx`) in `OfflineTtsVitsModelConfig`.
5. **Different architecture** (Matcha, Kokoro, Kitten): use the matching config class (`OfflineTtsMatchaModelConfig` and so on) instead of `OfflineTtsVitsModelConfig`. Their fields are in `sherpa_onnx-<version>/lib/src/tts_config.dart` in your pub cache.
6. **Sample rate:** the WAV writer takes the rate from the model output, so nothing to change.
7. **Reset the copied files.** The app copies assets only once. Uninstall the app, or clear its storage, or the old model is still used.

---

## Swapping the STT model

1. Get the model files: the `.onnx` (prefer an int8 one for size) plus `tokens.txt`.
2. **Metadata.** The model must carry the metadata sherpa expects.
   - A sherpa-onnx packaged model already has it, so skip this.
   - A raw NeMo CTC model needs it added. Edit `tools/add_stt_metadata.py`:
     - `vocab_size` = number of tokens **including** the blank (the blank is the last id).
     - `subsampling_factor`: 8 for Conformer-Large, 4 for some smaller models, and check the model's config.
     - `normalize_type`: `per_feature` for most NeMo models.
   - Then run `python3 tools/add_stt_metadata.py <path-to-model.onnx>`. It writes `assets/stt/model.int8.onnx`.
3. Put `tokens.txt` in `assets/stt/`. Each line is `token id`. The blank token must be the last entry.
4. In `lib/services/stt_service.dart`:
   - **Same family (NeMo CTC):** only change the filename if it differs.
   - **Other family** (transducer, Whisper, Paraformer): replace `nemoCtc:` with the matching config in `OfflineModelConfig`. A transducer needs three files (encoder, decoder, joiner) instead of one.
5. **Input audio stays 16 kHz mono.** This is handled in `lib/services/recorder.dart`. Check the new model's expected sample rate. If it isn't 16 kHz, change the recorder and the `sampleRate` in `transcribe()`.
6. Reset the copied files: uninstall the app or clear its storage.

---

## Checks after any swap

- Test a debug build, then a **release** build: `flutter run --release`. The proguard rule is in `android/app/proguard-rules.pro`.
- If the output is empty or garbage, check the metadata values first, then `tokens.txt`.
- If the app crashes at launch, check the filenames in the service files against the files in `assets/`.
- Different languages: each language needs its own STT and TTS model, and the UI dropdown entries for other languages are still placeholders.
- **Check the model's license** before using it. The current Hindi voice's license is not clearly open-source. See the note in `assets/tts/MODEL_CARD`.
- Size: keep an eye on the APK size, since Efficiency is scored.
