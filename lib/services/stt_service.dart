import 'dart:typed_data';

import 'package:sherpa_onnx/sherpa_onnx.dart' as sherpa;

import 'asset_copy.dart';

/// Offline Hindi STT (IndicConformer CTC int8 via sherpa-onnx NeMo-CTC). Input: 16 kHz mono.
class SttService {
  sherpa.OfflineRecognizer? _rec;

  Future<void> init() async {
    final root = await ensureAssetsOnDisk('stt');
    sherpa.initBindings();
    _rec = sherpa.OfflineRecognizer(sherpa.OfflineRecognizerConfig(
      model: sherpa.OfflineModelConfig(
        nemoCtc: sherpa.OfflineNemoEncDecCtcModelConfig(model: '$root/model.int8.onnx'),
        tokens: '$root/tokens.txt',
        numThreads: 2,
        debug: false,
      ),
    ));
  }

  /// [samples] must be float32 in [-1, 1], 16 kHz mono. Whole utterance in one call.
  String transcribe(Float32List samples) {
    final r = _rec!;
    final s = r.createStream();
    s.acceptWaveform(samples: samples, sampleRate: 16000);
    r.decode(s);
    final text = r.getResult(s).text.trim();
    s.free();
    return text;
  }

  void dispose() => _rec?.free();
}
