import 'dart:math';
import 'dart:typed_data';

import 'package:sherpa_onnx/sherpa_onnx.dart' as sherpa;

import 'asset_copy.dart';

/// Offline Hindi STT (IndicConformer CTC int8 via sherpa-onnx NeMo-CTC). Input: 16 kHz mono.
class SttService {
  sherpa.OfflineRecognizer? _rec;

  static const _sampleRate = 16000;
  static const _tailPadSeconds = 4;
  static const _padAmplitude = 0.0005; // near-silence; set 0 for pure zeros

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

  /// Appends [_tailPadSeconds] of near-silence so the CTC head can flush the final tokens.
  Float32List _padTail(Float32List s) {
    final pad = _sampleRate * _tailPadSeconds;
    final out = Float32List(s.length + pad);
    out.setRange(0, s.length, s);
    final rnd = Random(1);
    for (var i = s.length; i < out.length; i++) {
      out[i] = (rnd.nextDouble() * 2 - 1) * _padAmplitude;
    }
    return out;
  }

  /// [samples] must be float32 in [-1, 1], 16 kHz mono. Whole utterance in one call.
  String transcribe(Float32List samples) {
    if (samples.isEmpty) return '';
    final r = _rec!;
    final s = r.createStream();
    s.acceptWaveform(samples: _padTail(samples), sampleRate: _sampleRate);
    r.decode(s);
    final text = r.getResult(s).text.trim();
    s.free();
    return text;
  }

  void dispose() => _rec?.free();
}