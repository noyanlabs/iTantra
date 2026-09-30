import 'dart:io';
import 'dart:typed_data';

import 'package:path_provider/path_provider.dart';

import 'asset_copy.dart';
import 'package:sherpa_onnx/sherpa_onnx.dart' as sherpa;

/// Offline Hindi TTS (Piper VITS via sherpa-onnx, espeak-ng phonemization).
class TtsService {
  sherpa.OfflineTts? _tts;
  int _n = 0;

  Future<void> init() async {
    final root = await ensureAssetsOnDisk('tts');

    sherpa.initBindings();
    _tts = sherpa.OfflineTts(sherpa.OfflineTtsConfig(
      model: sherpa.OfflineTtsModelConfig(
        vits: sherpa.OfflineTtsVitsModelConfig(
          model: '$root/hi_IN-rohan-medium.onnx',
          tokens: '$root/tokens.txt',
          dataDir: '$root/espeak-ng-data',
        ),
        numThreads: 2,
        debug: false,
      ),
    ));
  }

  /// Returns a playable 16-bit PCM WAV file path for [text].
  Future<String> synthesizeToWav(String text) async {
    final audio = _tts!.generate(text: text);
    final dir = await getTemporaryDirectory();
    final path = '${dir.path}/tts_out_${_n++ % 4}.wav'; // rotate so a file is never rewritten while playing
    await File(path).writeAsBytes(_wav(audio.samples, audio.sampleRate));
    return path;
  }

  void dispose() => _tts?.free();

  Uint8List _wav(Float32List samples, int rate) {
    final n = samples.length;
    final b = ByteData(44 + n * 2);
    void str(int o, String s) {
      for (var i = 0; i < s.length; i++) {
        b.setUint8(o + i, s.codeUnitAt(i));
      }
    }

    str(0, 'RIFF');
    b.setUint32(4, 36 + n * 2, Endian.little);
    str(8, 'WAVEfmt ');
    b.setUint32(16, 16, Endian.little);
    b.setUint16(20, 1, Endian.little);
    b.setUint16(22, 1, Endian.little);
    b.setUint32(24, rate, Endian.little);
    b.setUint32(28, rate * 2, Endian.little);
    b.setUint16(32, 2, Endian.little);
    b.setUint16(34, 16, Endian.little);
    str(36, 'data');
    b.setUint32(40, n * 2, Endian.little);
    for (var i = 0; i < n; i++) {
      b.setInt16(44 + i * 2, (samples[i].clamp(-1.0, 1.0) * 32767).round(),
          Endian.little);
    }
    return b.buffer.asUint8List();
  }
}
