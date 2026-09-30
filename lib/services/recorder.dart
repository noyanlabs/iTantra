import 'dart:async';
import 'dart:math';
import 'dart:typed_data';

import 'package:record/record.dart';

/// Mic capture at 16 kHz mono PCM16, with push-to-talk and energy-based VAD modes.
class Recorder {
  final _rec = AudioRecorder();
  StreamSubscription<Uint8List>? _sub;
  final _buf = <double>[];

  static const _energyThreshold = 0.015; // RMS on [-1,1]
  static const _pauseMs = 2000;
  static const _minSpeechMs = 300;

  Future<bool> _start(void Function(Float32List chunk) onChunk) async {
    if (!await _rec.hasPermission()) return false;
    final stream = await _rec.startStream(const RecordConfig(
        encoder: AudioEncoder.pcm16bits, sampleRate: 16000, numChannels: 1));
    _sub = stream.listen((bytes) {
      final n = bytes.length ~/ 2;
      final bd = ByteData.sublistView(bytes);
      final f = Float32List(n);
      for (var i = 0; i < n; i++) {
        f[i] = bd.getInt16(i * 2, Endian.little) / 32768.0;
      }
      onChunk(f);
    });
    return true;
  }

  Future<void> _stop() async {
    await _sub?.cancel();
    _sub = null;
    await _rec.stop();
  }

  // ---- Walkie-talkie: press → release returns the whole block ----
  Future<bool> startPushToTalk() async {
    _buf.clear();
    return _start(_buf.addAll);
  }

  Future<Float32List> stopPushToTalk() async {
    await _stop();
    final out = Float32List.fromList(_buf);
    _buf.clear();
    return out;
  }

  // ---- Live: continuous, emits an utterance after ~2 s of silence ----
  Future<bool> startLive(void Function(Float32List utterance) onUtterance) {
    _buf.clear();
    var speaking = false;
    var speechMs = 0;
    var silenceMs = 0;
    return _start((chunk) {
      final ms = (chunk.length * 1000 / 16000).round();
      var sum = 0.0;
      for (final v in chunk) {
        sum += v * v;
      }
      final voiced = sqrt(sum / chunk.length) > _energyThreshold;
      if (voiced) {
        speaking = true;
        speechMs += ms;
        silenceMs = 0;
      } else if (speaking) {
        silenceMs += ms;
      }
      if (speaking) _buf.addAll(chunk);
      if (speaking && silenceMs >= _pauseMs) {
        if (speechMs >= _minSpeechMs) onUtterance(Float32List.fromList(_buf));
        _buf.clear();
        speaking = false;
        speechMs = 0;
        silenceMs = 0;
      }
    });
  }

  Future<void> stopLive() async {
    await _stop();
    _buf.clear();
  }

  Future<void> dispose() async {
    await _stop();
    await _rec.dispose();
  }
}
