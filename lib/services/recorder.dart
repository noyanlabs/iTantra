import 'dart:async';
import 'dart:math';
import 'dart:typed_data';

import 'package:record/record.dart';

/// Mic capture at 16 kHz mono PCM16, with push-to-talk and relative-loudness VAD (live) modes.
class Recorder {
  final _rec = AudioRecorder();
  StreamSubscription<Uint8List>? _sub;
  final _buf = <double>[];

  static const _sampleRate = 16000;
  static const _frameSamples = 320; // 20 ms analysis frames, independent of chunk size

  // --- Live-mode VAD tuning ---
  static const _pauseMs = 350; // quiet this long => end of utterance
  static const _pauseMsLong = 200; // used once the utterance is long
  static const _longUtteranceMs = 8000;
  static const _maxUtteranceMs = 15000; // hard cap
  static const _minSpeechMs = 300; // ignore blips shorter than this
  static const _startFrames = 3; // 60 ms of voiced frames to start
  static const _prerollMs = 500; // audio kept from before speech onset
  static const _minStartRms = 0.008; // absolute floor for speech onset
  static const _startFactor = 3.0; // onset must be 3x the noise floor
  static const _quietRatio = 0.30; // quiet = below 30% of running speech level (~ -10 dB)

  Future<bool> _start(void Function(Float32List chunk) onChunk) async {
    if (!await _rec.hasPermission()) return false;
    final stream = await _rec.startStream(const RecordConfig(
        encoder: AudioEncoder.pcm16bits, sampleRate: _sampleRate, numChannels: 1));
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

  // ---- Live: continuous; cuts an utterance when loudness drops for ~350 ms ----
  Future<bool> startLive(void Function(Float32List utterance) onUtterance) {
    _buf.clear();
    final preroll = <double>[];
    const prerollMax = _sampleRate * _prerollMs ~/ 1000;

    var noise = 0.003; // running noise-floor estimate (RMS)
    var level = 0.0; // running speech level estimate (RMS)
    var speaking = false;
    var voicedRun = 0;
    double speechMs = 0;
    double silenceMs = 0;

    void emit() {
      if (speechMs >= _minSpeechMs) onUtterance(Float32List.fromList(_buf));
      _buf.clear();
      preroll.clear();
      speaking = false;
      voicedRun = 0;
      speechMs = 0;
      silenceMs = 0;
    }

    return _start((chunk) {
      for (var o = 0; o < chunk.length; o += _frameSamples) {
        final end = min(o + _frameSamples, chunk.length);
        final len = end - o;
        var sum = 0.0;
        for (var i = o; i < end; i++) {
          sum += chunk[i] * chunk[i];
        }
        final rms = sqrt(sum / len);
        final ms = len * 1000 / _sampleRate;

        if (!speaking) {
          final voiced = rms > max(_minStartRms, noise * _startFactor);
          if (voiced) {
            voicedRun++;
          } else {
            voicedRun = 0;
            noise = max(0.0005, 0.95 * noise + 0.05 * rms);
          }
          for (var i = o; i < end; i++) {
            preroll.add(chunk[i]);
          }
          if (preroll.length > prerollMax) {
            preroll.removeRange(0, preroll.length - prerollMax);
          }
          if (voicedRun >= _startFrames) {
            speaking = true;
            _buf
              ..clear()
              ..addAll(preroll);
            preroll.clear();
            level = rms;
            speechMs = voicedRun * ms;
            silenceMs = 0;
          }
          continue;
        }

        // Speaking: collect audio and watch for a relative drop in loudness.
        for (var i = o; i < end; i++) {
          _buf.add(chunk[i]);
        }
        final quiet = rms < max(noise * 2, level * _quietRatio);
        if (quiet) {
          silenceMs += ms;
          noise = max(0.0005, 0.98 * noise + 0.02 * rms);
        } else {
          silenceMs = 0;
          speechMs += ms;
          level = rms > level ? 0.7 * level + 0.3 * rms : 0.95 * level + 0.05 * rms;
        }

        final bufMs = _buf.length * 1000 / _sampleRate;
        final pause = bufMs > _longUtteranceMs ? _pauseMsLong : _pauseMs;
        if (silenceMs >= pause || bufMs >= _maxUtteranceMs) emit();
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