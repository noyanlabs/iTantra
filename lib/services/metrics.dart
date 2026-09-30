import 'dart:async';
import 'dart:io';

/// RAM (RSS) and CPU% of this process, sampled from /proc.
class Metrics {
  double cpuPercent = 0;
  double ramMb = 0;
  Timer? _t;
  int _lastTicks = 0;
  int _lastMs = 0;

  void start(void Function() onUpdate) {
    _t = Timer.periodic(const Duration(seconds: 1), (_) async {
      ramMb = ProcessInfo.currentRss / (1024 * 1024);
      try {
        final f = (await File('/proc/self/stat').readAsString());
        final parts = f.substring(f.lastIndexOf(')') + 2).split(' ');
        final ticks = int.parse(parts[11]) + int.parse(parts[12]); // utime+stime
        final now = DateTime.now().millisecondsSinceEpoch;
        if (_lastMs != 0) {
          cpuPercent = (ticks - _lastTicks) * 10 / (now - _lastMs) * 100; // 100 Hz ticks
        }
        _lastTicks = ticks;
        _lastMs = now;
      } catch (_) {}
      onUpdate();
    });
  }

  void stop() => _t?.cancel();
}
