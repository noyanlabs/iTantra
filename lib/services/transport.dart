import 'dart:async';

import 'package:flutter/services.dart';

import 'protocol.dart';

enum LinkState { disconnected, connecting, connected }

class Peer {
  Peer(this.name, this.address);
  final String name;
  final String address;
}

/// WiFi Direct transport via Kotlin channel, with send-queue and auto-reconnect/backoff.
class Transport {
  static const _m = MethodChannel('itantra/p2p');
  static const _e = EventChannel('itantra/p2p/events');

  final peers = StreamController<List<Peer>>.broadcast();
  final state = StreamController<LinkState>.broadcast();
  final incoming = StreamController<Message>.broadcast();
  final logs = StreamController<String>.broadcast();

  LinkState current = LinkState.disconnected;
  String? _lastAddress;
  bool _userDisconnect = false;
  bool _wasConnected = false; // auto-retry only after a real connected session drops
  int _attempt = 0;
  Timer? _retry;
  final _queue = <String>[];

  Transport() {
    _e.receiveBroadcastStream().listen((ev) {
      final m = Map<String, dynamic>.from(ev as Map);
      switch (m['e']) {
        case 'peers':
          peers.add([
            for (final p in (m['peers'] as List))
              Peer(p['name'] as String, p['address'] as String)
          ]);
        case 'state':
          _onState(switch (m['state']) {
            'connected' => LinkState.connected,
            'connecting' => LinkState.connecting,
            _ => LinkState.disconnected,
          });
        case 'log':
          logs.add(m['msg'] as String);
        case 'msg':
          try {
            incoming.add(Message.decode(m['line'] as String));
          } catch (_) {}
      }
    });
  }

  void _onState(LinkState s) {
    current = s;
    state.add(s);
    if (s == LinkState.connected) {
      _attempt = 0;
      _wasConnected = true;
      _retry?.cancel();
      _flush();
    } else if (s == LinkState.disconnected && _wasConnected && !_userDisconnect && _lastAddress != null) {
      _scheduleRetry();
    }
  }

  void _scheduleRetry() {
    _retry?.cancel();
    final delay = Duration(seconds: (1 << _attempt.clamp(0, 5)));
    _attempt++;
    _retry = Timer(delay, () {
      if (current != LinkState.connected) connect(_lastAddress!, retry: true);
    });
  }

  Future<void> discover() => _m.invokeMethod('discover');

  Future<void> connect(String address, {bool retry = false}) async {
    _lastAddress = address;
    _userDisconnect = false;
    if (!retry) _wasConnected = false;
    if (!retry) _attempt = 0;
    final ok = await _m.invokeMethod<bool>('connect', {'address': address});
    // A late failure callback must not override a link that already came up.
    if (ok != true && current == LinkState.connecting) _onState(LinkState.disconnected);
  }

  Future<void> disconnect() async {
    _userDisconnect = true;
    _retry?.cancel();
    await _m.invokeMethod('disconnect');
  }

  /// Queues if not connected; returns true if sent immediately.
  Future<bool> send(Message msg, {bool compress = false}) async {
    final line = Message.encode(msg, compress: compress);
    // Native side knows whether a socket exists; don't gate on the Dart-side state.
    if (await _m.invokeMethod<bool>('send', {'line': line}) == true) return true;
    _queue.add(line);
    return false;
  }

  int get queued => _queue.length;

  Future<void> _flush() async {
    while (_queue.isNotEmpty) {
      if (await _m.invokeMethod<bool>('send', {'line': _queue.first}) != true) break;
      _queue.removeAt(0);
    }
  }

  Future<void> startAlarm() => _m.invokeMethod('startAlarm');
  Future<void> stopAlarm() => _m.invokeMethod('stopAlarm');
}
