import 'dart:async';

import 'package:audioplayers/audioplayers.dart';
import 'package:flutter/material.dart';
import 'package:permission_handler/permission_handler.dart';

import 'services/metrics.dart';
import 'services/protocol.dart';
import 'services/recorder.dart';
import 'services/stt_service.dart';
import 'services/transport.dart';
import 'services/tts_service.dart';

void main() => runApp(MaterialApp(
      theme: ThemeData(colorSchemeSeed: Colors.indigo, useMaterial3: true),
      home: const HomePage(),
    ));

const languages = [
  'Hindi', 'Gujarati', 'Marathi', 'Kannada', 'Malayalam',
  'Tamil', 'Telugu', 'Odia', 'Bengali', 'English',
];

class Chat {
  Chat(this.text, this.mine);
  final String text;
  final bool mine;
}

class HomePage extends StatefulWidget {
  const HomePage({super.key});
  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  final _stt = SttService();
  final _tts = TtsService();
  final _rec = Recorder();
  final _player = AudioPlayer();
  final _net = Transport();
  final _metrics = Metrics();
  final _scroll = ScrollController();
  final _chat = <Chat>[];
  final _speakQueue = <String>[];
  final _logs = <String>[];
  bool _speaking = false;

  String _language = 'Hindi'; // non-Hindi entries keep the Hindi pipeline underneath
  bool _live = false; // false = walkie-talkie
  bool _ready = false;
  bool _listening = false;
  bool _emergency = false;
  String _status = 'Loading models...';
  LinkState _link = LinkState.disconnected;
  int? _sttMs, _ttsMs, _linkMs;

  @override
  void initState() {
    super.initState();
    _init();
  }

  Future<void> _init() async {
    await [Permission.microphone, Permission.locationWhenInUse, Permission.nearbyWifiDevices]
        .request();
    _metrics.start(() => mounted ? setState(() {}) : null);
    _net.state.stream.listen((s) => setState(() => _link = s));
    _net.incoming.stream.listen(_onMessage);
    _net.logs.stream.listen((l) => setState(() {
          _logs.add(l);
          if (_logs.length > 5) _logs.removeAt(0);
        }));
    _player.onPlayerComplete.listen((_) {
      _speaking = false;
      _drainSpeech();
    });
    try {
      await _tts.init();
      await _stt.init();
      setState(() {
        _ready = true;
        _status = 'Ready';
      });
    } catch (e) {
      setState(() => _status = 'Init error: $e');
    }
  }

  // ---------- incoming ----------
  void _onMessage(Message m) async {
    if (m.type == 'emergency') {
      setState(() => _emergency = true);
      await _net.startAlarm();
      return;
    }
    _linkMs = DateTime.now().millisecondsSinceEpoch - m.sentAtMs; // needs roughly synced clocks
    _add(m.text, false);
    _speakQueue.add(m.text);
    _drainSpeech();
  }

  Future<void> _drainSpeech() async {
    if (_speaking || _speakQueue.isEmpty) return;
    _speaking = true;
    try {
      final sw = Stopwatch()..start();
      final path = await _tts.synthesizeToWav(_speakQueue.removeAt(0));
      setState(() => _ttsMs = sw.elapsedMilliseconds);
      // Force speaker route + media focus: the mic session can leave the route on earpiece/call mode.
      await _player.setAudioContext(AudioContextConfig(
        route: AudioContextConfigRoute.speaker,
        focus: AudioContextConfigFocus.gain,
        respectSilence: false,
      ).build());
      await _player.stop();
      await _player.setVolume(1.0);
      await _player.play(DeviceFileSource(path));
    } catch (e) {
      _speaking = false; // don't stay stuck after a failure
      setState(() => _status = 'Playback error: $e');
      _drainSpeech();
    }
  }

  // ---------- outgoing ----------
  Future<void> _sendUtterance(dynamic samples) async {
    final sw = Stopwatch()..start();
    final text = _stt.transcribe(samples);
    setState(() => _sttMs = sw.elapsedMilliseconds);
    if (text.isEmpty) return;
    _add(text, true);
    // Smart compression (gzip) only in walkie-talkie mode, per spec.
    final sent = await _net.send(Message('text', text: text), compress: !_live);
    if (!sent) setState(() => _status = 'Not connected — queued (${_net.queued})');
  }

  Future<void> _pttDown() async {
    if (!_ready || _live) return;
    if (await _rec.startPushToTalk()) setState(() => _listening = true);
  }

  Future<void> _pttUp() async {
    if (!_listening || _live) return;
    setState(() => _listening = false);
    await _sendUtterance(await _rec.stopPushToTalk());
  }

  Future<void> _toggleLive() async {
    if (!_ready) return;
    if (_listening) {
      await _rec.stopLive();
      setState(() => _listening = false);
    } else if (await _rec.startLive((u) => _sendUtterance(u))) {
      setState(() => _listening = true);
    }
  }

  Future<void> _sendEmergency() async {
    await _net.send(Message('emergency'));
    setState(() => _status = 'Emergency alert sent');
  }

  void _add(String t, bool mine) {
    setState(() => _chat.add(Chat(t, mine)));
    Future.delayed(const Duration(milliseconds: 50), () {
      if (_scroll.hasClients) _scroll.jumpTo(_scroll.position.maxScrollExtent);
    });
  }

  // ---------- connection sheet ----------
  void _showConnect() {
    _net.discover();
    showModalBottomSheet(
      context: context,
      builder: (_) => StreamBuilder<List<Peer>>(
        stream: _net.peers.stream,
        builder: (_, snap) {
          final peers = snap.data ?? [];
          return ListView(children: [
            const ListTile(title: Text('Nearby devices (WiFi Direct)')),
            if (peers.isEmpty) const ListTile(title: Text('Searching...')),
            for (final p in peers)
              ListTile(
                leading: const Icon(Icons.phone_android),
                title: Text(p.name),
                onTap: () {
                  _net.connect(p.address);
                  Navigator.pop(context);
                },
              ),
            if (_link != LinkState.disconnected)
              ListTile(
                leading: const Icon(Icons.link_off),
                title: const Text('Disconnect'),
                onTap: () {
                  _net.disconnect();
                  Navigator.pop(context);
                },
              ),
          ]);
        },
      ),
    );
  }

  @override
  void dispose() {
    _metrics.stop();
    _rec.dispose();
    _stt.dispose();
    _tts.dispose();
    _player.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    if (_emergency) return _emergencyScreen();
    final linkColor = switch (_link) {
      LinkState.connected => Colors.green,
      LinkState.connecting => Colors.orange,
      _ => Colors.grey,
    };
    return Scaffold(
      appBar: AppBar(
        title: const Text('iTantra'),
        actions: [
          TextButton.icon(
            onPressed: _showConnect,
            icon: Icon(Icons.wifi_tethering, color: linkColor),
            label: Text(_link.name),
          ),
        ],
      ),
      body: Column(children: [
        _metricsBar(),
        Padding(
          padding: const EdgeInsets.all(8),
          child: DropdownButton<String>(
            value: _language,
            items: [for (final l in languages) DropdownMenuItem(value: l, child: Text(l))],
            onChanged: (v) => setState(() => _language = v!),
          ),
        ),
        Expanded(
          child: ListView.builder(
            controller: _scroll,
            padding: const EdgeInsets.all(12),
            itemCount: _chat.length,
            itemBuilder: (_, i) {
              final c = _chat[i];
              return Align(
                alignment: c.mine ? Alignment.centerRight : Alignment.centerLeft,
                child: Container(
                  margin: const EdgeInsets.symmetric(vertical: 4),
                  padding: const EdgeInsets.all(10),
                  constraints: const BoxConstraints(maxWidth: 300),
                  decoration: BoxDecoration(
                    color: c.mine ? Colors.indigo.shade100 : Colors.grey.shade200,
                    borderRadius: BorderRadius.circular(12),
                  ),
                  child: Text(c.text, style: const TextStyle(fontSize: 18)),
                ),
              );
            },
          ),
        ),
        Text(_status, style: const TextStyle(fontSize: 12)),
        Text('link: ${_link.name}\n${_logs.join('\n')}',
            style: const TextStyle(fontSize: 11, color: Colors.grey)),
        SegmentedButton<bool>(
          segments: const [
            ButtonSegment(value: false, label: Text('Walkie-Talkie')),
            ButtonSegment(value: true, label: Text('Live')),
          ],
          selected: {_live},
          onSelectionChanged: _listening ? null : (s) => setState(() => _live = s.first),
        ),
        Padding(
          padding: const EdgeInsets.symmetric(vertical: 16),
          child: Row(mainAxisAlignment: MainAxisAlignment.spaceEvenly, children: [
            const SizedBox(width: 72),
            GestureDetector(
              onTapDown: (_) => _pttDown(),
              onTapUp: (_) => _pttUp(),
              onTapCancel: _pttUp,
              onTap: _live ? _toggleLive : null,
              child: CircleAvatar(
                radius: 44,
                backgroundColor: _listening ? Colors.redAccent : Colors.indigo,
                child: Icon(_listening ? Icons.mic : Icons.mic_none,
                    size: 44, color: Colors.white),
              ),
            ),
            SizedBox(
              width: 72,
              height: 72,
              child: FilledButton(
                style: FilledButton.styleFrom(
                    backgroundColor: Colors.red.shade900,
                    shape: const RoundedRectangleBorder(
                        borderRadius: BorderRadius.all(Radius.circular(8)))),
                onPressed: _sendEmergency,
                child: const Text('SOS', style: TextStyle(fontWeight: FontWeight.bold)),
              ),
            ),
          ]),
        ),
      ]),
    );
  }

  Widget _metricsBar() {
    String v(int? x) => x == null ? '-' : '$x ms';
    return Container(
      color: Colors.black87,
      padding: const EdgeInsets.all(8),
      width: double.infinity,
      child: Text(
        'STT ${v(_sttMs)} | TTS ${v(_ttsMs)} | Link ${v(_linkMs)}\n'
        'RAM ${_metrics.ramMb.toStringAsFixed(0)} MB | CPU ${_metrics.cpuPercent.toStringAsFixed(0)}%',
        style: const TextStyle(color: Colors.greenAccent, fontFamily: 'monospace'),
      ),
    );
  }

  Widget _emergencyScreen() => Scaffold(
        backgroundColor: Colors.red,
        body: Center(
          child: Column(mainAxisSize: MainAxisSize.min, children: [
            const Icon(Icons.warning_amber, size: 120, color: Colors.white),
            const Text('EMERGENCY ALERT',
                style: TextStyle(fontSize: 32, color: Colors.white, fontWeight: FontWeight.bold)),
            const SizedBox(height: 24),
            FilledButton(
              style: FilledButton.styleFrom(backgroundColor: Colors.white, foregroundColor: Colors.red),
              onPressed: () async {
                await _net.stopAlarm();
                setState(() => _emergency = false);
              },
              child: const Text('DISMISS'),
            ),
          ]),
        ),
      );
}
