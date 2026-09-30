import 'dart:convert';
import 'dart:io';

/// Wire format: one JSON object per line. Text: t, z (1 = gzip+base64), d, ts. Emergency: t, ts.
class Message {
  Message(this.type, {this.text = '', this.sentAtMs = 0});
  final String type; // 'text' | 'emergency'
  final String text;
  final int sentAtMs;

  static String encode(Message m, {bool compress = false}) {
    final ts = DateTime.now().millisecondsSinceEpoch;
    if (m.type == 'emergency') return jsonEncode({'t': 'emergency', 'ts': ts});
    final d = compress
        ? base64Encode(gzip.encode(utf8.encode(m.text)))
        : m.text;
    return jsonEncode({'t': 'text', 'z': compress ? 1 : 0, 'd': d, 'ts': ts});
  }

  static Message decode(String line) {
    final j = jsonDecode(line) as Map<String, dynamic>;
    final ts = (j['ts'] as num?)?.toInt() ?? 0;
    if (j['t'] == 'emergency') return Message('emergency', sentAtMs: ts);
    final d = j['d'] as String;
    final text = j['z'] == 1 ? utf8.decode(gzip.decode(base64Decode(d))) : d;
    return Message('text', text: text, sentAtMs: ts);
  }
}
