import 'dart:io';

import 'package:flutter/services.dart';
import 'package:path_provider/path_provider.dart';

/// Copies bundled assets under [prefix] to app storage once (native libs need real paths).
Future<String> ensureAssetsOnDisk(String prefix) async {
  final dir = await getApplicationSupportDirectory();
  final root = '${dir.path}/$prefix';
  final marker = File('$root/.done');
  if (await marker.exists()) return root;
  final manifest = await AssetManifest.loadFromAssetBundle(rootBundle);
  for (final key in manifest.listAssets().where((a) => a.startsWith('assets/$prefix/'))) {
    final out = File('$root/${key.substring('assets/$prefix/'.length)}');
    await out.parent.create(recursive: true);
    final data = await rootBundle.load(key);
    await out.writeAsBytes(data.buffer.asUint8List(data.offsetInBytes, data.lengthInBytes));
  }
  await marker.writeAsString('ok');
  return root;
}
