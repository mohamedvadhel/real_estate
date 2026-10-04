import 'dart:async';
import 'dart:convert';

import 'package:flutter/foundation.dart';
import 'package:http/http.dart' as http;
import 'package:shared_preferences/shared_preferences.dart';
import 'package:sqflite/sqflite.dart';

import '../data/db.dart';

class SyncState {
  const SyncState({this.running = false, this.lastSync, this.error});

  final bool running;
  final DateTime? lastSync;
  final String? error;
}

/// Synchronisation avec le serveur (Vercel + Neon).
/// Envoie les lignes modifiées localement, puis récupère celles du serveur.
/// En cas de conflit, la version modifiée le plus récemment gagne.
class SyncService {
  SyncService._(this._db, this._prefs) {
    final last = _prefs.getInt(_kLastSync);
    state.value = SyncState(lastSync: last == null ? null : DateTime.fromMillisecondsSinceEpoch(last));
    _db.onLocalWrite = _scheduleAuto;
  }

  static SyncService? _instance;
  static SyncService get instance => _instance!;

  static Future<SyncService> init(AppDb db) async =>
      _instance = SyncService._(db, await SharedPreferences.getInstance());

  static const _kUrl = 'server_url';
  static const _kKey = 'app_key';
  static const _kCursor = 'sync_cursor';
  static const _kLastSync = 'last_sync';

  final AppDb _db;
  final SharedPreferences _prefs;
  final ValueNotifier<SyncState> state = ValueNotifier(const SyncState());
  Timer? _debounce;
  Future<String?>? _current;

  String get serverUrl => _prefs.getString(_kUrl) ?? '';
  String get appKey => _prefs.getString(_kKey) ?? '';
  bool get configured => serverUrl.isNotEmpty && appKey.isNotEmpty;

  Future<void> configure(String url, String key) async {
    var u = url.trim();
    while (u.endsWith('/')) {
      u = u.substring(0, u.length - 1);
    }
    if (u.isNotEmpty && !u.startsWith('http')) u = 'https://$u';
    await _prefs.setString(_kUrl, u);
    await _prefs.setString(_kKey, key.trim());
  }

  void _scheduleAuto() {
    if (!configured) return;
    _debounce?.cancel();
    _debounce = Timer(const Duration(seconds: 5), () => sync());
  }

  Map<String, String> get _headers => {'Content-Type': 'application/json', 'x-app-key': appKey};

  /// Vérifie l'adresse et la clé. Renvoie null si tout va bien, sinon le message d'erreur.
  Future<String?> test() async {
    try {
      final r = await http
          .get(Uri.parse('$serverUrl/api/health'), headers: _headers)
          .timeout(const Duration(seconds: 20));
      return r.statusCode == 200 ? null : _errorOf(r);
    } catch (e) {
      return 'Connexion impossible : $e';
    }
  }

  /// Lance une synchronisation (une seule à la fois). Renvoie null si succès.
  Future<String?> sync() {
    if (!configured) return Future.value('Serveur non configuré (voir Réglages)');
    return _current ??= _run().whenComplete(() => _current = null);
  }

  Future<String?> _run() async {
    state.value = SyncState(running: true, lastSync: state.value.lastSync);
    String? error;
    try {
      var since = _prefs.getInt(_kCursor) ?? 0;
      var first = true;
      var more = true;
      while (more) {
        // On n'envoie les modifications locales qu'au premier tour.
        final pushed = first ? await _dirtyRows() : <String, List<Map<String, Object?>>>{};
        first = false;
        final r = await http
            .post(Uri.parse('$serverUrl/api/sync'),
                headers: _headers, body: jsonEncode({'since': since, 'changes': pushed}))
            .timeout(const Duration(seconds: 60));
        if (r.statusCode != 200) throw _errorOf(r);
        final body = jsonDecode(r.body) as Map<String, dynamic>;
        await _markClean(pushed);
        await _apply((body['changes'] as Map?)?.cast<String, dynamic>() ?? {});
        since = (body['cursor'] as num?)?.toInt() ?? since;
        await _prefs.setInt(_kCursor, since);
        more = body['more'] == true;
      }
      await _prefs.setInt(_kLastSync, DateTime.now().millisecondsSinceEpoch);
      _db.notify(local: false);
    } catch (e) {
      error = e is String ? e : 'Hors ligne ou serveur injoignable';
      debugPrint('Synchro échouée : $e');
    }
    final last = _prefs.getInt(_kLastSync);
    state.value = SyncState(
      lastSync: last == null ? null : DateTime.fromMillisecondsSinceEpoch(last),
      error: error,
    );
    return error;
  }

  String _errorOf(http.Response r) {
    try {
      return (jsonDecode(r.body) as Map)['error']?.toString() ?? 'Erreur ${r.statusCode}';
    } catch (_) {
      return 'Erreur serveur ${r.statusCode}';
    }
  }

  Future<Map<String, List<Map<String, Object?>>>> _dirtyRows() async {
    final out = <String, List<Map<String, Object?>>>{};
    for (final table in syncTables.keys) {
      final rows = await _db.db.query(table, where: 'dirty = 1');
      if (rows.isEmpty) continue;
      out[table] = rows.map((r) => Map<String, Object?>.of(r)..remove('dirty')).toList();
    }
    return out;
  }

  Future<void> _markClean(Map<String, List<Map<String, Object?>>> pushed) async {
    final batch = _db.db.batch();
    pushed.forEach((table, rows) {
      for (final r in rows) {
        // Si la ligne a encore changé pendant l'envoi, elle reste à synchroniser.
        batch.update(table, {'dirty': 0},
            where: 'id = ? AND updated_at = ?', whereArgs: [r['id'], r['updated_at']]);
      }
    });
    await batch.commit(noResult: true);
  }

  Future<void> _apply(Map<String, dynamic> changes) async {
    await _db.db.transaction((txn) async {
      for (final entry in changes.entries) {
        final cols = syncTables[entry.key];
        if (cols == null) continue;
        for (final raw in (entry.value as List).cast<Map<String, dynamic>>()) {
          final row = <String, Object?>{
            'id': raw['id'],
            'created_at': (raw['created_at'] as num?)?.toInt() ?? 0,
            'updated_at': (raw['updated_at'] as num?)?.toInt() ?? 0,
            'deleted': (raw['deleted'] as num?)?.toInt() ?? 0,
            'dirty': 0,
            for (final c in cols.entries)
              c.key: c.value == 'INTEGER' ? (raw[c.key] as num?)?.toInt() : raw[c.key],
          };
          final local = await txn.query(entry.key,
              columns: ['updated_at', 'dirty'], where: 'id = ?', whereArgs: [row['id']]);
          if (local.isNotEmpty) {
            final localUpdated = local.first['updated_at'] as int;
            final localDirty = local.first['dirty'] == 1;
            // Une modification locale plus récente, pas encore envoyée, est conservée.
            if (localDirty && localUpdated > (row['updated_at'] as int)) continue;
          }
          await txn.insert(entry.key, row, conflictAlgorithm: ConflictAlgorithm.replace);
        }
      }
    });
  }
}
