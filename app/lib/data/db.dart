import 'package:flutter/foundation.dart';
import 'package:path/path.dart' as p;
import 'package:sqflite/sqflite.dart';
import 'package:uuid/uuid.dart';

/// Tables synchronisées (mêmes colonnes que sur le serveur, voir server/lib/tables.js).
/// Chaque table a aussi : id, created_at, updated_at, deleted, dirty (local seulement).
const Map<String, Map<String, String>> syncTables = {
  'units': {'name': 'TEXT', 'symbol': 'TEXT', 'allow_decimal': 'INTEGER'},
  'products': {
    'name': 'TEXT',
    'category': 'TEXT',
    'unit_id': 'TEXT',
    'purchase_price': 'REAL',
    'sale_price': 'REAL',
    'min_stock': 'REAL',
    'note': 'TEXT',
  },
  'stock_movements': {
    'product_id': 'TEXT',
    'qty': 'REAL',
    'kind': 'TEXT',
    'unit_cost': 'REAL',
    'date': 'INTEGER',
    'note': 'TEXT',
  },
  'parties': {'name': 'TEXT', 'kind': 'TEXT', 'phone': 'TEXT', 'note': 'TEXT'},
  'debt_entries': {
    'party_id': 'TEXT',
    'amount': 'REAL',
    'kind': 'TEXT',
    'date': 'INTEGER',
    'note': 'TEXT',
  },
  'accounts': {'name': 'TEXT', 'kind': 'TEXT', 'position': 'INTEGER'},
  'account_movements': {
    'account_id': 'TEXT',
    'amount': 'REAL',
    'kind': 'TEXT',
    'date': 'INTEGER',
    'note': 'TEXT',
  },
};

const _seedUnits = [
  ['u-piece', 'Pièce', 'pce', 0],
  ['u-kg', 'Kilogramme', 'kg', 1],
  ['u-g', 'Gramme', 'g', 1],
  ['u-l', 'Litre', 'L', 1],
  ['u-m', 'Mètre', 'm', 1],
  ['u-sac', 'Sac', 'sac', 1],
  ['u-carton', 'Carton', 'carton', 1],
  ['u-paquet', 'Paquet', 'paquet', 1],
  ['u-boite', 'Boîte', 'boîte', 1],
  ['u-bidon', 'Bidon', 'bidon', 1],
  ['u-sachet', 'Sachet', 'sachet', 1],
  ['u-plateau', 'Plateau', 'plateau', 1],
  ['u-douzaine', 'Douzaine', 'dz', 1],
];

const _seedAccounts = [
  ['a-cash', 'Caisse (espèces)', 'cash', 0],
  ['a-bankily', 'Bankily', 'wallet', 1],
  ['a-masrvi', 'Masrvi', 'wallet', 2],
  ['a-sedad', 'Sedad', 'wallet', 3],
];

/// Accès à la base SQLite locale. Toutes les écritures passent par ici :
/// elles mettent à jour `updated_at`, marquent la ligne à synchroniser
/// et préviennent l'interface via [changes].
class AppDb {
  AppDb._(this.db);

  static AppDb? _instance;
  static AppDb get instance => _instance!;

  final Database db;

  /// Incrémenté à chaque modification locale ou synchronisation.
  final ValueNotifier<int> changes = ValueNotifier(0);

  /// Appelé après chaque écriture locale (utilisé pour la synchro automatique).
  VoidCallback? onLocalWrite;

  static const _uuid = Uuid();
  static String newId() => _uuid.v4();
  static int now() => DateTime.now().millisecondsSinceEpoch;

  static Future<AppDb> open({String? path, DatabaseFactory? factory}) async {
    final f = factory ?? databaseFactory;
    final dbPath = path ?? p.join(await f.getDatabasesPath(), 'compte_boutique.db');
    final db = await f.openDatabase(
      dbPath,
      options: OpenDatabaseOptions(version: 1, onCreate: (db, _) => _create(db)),
    );
    return _instance = AppDb._(db);
  }

  static Future<void> _create(Database db) async {
    final batch = db.batch();
    syncTables.forEach((table, cols) {
      final defs = cols.entries.map((e) => '${e.key} ${e.value}').join(', ');
      batch.execute(
        'CREATE TABLE $table (id TEXT PRIMARY KEY, created_at INTEGER NOT NULL, '
        'updated_at INTEGER NOT NULL, deleted INTEGER NOT NULL DEFAULT 0, '
        'dirty INTEGER NOT NULL DEFAULT 1, $defs)',
      );
    });
    batch.execute('CREATE INDEX stock_mv_product ON stock_movements (product_id)');
    batch.execute('CREATE INDEX debt_party ON debt_entries (party_id)');
    batch.execute('CREATE INDEX account_mv_account ON account_movements (account_id)');
    // Données de départ avec des id fixes : pas de doublon entre téléphones.
    for (final u in _seedUnits) {
      batch.insert('units', {
        'id': u[0], 'created_at': 0, 'updated_at': 0, //
        'name': u[1], 'symbol': u[2], 'allow_decimal': u[3],
      });
    }
    for (final a in _seedAccounts) {
      batch.insert('accounts', {
        'id': a[0], 'created_at': 0, 'updated_at': 0, //
        'name': a[1], 'kind': a[2], 'position': a[3],
      });
    }
    await batch.commit(noResult: true);
  }

  void notify({bool local = true}) {
    changes.value++;
    if (local) onLocalWrite?.call();
  }

  /// Insère une nouvelle ligne et renvoie son id.
  Future<String> insert(String table, Map<String, Object?> values, {bool notifyUi = true}) async {
    final t = now();
    final id = (values['id'] as String?) ?? newId();
    await db.insert(table, {
      ...values,
      'id': id,
      'created_at': t,
      'updated_at': t,
      'deleted': 0,
      'dirty': 1,
    });
    if (notifyUi) notify();
    return id;
  }

  Future<void> update(String table, String id, Map<String, Object?> values) async {
    await db.update(
      table,
      {...values, 'updated_at': now(), 'dirty': 1},
      where: 'id = ?',
      whereArgs: [id],
    );
    notify();
  }

  /// Suppression « douce » : la ligne reste pour l'historique et la synchro.
  Future<void> softDelete(String table, String id) => update(table, id, {'deleted': 1});
}
