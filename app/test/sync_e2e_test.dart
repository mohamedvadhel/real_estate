// Test de bout en bout avec un serveur lancé localement :
//   (cd server && DATABASE_URL=... APP_KEY=secret PORT=3999 node scripts/local-server.mjs)
//   SYNC_TEST_URL=http://localhost:3999 flutter test test/sync_e2e_test.dart
import 'dart:io';

import 'package:compte_boutique/data/db.dart';
import 'package:compte_boutique/data/repo.dart';
import 'package:compte_boutique/sync/sync_service.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:sqflite_common_ffi/sqflite_ffi.dart';

final _url = Platform.environment['SYNC_TEST_URL'];

/// Simule un téléphone : sa propre base et ses propres préférences.
Future<(Repo, SyncService)> phone(String name) async {
  SharedPreferences.setMockInitialValues({});
  final dir = await Directory.systemTemp.createTemp('phone_$name');
  final db = await AppDb.open(path: '${dir.path}/db.sqlite', factory: databaseFactoryFfiNoIsolate);
  final sync = await SyncService.init(db);
  await sync.configure(_url!, 'secret');
  return (Repo(db), sync);
}

void main() {
  sqfliteFfiInit();

  test('deux téléphones partagent les mêmes données', skip: _url == null, () async {
    final (repoA, syncA) = await phone('a');
    expect(await syncA.test(), isNull);
    final riz = await repoA.saveProduct(name: 'Riz', unitId: 'u-kg', purchasePrice: 30, qty: 100);
    final ali = await repoA.saveParty(name: 'Ali', kind: 'client', initialKind: 'credit', initialAmount: 2000);
    await repoA.setAccountBalance('a-cash', 5000);
    expect(await syncA.sync(), isNull);
    final dirty = await AppDb.instance.db.rawQuery('SELECT COUNT(*) n FROM products WHERE dirty = 1');
    expect(dirty.first['n'], 0);

    // Le téléphone B (ou une réinstallation) récupère tout
    final (repoB, syncB) = await phone('b');
    expect(await syncB.sync(), isNull);
    var s = await repoB.summary();
    expect(s.stockValue, 3000);
    expect(s.receivables, 2000);
    expect(s.cash, 5000);
    expect((await repoB.units()).length, 13); // pas de doublon des unités de départ

    // B modifie ; A récupère
    await repoB.addDebtEntry(ali, 'recu', 500);
    await repoB.setStockQty(riz, 90);
    expect(await syncB.sync(), isNull);
    expect(await syncA.sync(), isNull);
    s = await repoA.summary();
    expect(s.receivables, 1500);
    expect(s.stockValue, 2700);

    // Suppression propagée
    await repoA.deleteParty(ali);
    expect(await syncA.sync(), isNull);
    expect(await syncB.sync(), isNull);
    expect((await repoB.summary()).receivables, 0);
  });
}
