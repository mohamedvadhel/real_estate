import 'package:compte_boutique/data/db.dart';
import 'package:compte_boutique/data/repo.dart';
import 'package:compte_boutique/data/settings.dart';
import 'package:compte_boutique/main.dart';
import 'package:compte_boutique/sync/sync_service.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:sqflite_common_ffi/sqflite_ffi.dart';

void main() {
  sqfliteFfiInit();

  Future<void> settle(WidgetTester tester) async {
    for (var i = 0; i < 5; i++) {
      await tester.runAsync(() => Future.delayed(const Duration(milliseconds: 50)));
      await tester.pump(const Duration(milliseconds: 400));
    }
  }

  testWidgets('après création, la liste « Tous » défile jusqu\'au nouveau client', (tester) async {
    tester.view.physicalSize = const Size(824, 1600);
    tester.view.devicePixelRatio = 2;
    addTearDown(tester.view.reset);
    await tester.runAsync(() async {
      // ignore: invalid_use_of_visible_for_testing_member
      SharedPreferences.setMockInitialValues({});
      final db = await AppDb.open(path: inMemoryDatabasePath, factory: databaseFactoryFfiNoIsolate);
      await AppSettings.init();
      await SyncService.init(db);
      final r = Repo(db);
      for (var i = 0; i < 30; i++) {
        await r.saveParty(
          name: 'Client ${i.toString().padLeft(2, '0')}',
          kind: 'client',
          initialKind: 'credit',
          initialAmount: 100.0 + i,
        );
      }
    });
    await tester.pumpWidget(const CompteBoutiqueApp());
    await settle(tester);
    await tester.tap(find.byIcon(Icons.people_alt_outlined));
    await settle(tester);
    expect(find.text('Zakaria'), findsNothing);

    // + Nouvelle dette → nom + montant → Enregistrer
    await tester.tap(find.byIcon(Icons.person_add_alt));
    await settle(tester);
    await tester.enterText(find.byType(TextFormField).first, 'Zakaria');
    await tester.enterText(find.byType(TextFormField).last, '2500');
    await tester.tap(find.widgetWithText(FilledButton, 'Enregistrer'));
    await settle(tester);

    // De retour sur « Tous », la ligne du nouveau client est visible à l'écran et surlignée.
    final tile = find.text('Zakaria');
    expect(tile, findsOneWidget);
    final screen = tester.getRect(find.byType(Scaffold).first);
    expect(screen.contains(tester.getCenter(tile)), isTrue);
    final row = tester.widget<ListTile>(find.ancestor(of: tile, matching: find.byType(ListTile)));
    expect(row.tileColor, isNotNull);

    // Le surlignage disparaît après quelques secondes.
    await tester.pump(const Duration(seconds: 4));
    await settle(tester);
    expect(
      tester.widget<ListTile>(find.ancestor(of: tile, matching: find.byType(ListTile))).tileColor,
      isNull,
    );
    expect(tester.widget<TabBar>(find.byType(TabBar)).controller!.index, 0);

    await tester.runAsync(() => AppDb.instance.db.close());
  });
}
