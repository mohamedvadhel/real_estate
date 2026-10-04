import 'package:compte_boutique/data/db.dart';
import 'package:compte_boutique/data/models.dart';
import 'package:compte_boutique/data/repo.dart';
import 'package:compte_boutique/report/pdf_report.dart';
import 'package:compte_boutique/ui/format.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:sqflite_common_ffi/sqflite_ffi.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();
  sqfliteFfiInit();

  late Repo repo;

  setUp(() async {
    final db = await AppDb.open(path: inMemoryDatabasePath, factory: databaseFactoryFfiNoIsolate);
    repo = Repo(db);
  });

  tearDown(() => AppDb.instance.db.close());

  group('saisie des nombres', () {
    test('virgule, espaces et calculs', () {
      expect(parseNum('12,5'), 12.5);
      expect(parseNum('12 500'), 12500);
      expect(parseNum(numToInput(12500.5)), 12500.5);
      expect(parseNum('3x50 + 20'), 170);
      expect(parseNum('3*50-5'), 145);
      expect(parseNum('abc'), isNull);
      expect(parseNum(''), isNull);
    });

    test('formatage', () {
      expect(fmtMoney(1234567.5), '1\u00A0234\u00A0567,5 MRU');
      expect(fmtNum(-1500), '-1\u00A0500');
      expect(fmtQty(2.25, 'kg'), '2,25 kg');
    });
  });

  test('données de départ : unités et comptes', () async {
    final units = await repo.units();
    expect(units.map((u) => u.symbol), containsAll(['kg', 'L', 'pce', 'sac']));
    final accounts = await repo.accounts();
    expect(accounts.first.name, 'Caisse (espèces)');
    expect(accounts.map((a) => a.name), containsAll(['Bankily', 'Masrvi', 'Sedad']));
  });

  test('valorisation complète de la boutique', () async {
    // Stock
    final riz = await repo.saveProduct(
      name: 'Riz',
      unitId: 'u-kg',
      purchasePrice: 30,
      salePrice: 35,
      qty: 500,
      category: 'Alimentation',
    );
    await repo.saveProduct(name: 'Huile', unitId: 'u-l', purchasePrice: 80, salePrice: 90, qty: 40);
    await repo.saveProduct(name: 'Savon', unitId: 'u-piece', qty: 10); // sans prix
    // Mouvements : vente de 20 kg, puis inventaire à 470 kg
    await repo.addStockMovement(riz, -20, 'sortie');
    expect((await repo.product(riz))!.qty, 480);
    await repo.setStockQty(riz, 470);
    expect((await repo.product(riz))!.qty, 470);
    expect((await repo.stockMovements(riz)).length, 3);

    // Dettes
    final ali = await repo.saveParty(
      name: 'Ali',
      kind: 'client',
      initialKind: 'credit',
      initialAmount: 5000,
    );
    await repo.addDebtEntry(ali, 'recu', 1500);
    final fournisseur = await repo.saveParty(
      name: 'Grossiste',
      kind: 'fournisseur',
      initialKind: 'dette',
      initialAmount: 12000,
    );
    await repo.addDebtEntry(fournisseur, 'paye', 2000);

    // Caisse
    await repo.setAccountBalance('a-cash', 7000);
    await repo.setAccountBalance('a-bankily', 3000);
    await repo.addAccountMovement('a-bankily', 'sortie', -500);

    final s = await repo.summary();
    expect(s.stockValue, 470 * 30 + 40 * 80);
    expect(s.stockSaleValue, 470 * 35 + 40 * 90);
    expect(s.receivables, 3500);
    expect(s.payables, 10000);
    expect(s.cash, 9500);
    expect(s.netValue, 17300 + 3500 + 9500 - 10000);
    expect(s.missingPrice.map((p) => p.name), ['Savon']);

    // Modifier le produit avec une nouvelle quantité crée un mouvement d'inventaire
    await repo.saveProduct(id: riz, name: 'Riz', unitId: 'u-kg', purchasePrice: 30, qty: 400);
    expect((await repo.product(riz))!.qty, 400);

    // Suppression : le produit ne compte plus
    await repo.deleteProduct(riz);
    expect((await repo.summary()).stockValue, 40 * 80);

    // Toutes les écritures sont à synchroniser
    final dirty = await AppDb.instance.db.rawQuery(
      'SELECT COUNT(*) AS n FROM stock_movements WHERE dirty = 1',
    );
    expect(dirty.first['n'], greaterThan(0));
  });

  test('sens des dettes', () {
    expect(DebtKind.credit.sign, 1);
    expect(DebtKind.recu.sign, -1);
    expect(DebtKind.dette.sign, -1);
    expect(DebtKind.paye.sign, 1);
  });

  test('rapport PDF généré (texte arabe compris)', () async {
    await repo.saveProduct(name: 'Thé vert', unitId: 'u-paquet', purchasePrice: 150, qty: 24);
    await repo.saveParty(name: 'محمد', kind: 'client', initialKind: 'credit', initialAmount: 800);
    final bytes = await buildReport(
      await repo.summary(),
      shopName: 'Boutique test',
      date: DateTime(2026, 10, 5),
    );
    expect(bytes.length, greaterThan(1000));
    expect(String.fromCharCodes(bytes.take(5)), '%PDF-');
  });
}
