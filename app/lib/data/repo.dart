import 'db.dart';
import 'models.dart';

/// Lectures et écritures métier. Les quantités et soldes ne sont jamais
/// stockés directement : ils sont la somme des mouvements (historique complet).
/// Note des soldes de départ (traduite à l'affichage).
const initialBalanceNote = 'Solde de départ';

class Repo {
  Repo(this._db);

  final AppDb _db;

  static Repo? _instance;
  static Repo get instance => _instance ??= Repo(AppDb.instance);

  // ---------------------------------------------------------------- Unités

  Future<List<Unit>> units() async {
    final rows = await _db.db.query('units', where: 'deleted = 0', orderBy: 'name COLLATE NOCASE');
    return rows.map(Unit.fromRow).toList();
  }

  Future<String> addUnit(String name, String symbol, {bool allowDecimal = true}) =>
      _db.insert('units', {
        'name': name.trim(),
        'symbol': symbol.trim().isEmpty ? name.trim() : symbol.trim(),
        'allow_decimal': allowDecimal ? 1 : 0,
      });

  Future<void> deleteUnit(String id) => _db.softDelete('units', id);

  /// Nombre de produits actifs qui utilisent cette unité.
  Future<int> unitUsage(String id) async {
    final r = await _db.db.rawQuery(
      'SELECT COUNT(*) AS n FROM products WHERE deleted = 0 AND unit_id = ?',
      [id],
    );
    return (r.first['n'] as int?) ?? 0;
  }

  // -------------------------------------------------------------- Produits

  static const _productSelect = '''
    SELECT p.*, u.symbol AS unit_symbol, u.allow_decimal AS unit_allow_decimal,
      COALESCE((SELECT SUM(m.qty) FROM stock_movements m
                WHERE m.product_id = p.id AND m.deleted = 0), 0) AS qty
    FROM products p LEFT JOIN units u ON u.id = p.unit_id
    WHERE p.deleted = 0''';

  Future<List<Product>> products({String search = ''}) async {
    final s = search.trim();
    final rows = await _db.db.rawQuery(
      '$_productSelect ${s.isEmpty ? '' : 'AND (p.name LIKE ? OR p.category LIKE ?)'} '
      'ORDER BY p.name COLLATE NOCASE',
      s.isEmpty ? [] : ['%$s%', '%$s%'],
    );
    return rows.map(Product.fromRow).toList();
  }

  Future<Product?> product(String id) async {
    final rows = await _db.db.rawQuery('$_productSelect AND p.id = ?', [id]);
    return rows.isEmpty ? null : Product.fromRow(rows.first);
  }

  Future<List<String>> categories() async {
    final rows = await _db.db.rawQuery(
      "SELECT DISTINCT category FROM products WHERE deleted = 0 AND category IS NOT NULL "
      "AND category <> '' ORDER BY category COLLATE NOCASE",
    );
    return rows.map((r) => r['category'] as String).toList();
  }

  /// Crée ou modifie un produit. Si la quantité saisie diffère du stock
  /// actuel, un mouvement « inventaire » est ajouté pour la différence.
  Future<String> saveProduct({
    String? id,
    required String name,
    String? category,
    required String unitId,
    double? purchasePrice,
    double? salePrice,
    double? minStock,
    String? note,
    double? qty,
  }) async {
    final values = {
      'name': name.trim(),
      'category': (category ?? '').trim().isEmpty ? null : category!.trim(),
      'unit_id': unitId,
      'purchase_price': purchasePrice,
      'sale_price': salePrice,
      'min_stock': minStock,
      'note': (note ?? '').trim().isEmpty ? null : note!.trim(),
    };
    double current = 0;
    if (id == null) {
      id = await _db.insert('products', values, notifyUi: false);
    } else {
      current = (await product(id))?.qty ?? 0;
      await _db.update('products', id, values);
    }
    if (qty != null && (qty - current).abs() > 1e-9) {
      await _db.insert('stock_movements', {
        'product_id': id,
        'qty': qty - current,
        'kind': current == 0 && qty > 0 ? 'initial' : 'inventaire',
        'unit_cost': purchasePrice,
        'date': AppDb.now(),
      }, notifyUi: false);
    }
    _db.notify();
    return id;
  }

  Future<void> deleteProduct(String id) => _db.softDelete('products', id);

  Future<void> addStockMovement(
    String productId,
    double qty,
    String kind, {
    double? unitCost,
    String? note,
  }) async {
    await _db.insert('stock_movements', {
      'product_id': productId,
      'qty': qty,
      'kind': kind,
      'unit_cost': unitCost,
      'date': AppDb.now(),
      'note': note,
    });
  }

  Future<void> setStockQty(String productId, double newQty, {String? note}) async {
    final current = (await product(productId))?.qty ?? 0;
    if ((newQty - current).abs() < 1e-9) return;
    await addStockMovement(productId, newQty - current, 'inventaire', note: note);
  }

  Future<List<Movement>> stockMovements(String productId) async {
    final rows = await _db.db.query(
      'stock_movements',
      where: 'deleted = 0 AND product_id = ?',
      whereArgs: [productId],
      orderBy: 'date DESC',
    );
    return rows.map((r) => Movement.fromRow(r, amountKey: 'qty')).toList();
  }

  Future<void> deleteStockMovement(String id) => _db.softDelete('stock_movements', id);

  // ----------------------------------------------------- Clients / dettes

  static const _partySelect = '''
    SELECT p.*, COALESCE((SELECT SUM(d.amount) FROM debt_entries d
              WHERE d.party_id = p.id AND d.deleted = 0), 0) AS balance
    FROM parties p WHERE p.deleted = 0''';

  Future<List<Party>> parties() async {
    final rows = await _db.db.rawQuery('$_partySelect ORDER BY p.name COLLATE NOCASE');
    return rows.map(Party.fromRow).toList();
  }

  Future<Party?> party(String id) async {
    final rows = await _db.db.rawQuery('$_partySelect AND p.id = ?', [id]);
    return rows.isEmpty ? null : Party.fromRow(rows.first);
  }

  Future<String> saveParty({
    String? id,
    required String name,
    required String kind,
    String? phone,
    String? note,
    String? initialKind,
    double? initialAmount,
  }) async {
    final values = {
      'name': name.trim(),
      'kind': kind,
      'phone': (phone ?? '').trim().isEmpty ? null : phone!.trim(),
      'note': (note ?? '').trim().isEmpty ? null : note!.trim(),
    };
    if (id == null) {
      id = await _db.insert('parties', values, notifyUi: false);
    } else {
      await _db.update('parties', id, values);
    }
    if (initialKind != null && initialAmount != null && initialAmount > 0) {
      await addDebtEntry(id, initialKind, initialAmount, note: initialBalanceNote, notifyUi: false);
    }
    _db.notify();
    return id;
  }

  Future<void> deleteParty(String id) => _db.softDelete('parties', id);

  /// [kind] : voir [DebtKind]. Le montant est saisi positif, le signe dépend du type.
  Future<void> addDebtEntry(
    String partyId,
    String kind,
    double amount, {
    String? note,
    int? date,
    bool notifyUi = true,
  }) async {
    await _db.insert('debt_entries', {
      'party_id': partyId,
      'amount': DebtKind.of(kind).sign * amount.abs(),
      'kind': kind,
      'date': date ?? AppDb.now(),
      'note': (note ?? '').trim().isEmpty ? null : note!.trim(),
    }, notifyUi: notifyUi);
  }

  Future<List<Movement>> debtEntries(String partyId) async {
    final rows = await _db.db.query(
      'debt_entries',
      where: 'deleted = 0 AND party_id = ?',
      whereArgs: [partyId],
      orderBy: 'date DESC',
    );
    return rows.map((r) => Movement.fromRow(r, amountKey: 'amount')).toList();
  }

  Future<void> deleteDebtEntry(String id) => _db.softDelete('debt_entries', id);

  // ------------------------------------------------------ Caisse / wallets

  static const _accountSelect = '''
    SELECT a.*, COALESCE((SELECT SUM(m.amount) FROM account_movements m
              WHERE m.account_id = a.id AND m.deleted = 0), 0) AS balance
    FROM accounts a WHERE a.deleted = 0''';

  Future<List<Account>> accounts() async {
    final rows = await _db.db.rawQuery(
      '$_accountSelect ORDER BY a.position, a.name COLLATE NOCASE',
    );
    return rows.map(Account.fromRow).toList();
  }

  Future<Account?> account(String id) async {
    final rows = await _db.db.rawQuery('$_accountSelect AND a.id = ?', [id]);
    return rows.isEmpty ? null : Account.fromRow(rows.first);
  }

  Future<String> saveAccount({String? id, required String name, required String kind}) async {
    if (id != null) {
      await _db.update('accounts', id, {'name': name.trim(), 'kind': kind});
      return id;
    }
    final r = await _db.db.rawQuery('SELECT COALESCE(MAX(position), 0) + 1 AS n FROM accounts');
    return _db.insert('accounts', {'name': name.trim(), 'kind': kind, 'position': r.first['n']});
  }

  Future<void> deleteAccount(String id) => _db.softDelete('accounts', id);

  Future<void> addAccountMovement(
    String accountId,
    String kind,
    double amount, {
    String? note,
  }) async {
    await _db.insert('account_movements', {
      'account_id': accountId,
      'amount': amount,
      'kind': kind,
      'date': AppDb.now(),
      'note': (note ?? '').trim().isEmpty ? null : note!.trim(),
    });
  }

  /// Fixe le solde réel (comptage de la caisse, solde affiché dans le wallet).
  Future<void> setAccountBalance(String accountId, double balance, {String? note}) async {
    final current = (await account(accountId))?.balance ?? 0;
    if ((balance - current).abs() < 1e-9) return;
    await addAccountMovement(accountId, 'solde', balance - current, note: note);
  }

  Future<List<Movement>> accountMovements(String accountId) async {
    final rows = await _db.db.query(
      'account_movements',
      where: 'deleted = 0 AND account_id = ?',
      whereArgs: [accountId],
      orderBy: 'date DESC',
    );
    return rows.map((r) => Movement.fromRow(r, amountKey: 'amount')).toList();
  }

  Future<void> deleteAccountMovement(String id) => _db.softDelete('account_movements', id);

  // ------------------------------------------------------------- Situation

  Future<Summary> summary() async {
    final products = await this.products();
    final parties = await this.parties();
    final accounts = await this.accounts();
    return Summary.compute(products, parties, accounts);
  }
}
