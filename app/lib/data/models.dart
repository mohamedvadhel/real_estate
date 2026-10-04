double _d(Object? v) => (v as num?)?.toDouble() ?? 0;
double? _dn(Object? v) => (v as num?)?.toDouble();

class Unit {
  Unit({required this.id, required this.name, required this.symbol, required this.allowDecimal});

  factory Unit.fromRow(Map<String, Object?> r) => Unit(
        id: r['id'] as String,
        name: r['name'] as String? ?? '',
        symbol: r['symbol'] as String? ?? '',
        allowDecimal: (r['allow_decimal'] as int? ?? 1) == 1,
      );

  final String id;
  final String name;
  final String symbol;
  final bool allowDecimal;

  String get label => symbol.isEmpty || symbol == name ? name : '$name ($symbol)';
}

class Product {
  Product({
    required this.id,
    required this.name,
    this.category,
    required this.unitId,
    required this.unitSymbol,
    this.purchasePrice,
    this.salePrice,
    this.minStock,
    this.note,
    required this.qty,
  });

  factory Product.fromRow(Map<String, Object?> r) => Product(
        id: r['id'] as String,
        name: r['name'] as String? ?? '',
        category: r['category'] as String?,
        unitId: r['unit_id'] as String? ?? '',
        unitSymbol: r['unit_symbol'] as String? ?? '',
        purchasePrice: _dn(r['purchase_price']),
        salePrice: _dn(r['sale_price']),
        minStock: _dn(r['min_stock']),
        note: r['note'] as String?,
        qty: _d(r['qty']),
      );

  final String id;
  final String name;
  final String? category;
  final String unitId;
  final String unitSymbol;
  final double? purchasePrice;
  final double? salePrice;
  final double? minStock;
  final String? note;
  final double qty;

  /// Valeur du stock au prix d'achat (ce que la marchandise a coûté).
  double get stockValue => qty > 0 ? qty * (purchasePrice ?? 0) : 0;

  /// Valeur du stock si tout est vendu au prix de vente.
  double get saleValue => qty > 0 ? qty * (salePrice ?? 0) : 0;

  bool get missingPurchasePrice => qty > 0 && (purchasePrice == null || purchasePrice == 0);
  bool get lowStock => minStock != null && minStock! > 0 && qty <= minStock!;
}

/// Type d'opération sur une dette. Solde positif = la personne nous doit,
/// solde négatif = nous lui devons.
class DebtKind {
  const DebtKind(this.code, this.sign, this.label, this.verb);

  final String code;
  final int sign;
  final String label;
  final String verb;

  static const credit = DebtKind('credit', 1, 'Il me doit', 'Ajouter une dette (il me doit)');
  static const recu = DebtKind('recu', -1, "Il m'a payé", "Il m'a payé");
  static const dette = DebtKind('dette', -1, 'Je lui dois', 'Ajouter ce que je lui dois');
  static const paye = DebtKind('paye', 1, "Je l'ai payé", "Je l'ai payé");

  static const all = [credit, recu, dette, paye];

  static DebtKind of(String code) =>
      all.firstWhere((k) => k.code == code, orElse: () => credit);
}

const partyKinds = {'client': 'Client', 'fournisseur': 'Fournisseur', 'autre': 'Autre'};

const stockKindLabels = {
  'initial': 'Stock de départ',
  'inventaire': 'Correction inventaire',
  'entree': 'Entrée (achat)',
  'sortie': 'Sortie (vente)',
  'perte': 'Perte / casse',
};

const accountKinds = {'cash': 'Espèces', 'wallet': 'Wallet', 'banque': 'Banque'};

const accountMovementLabels = {
  'solde': 'Ajustement du solde',
  'entree': 'Entrée',
  'sortie': 'Sortie',
};

class Party {
  Party({
    required this.id,
    required this.name,
    required this.kind,
    this.phone,
    this.note,
    required this.balance,
  });

  factory Party.fromRow(Map<String, Object?> r) => Party(
        id: r['id'] as String,
        name: r['name'] as String? ?? '',
        kind: r['kind'] as String? ?? 'client',
        phone: r['phone'] as String?,
        note: r['note'] as String?,
        balance: _d(r['balance']),
      );

  final String id;
  final String name;
  final String kind;
  final String? phone;
  final String? note;

  /// > 0 : il nous doit ; < 0 : nous lui devons.
  final double balance;
}

class Account {
  Account({
    required this.id,
    required this.name,
    required this.kind,
    required this.position,
    required this.balance,
  });

  factory Account.fromRow(Map<String, Object?> r) => Account(
        id: r['id'] as String,
        name: r['name'] as String? ?? '',
        kind: r['kind'] as String? ?? 'cash',
        position: r['position'] as int? ?? 0,
        balance: _d(r['balance']),
      );

  final String id;
  final String name;
  final String kind;
  final int position;
  final double balance;
}

/// Ligne d'historique (mouvement de stock, de dette ou de caisse).
class Movement {
  Movement({
    required this.id,
    required this.amount,
    required this.kind,
    required this.date,
    this.note,
    this.unitCost,
  });

  factory Movement.fromRow(Map<String, Object?> r, {required String amountKey}) => Movement(
        id: r['id'] as String,
        amount: _d(r[amountKey]),
        kind: r['kind'] as String? ?? '',
        date: DateTime.fromMillisecondsSinceEpoch(r['date'] as int? ?? 0),
        note: r['note'] as String?,
        unitCost: _dn(r['unit_cost']),
      );

  final String id;
  final double amount;
  final String kind;
  final DateTime date;
  final String? note;
  final double? unitCost;
}

/// Situation de la boutique à un instant donné.
class Summary {
  Summary({
    required this.products,
    required this.parties,
    required this.accounts,
    required this.stockValue,
    required this.stockSaleValue,
    required this.receivables,
    required this.payables,
    required this.cash,
  });

  factory Summary.compute(List<Product> products, List<Party> parties, List<Account> accounts) {
    double stock = 0, sale = 0, rec = 0, pay = 0, cash = 0;
    for (final p in products) {
      stock += p.stockValue;
      sale += p.saleValue;
    }
    for (final p in parties) {
      if (p.balance > 0) rec += p.balance;
      if (p.balance < 0) pay += -p.balance;
    }
    for (final a in accounts) {
      cash += a.balance;
    }
    return Summary(
      products: products,
      parties: parties,
      accounts: accounts,
      stockValue: stock,
      stockSaleValue: sale,
      receivables: rec,
      payables: pay,
      cash: cash,
    );
  }

  final List<Product> products;
  final List<Party> parties;
  final List<Account> accounts;
  final double stockValue;
  final double stockSaleValue;
  final double receivables;
  final double payables;
  final double cash;

  /// Ce qui appartient réellement à la boutique.
  double get netValue => stockValue + receivables + cash - payables;

  /// Bénéfice potentiel si tout le stock est vendu au prix de vente.
  double get potentialMargin => stockSaleValue - stockValue;

  List<Party> get debtors => parties.where((p) => p.balance > 0.0001).toList()
    ..sort((a, b) => b.balance.compareTo(a.balance));
  List<Party> get creditors => parties.where((p) => p.balance < -0.0001).toList()
    ..sort((a, b) => a.balance.compareTo(b.balance));
  List<Product> get missingPrice => products.where((p) => p.missingPurchasePrice).toList();
  List<Product> get lowStock => products.where((p) => p.lowStock).toList();
}
