import '../i18n.dart';

double _d(Object? v) => (v as num?)?.toDouble() ?? 0;
double? _dn(Object? v) => (v as num?)?.toDouble();
DateTime _dt(Object? v) => DateTime.fromMillisecondsSinceEpoch((v as num?)?.toInt() ?? 0);

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

  String get displayName => seedName(id, name);
  String get displaySymbol => seedSymbol(id, symbol);
  String get label => displaySymbol.isEmpty || displaySymbol == displayName
      ? displayName
      : '$displayName ($displaySymbol)';
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
    required this.createdAt,
    required this.lastActivity,
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
    createdAt: _dt(r['created_at']),
    lastActivity: _dt(r['last_activity'] ?? r['created_at']),
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

  /// Date d'ajout du produit.
  final DateTime createdAt;

  /// Date du dernier mouvement de stock (ou de l'ajout s'il n'y en a pas).
  final DateTime lastActivity;

  /// Valeur du stock au prix d'achat (ce que la marchandise a coûté).
  double get stockValue => qty > 0 ? qty * (purchasePrice ?? 0) : 0;

  /// Valeur du stock si tout est vendu au prix de vente.
  double get saleValue => qty > 0 ? qty * (salePrice ?? 0) : 0;

  /// Symbole de l'unité dans la langue courante.
  String get unit => seedSymbol(unitId, unitSymbol);

  bool get missingPurchasePrice => qty > 0 && (purchasePrice == null || purchasePrice == 0);
  bool get lowStock => minStock != null && minStock! > 0 && qty <= minStock!;
}

/// Type d'opération sur une dette. Solde positif = la personne nous doit,
/// solde négatif = nous lui devons.
class DebtKind {
  const DebtKind(this.code, this.sign);

  final String code;
  final int sign;

  String get label => switch (code) {
    'credit' => t('Il me doit', 'عليه لي'),
    'recu' => t("Il m'a payé", 'دفع لي'),
    'dette' => t('Je lui dois', 'علي له'),
    _ => t("Je l'ai payé", 'دفعت له'),
  };

  static const credit = DebtKind('credit', 1);
  static const recu = DebtKind('recu', -1);
  static const dette = DebtKind('dette', -1);
  static const paye = DebtKind('paye', 1);

  static const all = [credit, recu, dette, paye];

  static DebtKind of(String code) => all.firstWhere((k) => k.code == code, orElse: () => credit);
}

Map<String, String> get partyKinds => {
  'client': t('Client', 'زبون'),
  'fournisseur': t('Fournisseur', 'مورد'),
  'autre': t('Autre', 'آخر'),
};

Map<String, String> get stockKindLabels => {
  'initial': t('Stock de départ', 'المخزون الأولي'),
  'inventaire': t('Correction inventaire', 'تصحيح الجرد'),
  'entree': t('Entrée (achat)', 'دخول (شراء)'),
  'sortie': t('Sortie (vente)', 'خروج (بيع)'),
  'perte': t('Perte / casse', 'تلف / خسارة'),
};

Map<String, String> get accountKinds => {
  'cash': t('Espèces', 'نقداً'),
  'wallet': t('Wallet', 'محفظة إلكترونية'),
  'banque': t('Banque', 'بنك'),
};

Map<String, String> get accountMovementLabels => {
  'solde': t('Ajustement du solde', 'تعديل الرصيد'),
  'entree': t('Entrée', 'دخول'),
  'sortie': t('Sortie', 'خروج'),
};

class Party {
  Party({
    required this.id,
    required this.name,
    required this.kind,
    this.phone,
    this.note,
    required this.balance,
    required this.createdAt,
    required this.lastActivity,
  });

  factory Party.fromRow(Map<String, Object?> r) => Party(
    id: r['id'] as String,
    name: r['name'] as String? ?? '',
    kind: r['kind'] as String? ?? 'client',
    phone: r['phone'] as String?,
    note: r['note'] as String?,
    balance: _d(r['balance']),
    createdAt: _dt(r['created_at']),
    lastActivity: _dt(r['last_activity'] ?? r['created_at']),
  );

  final String id;
  final String name;
  final String kind;
  final String? phone;
  final String? note;

  /// > 0 : il nous doit ; < 0 : nous lui devons.
  final double balance;

  /// Date d'ajout de la personne.
  final DateTime createdAt;

  /// Date de la dernière opération de dette (ou de l'ajout s'il n'y en a pas).
  final DateTime lastActivity;
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

  String get displayName => seedName(id, name);
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

  List<Party> get debtors =>
      parties.where((p) => p.balance > 0.0001).toList()
        ..sort((a, b) => b.balance.compareTo(a.balance));
  List<Party> get creditors =>
      parties.where((p) => p.balance < -0.0001).toList()
        ..sort((a, b) => a.balance.compareTo(b.balance));
  List<Product> get missingPrice => products.where((p) => p.missingPurchasePrice).toList();
  List<Product> get lowStock => products.where((p) => p.lowStock).toList();
}
