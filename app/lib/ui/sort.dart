import 'package:flutter/material.dart';

import '../data/models.dart';
import '../data/settings.dart';
import '../i18n.dart';
import 'format.dart';

/// Ordre d'affichage des listes de produits et de personnes.
enum ListSort { name, recentAdded, recentActivity, amount }

extension ListSortLabel on ListSort {
  IconData get icon => switch (this) {
    ListSort.name => Icons.sort_by_alpha,
    ListSort.recentAdded => Icons.fiber_new_outlined,
    ListSort.recentActivity => Icons.history,
    ListSort.amount => Icons.payments_outlined,
  };

  /// [products] : libellés adaptés à la liste du stock.
  String label({required bool products}) => switch (this) {
    ListSort.name => t('Nom (A → Z)', 'الاسم (أ ← ي)'),
    ListSort.recentAdded => t('Derniers ajoutés', 'آخر المضافين'),
    ListSort.recentActivity =>
      products ? t('Dernier mouvement', 'آخر حركة') : t('Dernière opération', 'آخر عملية'),
    ListSort.amount =>
      products
          ? t('Valeur la plus élevée', 'الأعلى قيمة')
          : t('Montant le plus élevé', 'الأعلى مبلغاً'),
  };
}

int _byName(String a, String b) => normalizeSearch(a).compareTo(normalizeSearch(b));

List<Product> sortProducts(List<Product> list, ListSort sort) => [...list]
  ..sort((a, b) {
    final c = switch (sort) {
      ListSort.name => 0,
      ListSort.recentAdded => b.createdAt.compareTo(a.createdAt),
      ListSort.recentActivity => b.lastActivity.compareTo(a.lastActivity),
      ListSort.amount => b.stockValue.compareTo(a.stockValue),
    };
    return c != 0 ? c : _byName(a.name, b.name);
  });

List<Party> sortParties(List<Party> list, ListSort sort) => [...list]
  ..sort((a, b) {
    final c = switch (sort) {
      ListSort.name => 0,
      ListSort.recentAdded => b.createdAt.compareTo(a.createdAt),
      ListSort.recentActivity => b.lastActivity.compareTo(a.lastActivity),
      ListSort.amount => b.balance.abs().compareTo(a.balance.abs()),
    };
    return c != 0 ? c : _byName(a.name, b.name);
  });

/// Date affichée sur une ligne quand la liste est triée par date (sinon null).
String? sortDateLabel(
  ListSort sort,
  DateTime created,
  DateTime activity, {
  required bool products,
}) => switch (sort) {
  ListSort.recentAdded => '${t('Ajout', 'الإضافة')} : ${fmtShortDateTime(created)}',
  ListSort.recentActivity =>
    '${products ? t('Mouvement', 'حركة') : t('Opération', 'عملية')} : ${fmtShortDateTime(activity)}',
  _ => null,
};

/// Bouton « Trier » de la barre du haut ; le choix est mémorisé pour chaque liste.
class SortButton extends StatelessWidget {
  const SortButton({super.key, required this.listKey, required this.products});

  final String listKey;
  final bool products;

  @override
  Widget build(BuildContext context) {
    final notifier = AppSettings.instance.sortFor(listKey);
    return ValueListenableBuilder<ListSort>(
      valueListenable: notifier,
      builder: (context, current, _) => PopupMenuButton<ListSort>(
        tooltip: t('Trier', 'ترتيب'),
        icon: Icon(current == ListSort.name ? Icons.sort : current.icon),
        initialValue: current,
        onSelected: (s) => AppSettings.instance.setSort(listKey, s),
        itemBuilder: (_) => [
          for (final s in ListSort.values)
            PopupMenuItem(
              value: s,
              child: Row(
                children: [
                  Icon(
                    s.icon,
                    size: 20,
                    color: s == current ? Theme.of(context).colorScheme.primary : null,
                  ),
                  const SizedBox(width: 12),
                  Expanded(child: Text(s.label(products: products))),
                  if (s == current) const Icon(Icons.check, size: 18),
                ],
              ),
            ),
        ],
      ),
    );
  }
}
