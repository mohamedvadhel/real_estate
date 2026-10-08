import 'package:flutter/material.dart';

import '../../data/models.dart';
import '../../data/repo.dart';
import '../../data/settings.dart';
import '../../i18n.dart';
import '../format.dart';
import '../theme.dart';
import '../sort.dart';
import '../widgets/common.dart';
import 'product_detail_screen.dart';
import 'product_form_screen.dart';

class StockScreen extends StatefulWidget {
  const StockScreen({super.key});

  @override
  State<StockScreen> createState() => _StockScreenState();
}

class _StockScreenState extends State<StockScreen> {
  final _search = TextEditingController();
  String _query = '';

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text(t('Stock', 'المخزون')),
        actions: const [SortButton(listKey: 'products', products: true)],
        bottom: PreferredSize(
          preferredSize: const Size.fromHeight(64),
          child: Padding(
            padding: const EdgeInsets.fromLTRB(16, 0, 16, 10),
            child: TextField(
              controller: _search,
              onChanged: (v) => setState(() => _query = v),
              decoration: InputDecoration(
                hintText: t('Rechercher un produit ou une catégorie', 'ابحث عن منتج أو فئة'),
                prefixIcon: const Icon(Icons.search),
                isDense: true,
                suffixIcon: _query.isEmpty
                    ? null
                    : IconButton(
                        icon: const Icon(Icons.clear),
                        onPressed: () => setState(() {
                          _search.clear();
                          _query = '';
                        }),
                      ),
              ),
            ),
          ),
        ),
      ),
      floatingActionButton: FloatingActionButton.extended(
        heroTag: null,
        onPressed: () =>
            Navigator.push(context, MaterialPageRoute(builder: (_) => const ProductFormScreen())),
        icon: const Icon(Icons.add),
        label: Text(t('Produit', 'منتج')),
      ),
      body: ValueListenableBuilder<ListSort>(
        valueListenable: AppSettings.instance.sortFor('products'),
        builder: (context, sort, _) => Reactive<List<Product>>(
          load: () => Repo.instance.products(search: _query),
          builder: (context, loaded) {
            final products = sortProducts(loaded, sort);
            if (products.isEmpty) {
              return EmptyState(
                icon: Icons.inventory_2_outlined,
                text: _query.isEmpty
                    ? t(
                        'Aucun produit.\nAppuyez sur « + Produit » pour saisir votre stock.',
                        'لا توجد منتجات.\nاضغط على « + منتج » لإدخال مخزونك.',
                      )
                    : t('Aucun résultat.', 'لا توجد نتائج.'),
              );
            }
            final total = products.fold<double>(0, (a, p) => a + p.stockValue);
            return ListView(
              padding: const EdgeInsets.only(bottom: 96),
              children: [
                TotalBanner(
                  label: t('${products.length} produits', '${products.length} منتج'),
                  value: fmtMoney(total),
                ),
                ListCard(
                  children: [for (final p in products) _ProductTile(product: p, sort: sort)],
                ),
              ],
            );
          },
        ),
      ),
    );
  }
}

class _ProductTile extends StatelessWidget {
  const _ProductTile({required this.product, required this.sort});

  final Product product;
  final ListSort sort;

  @override
  Widget build(BuildContext context) {
    final p = product;
    final price = p.purchasePrice == null ? t('prix ?', 'السعر ؟') : fmtMoney(p.purchasePrice!);
    final warn = p.missingPurchasePrice || p.lowStock;
    return ListTile(
      leading: Container(
        width: 44,
        height: 44,
        alignment: Alignment.center,
        decoration: BoxDecoration(
          color: (warn ? const Color(0xFFB45309) : brandColor).withValues(alpha: 0.1),
          borderRadius: BorderRadius.circular(12),
        ),
        child: warn
            ? const Icon(Icons.warning_amber_rounded, color: Color(0xFFB45309))
            : Text(
                p.unit,
                style: const TextStyle(
                  color: brandColor,
                  fontWeight: FontWeight.w700,
                  fontSize: 12,
                ),
                maxLines: 1,
                overflow: TextOverflow.ellipsis,
              ),
      ),
      title: Text(p.name, style: const TextStyle(fontWeight: FontWeight.w600)),
      subtitle: Text(
        [
          '${fmtQty(p.qty, p.unit)} × $price',
          if (p.category != null) p.category!,
          ?sortDateLabel(sort, p.createdAt, p.lastActivity, products: true),
        ].join('\n'),
      ),
      isThreeLine:
          p.category != null ||
          sortDateLabel(sort, p.createdAt, p.lastActivity, products: true) != null,
      trailing: Text(fmtMoney(p.stockValue), style: const TextStyle(fontWeight: FontWeight.w700)),
      onTap: () => Navigator.push(
        context,
        MaterialPageRoute(builder: (_) => ProductDetailScreen(productId: p.id)),
      ),
    );
  }
}
