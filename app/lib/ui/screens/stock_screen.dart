import 'package:flutter/material.dart';

import '../../data/models.dart';
import '../../data/repo.dart';
import '../format.dart';
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
        title: const Text('Stock'),
        bottom: PreferredSize(
          preferredSize: const Size.fromHeight(60),
          child: Padding(
            padding: const EdgeInsets.fromLTRB(12, 0, 12, 8),
            child: TextField(
              controller: _search,
              onChanged: (v) => setState(() => _query = v),
              decoration: InputDecoration(
                hintText: 'Rechercher un produit ou une catégorie',
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
        onPressed: () => Navigator.push(
            context, MaterialPageRoute(builder: (_) => const ProductFormScreen())),
        icon: const Icon(Icons.add),
        label: const Text('Produit'),
      ),
      body: Reactive<List<Product>>(
        load: () => Repo.instance.products(search: _query),
        builder: (context, products) {
          if (products.isEmpty) {
            return EmptyState(
              icon: Icons.inventory_2_outlined,
              text: _query.isEmpty
                  ? 'Aucun produit.\nAppuyez sur « + Produit » pour saisir votre stock.'
                  : 'Aucun résultat.',
            );
          }
          final total = products.fold<double>(0, (a, p) => a + p.stockValue);
          return Column(children: [
            Container(
              width: double.infinity,
              padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
              color: Theme.of(context).colorScheme.surfaceContainerHighest,
              child: Text('${products.length} produits · valeur ${fmtMoney(total)}',
                  style: const TextStyle(fontWeight: FontWeight.w600)),
            ),
            Expanded(
              child: ListView.separated(
                padding: const EdgeInsets.only(bottom: 88),
                itemCount: products.length,
                separatorBuilder: (_, _) => const Divider(height: 1),
                itemBuilder: (context, i) => _ProductTile(product: products[i]),
              ),
            ),
          ]);
        },
      ),
    );
  }
}

class _ProductTile extends StatelessWidget {
  const _ProductTile({required this.product});

  final Product product;

  @override
  Widget build(BuildContext context) {
    final p = product;
    final price = p.purchasePrice == null ? 'prix ?' : fmtMoney(p.purchasePrice!);
    return ListTile(
      title: Text(p.name),
      subtitle: Text(
        '${fmtQty(p.qty, p.unitSymbol)} × $price'
        '${p.category == null ? '' : ' · ${p.category}'}',
      ),
      leading: p.missingPurchasePrice
          ? Icon(Icons.warning_amber_rounded, color: Theme.of(context).colorScheme.error)
          : p.lowStock
              ? const Icon(Icons.trending_down, color: Colors.orange)
              : null,
      trailing: Text(fmtMoney(p.stockValue), style: const TextStyle(fontWeight: FontWeight.w600)),
      onTap: () => Navigator.push(
          context, MaterialPageRoute(builder: (_) => ProductDetailScreen(productId: p.id))),
    );
  }
}
