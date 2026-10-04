import 'package:flutter/material.dart';

import '../../data/models.dart';
import '../../data/repo.dart';
import '../format.dart';
import '../widgets/common.dart';
import 'product_form_screen.dart';

class ProductDetailScreen extends StatelessWidget {
  const ProductDetailScreen({super.key, required this.productId});

  final String productId;

  @override
  Widget build(BuildContext context) {
    final repo = Repo.instance;
    return Reactive<(Product?, List<Movement>)>(
      load: () async => (await repo.product(productId), await repo.stockMovements(productId)),
      builder: (context, data) {
        final (p, moves) = data;
        if (p == null) {
          return const Scaffold(body: EmptyState(icon: Icons.delete_outline, text: 'Produit supprimé'));
        }
        final unit = p.unitSymbol;
        return Scaffold(
          appBar: AppBar(
            title: Text(p.name),
            actions: [
              IconButton(
                tooltip: 'Modifier',
                icon: const Icon(Icons.edit_outlined),
                onPressed: () => Navigator.push(context,
                    MaterialPageRoute(builder: (_) => ProductFormScreen(product: p))),
              ),
              IconButton(
                tooltip: 'Supprimer',
                icon: const Icon(Icons.delete_outline),
                onPressed: () async {
                  if (await confirm(context, 'Supprimer « ${p.name} » ?')) {
                    await repo.deleteProduct(p.id);
                    if (context.mounted) Navigator.pop(context);
                  }
                },
              ),
            ],
          ),
          body: ListView(
            padding: const EdgeInsets.only(bottom: 24),
            children: [
              Card(
                child: Padding(
                  padding: const EdgeInsets.all(16),
                  child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                    Text(fmtQty(p.qty, unit),
                        style: Theme.of(context).textTheme.headlineMedium
                            ?.copyWith(fontWeight: FontWeight.bold)),
                    const SizedBox(height: 8),
                    _kv("Prix d'achat", p.purchasePrice == null ? 'non renseigné' : '${fmtMoney(p.purchasePrice!)} / $unit'),
                    _kv('Prix de vente', p.salePrice == null ? 'non renseigné' : '${fmtMoney(p.salePrice!)} / $unit'),
                    _kv("Valeur (prix d'achat)", fmtMoney(p.stockValue)),
                    _kv('Valeur (prix de vente)', fmtMoney(p.saleValue)),
                    if (p.category != null) _kv('Catégorie', p.category!),
                    if (p.minStock != null) _kv("Seuil d'alerte", fmtQty(p.minStock!, unit)),
                    if (p.note != null) _kv('Note', p.note!),
                  ]),
                ),
              ),
              Padding(
                padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 4),
                child: Wrap(spacing: 8, runSpacing: 8, children: [
                  FilledButton.tonalIcon(
                    icon: const Icon(Icons.fact_check_outlined),
                    label: const Text('Compter (inventaire)'),
                    onPressed: () async {
                      final r = await askAmount(context,
                          title: 'Quantité comptée', label: 'Quantité réelle', suffix: unit,
                          helper: 'Calcul possible : 3x50+20', initial: p.qty, allowZero: true);
                      if (r != null) await repo.setStockQty(p.id, r.value, note: r.note);
                    },
                  ),
                  FilledButton.tonalIcon(
                    icon: const Icon(Icons.add),
                    label: const Text('Entrée'),
                    onPressed: () async {
                      final r = await askAmount(context,
                          title: 'Entrée de stock (achat)', label: 'Quantité ajoutée', suffix: unit);
                      if (r != null) {
                        await repo.addStockMovement(p.id, r.value, 'entree',
                            unitCost: p.purchasePrice, note: r.note);
                      }
                    },
                  ),
                  FilledButton.tonalIcon(
                    icon: const Icon(Icons.remove),
                    label: const Text('Sortie'),
                    onPressed: () async {
                      final r = await askAmount(context,
                          title: 'Sortie de stock (vente)', label: 'Quantité retirée', suffix: unit);
                      if (r != null) {
                        await repo.addStockMovement(p.id, -r.value, 'sortie', note: r.note);
                      }
                    },
                  ),
                  FilledButton.tonalIcon(
                    icon: const Icon(Icons.broken_image_outlined),
                    label: const Text('Perte'),
                    onPressed: () async {
                      final r = await askAmount(context,
                          title: 'Perte / casse / périmé', label: 'Quantité perdue', suffix: unit);
                      if (r != null) {
                        await repo.addStockMovement(p.id, -r.value, 'perte', note: r.note);
                      }
                    },
                  ),
                ]),
              ),
              const Padding(
                padding: EdgeInsets.fromLTRB(16, 16, 16, 4),
                child: Text('Historique (appui long pour annuler une ligne)',
                    style: TextStyle(fontWeight: FontWeight.w600)),
              ),
              if (moves.isEmpty) const ListTile(title: Text('Aucun mouvement')),
              for (final m in moves)
                MovementTile(
                  title: stockKindLabels[m.kind] ?? m.kind,
                  date: m.date,
                  note: m.note,
                  amount: '${m.amount > 0 ? '+' : ''}${fmtQty(m.amount, unit)}',
                  onDelete: () async {
                    if (await confirm(context, 'Annuler ce mouvement ?')) {
                      await repo.deleteStockMovement(m.id);
                    }
                  },
                ),
            ],
          ),
        );
      },
    );
  }

  Widget _kv(String k, String v) => Padding(
        padding: const EdgeInsets.symmetric(vertical: 2),
        child: Row(crossAxisAlignment: CrossAxisAlignment.start, children: [
          SizedBox(width: 150, child: Text(k, style: const TextStyle(color: Colors.black54))),
          Expanded(child: Text(v)),
        ]),
      );
}
