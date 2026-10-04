import 'package:flutter/material.dart';

import '../../data/models.dart';
import '../../data/repo.dart';
import '../../i18n.dart';
import '../format.dart';
import '../theme.dart';
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
          return Scaffold(
            appBar: AppBar(),
            body: EmptyState(
              icon: Icons.delete_outline,
              text: t('Produit supprimé', 'تم حذف المنتج'),
            ),
          );
        }
        final unit = p.unit;
        final notSet = t('non renseigné', 'غير محدد');
        return Scaffold(
          appBar: AppBar(
            title: Text(p.name),
            actions: [
              IconButton(
                tooltip: t('Modifier', 'تعديل'),
                icon: const Icon(Icons.edit_outlined),
                onPressed: () => Navigator.push(
                  context,
                  MaterialPageRoute(builder: (_) => ProductFormScreen(product: p)),
                ),
              ),
              IconButton(
                tooltip: t('Supprimer', 'حذف'),
                icon: const Icon(Icons.delete_outline),
                onPressed: () async {
                  if (await confirm(
                    context,
                    t('Supprimer « ${p.name} » ?', 'حذف « ${p.name} » ؟'),
                  )) {
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
                  padding: const EdgeInsets.all(18),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        t('En stock', 'في المخزون'),
                        style: Theme.of(context).textTheme.bodySmall,
                      ),
                      Text(
                        fmtQty(p.qty, unit),
                        style: Theme.of(context).textTheme.headlineMedium
                            ?.copyWith(fontWeight: FontWeight.w800, color: brandColor),
                      ),
                      const Divider(height: 24),
                      _kv(
                        context,
                        t("Prix d'achat", 'سعر الشراء'),
                        p.purchasePrice == null ? notSet : '${fmtMoney(p.purchasePrice!)} / $unit',
                      ),
                      _kv(
                        context,
                        t('Prix de vente', 'سعر البيع'),
                        p.salePrice == null ? notSet : '${fmtMoney(p.salePrice!)} / $unit',
                      ),
                      _kv(
                        context,
                        t("Valeur (prix d'achat)", 'القيمة (سعر الشراء)'),
                        fmtMoney(p.stockValue),
                      ),
                      _kv(
                        context,
                        t('Valeur (prix de vente)', 'القيمة (سعر البيع)'),
                        fmtMoney(p.saleValue),
                      ),
                      if (p.category != null) _kv(context, t('Catégorie', 'الفئة'), p.category!),
                      if (p.minStock != null)
                        _kv(context, t("Seuil d'alerte", 'حد التنبيه'), fmtQty(p.minStock!, unit)),
                      if (p.note != null) _kv(context, t('Note', 'ملاحظة'), p.note!),
                    ],
                  ),
                ),
              ),
              Padding(
                padding: const EdgeInsets.fromLTRB(16, 8, 16, 0),
                child: FilledButton.icon(
                  style: FilledButton.styleFrom(minimumSize: const Size.fromHeight(50)),
                  icon: const Icon(Icons.fact_check_outlined),
                  label: Text(t('Compter (inventaire)', 'العد (الجرد)')),
                  onPressed: () async {
                    final r = await askAmount(
                      context,
                      title: t('Quantité comptée', 'الكمية المعدودة'),
                      label: t('Quantité réelle', 'الكمية الحقيقية'),
                      suffix: unit,
                      helper: t('Calcul possible : 3x50+20', 'يمكن الحساب: 3x50+20'),
                      initial: p.qty,
                      allowZero: true,
                    );
                    if (r != null) await repo.setStockQty(p.id, r.value, note: r.note);
                  },
                ),
              ),
              Padding(
                padding: const EdgeInsets.fromLTRB(16, 8, 16, 0),
                child: Row(
                  children: [
                    Expanded(
                      child: _action(context, Icons.add, t('Entrée', 'دخول'), () async {
                        final r = await askAmount(
                          context,
                          title: t('Entrée de stock (achat)', 'دخول مخزون (شراء)'),
                          label: t('Quantité ajoutée', 'الكمية المضافة'),
                          suffix: unit,
                        );
                        if (r != null) {
                          await repo.addStockMovement(
                            p.id,
                            r.value,
                            'entree',
                            unitCost: p.purchasePrice,
                            note: r.note,
                          );
                        }
                      }),
                    ),
                    const SizedBox(width: 8),
                    Expanded(
                      child: _action(context, Icons.remove, t('Sortie', 'خروج'), () async {
                        final r = await askAmount(
                          context,
                          title: t('Sortie de stock (vente)', 'خروج مخزون (بيع)'),
                          label: t('Quantité retirée', 'الكمية المسحوبة'),
                          suffix: unit,
                        );
                        if (r != null) {
                          await repo.addStockMovement(p.id, -r.value, 'sortie', note: r.note);
                        }
                      }),
                    ),
                    const SizedBox(width: 8),
                    Expanded(
                      child: _action(
                        context,
                        Icons.broken_image_outlined,
                        t('Perte', 'تلف'),
                        () async {
                          final r = await askAmount(
                            context,
                            title: t('Perte / casse / périmé', 'تلف / كسر / منتهي الصلاحية'),
                            label: t('Quantité perdue', 'الكمية التالفة'),
                            suffix: unit,
                          );
                          if (r != null) {
                            await repo.addStockMovement(p.id, -r.value, 'perte', note: r.note);
                          }
                        },
                      ),
                    ),
                  ],
                ),
              ),
              SectionTitle(historyHint),
              if (moves.isEmpty)
                ListCard(children: [ListTile(title: Text(t('Aucun mouvement', 'لا توجد حركات')))])
              else
                ListCard(
                  children: [
                    for (final m in moves)
                      MovementTile(
                        title: stockKindLabels[m.kind] ?? m.kind,
                        date: m.date,
                        note: m.note,
                        positive: m.amount >= 0,
                        amount: '${m.amount > 0 ? '+' : ''}${fmtQty(m.amount, unit)}',
                        onDelete: () async {
                          if (await confirm(
                            context,
                            t('Annuler ce mouvement ?', 'إلغاء هذه الحركة ؟'),
                          )) {
                            await repo.deleteStockMovement(m.id);
                          }
                        },
                      ),
                  ],
                ),
            ],
          ),
        );
      },
    );
  }

  Widget _action(BuildContext context, IconData icon, String label, VoidCallback onTap) =>
      FilledButton.tonal(
        onPressed: onTap,
        style: FilledButton.styleFrom(
          padding: const EdgeInsets.symmetric(vertical: 10),
          minimumSize: const Size(0, 56),
        ),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(icon, size: 20),
            Text(label, maxLines: 1, overflow: TextOverflow.ellipsis),
          ],
        ),
      );

  Widget _kv(BuildContext context, String k, String v) => Padding(
    padding: const EdgeInsets.symmetric(vertical: 3),
    child: Row(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Expanded(
          child: Text(k, style: TextStyle(color: Theme.of(context).colorScheme.onSurfaceVariant)),
        ),
        Text(v, style: const TextStyle(fontWeight: FontWeight.w600)),
      ],
    ),
  );
}
