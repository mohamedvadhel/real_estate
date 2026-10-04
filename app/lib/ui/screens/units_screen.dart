import 'package:flutter/material.dart';

import '../../data/models.dart';
import '../../data/repo.dart';
import '../../i18n.dart';
import '../widgets/common.dart';

/// Boîte de dialogue de création d'unité. Renvoie l'id créé.
Future<String?> showAddUnitDialog(BuildContext context) async {
  final name = TextEditingController();
  final symbol = TextEditingController();
  var decimals = true;
  final ok = await showDialog<bool>(
    context: context,
    builder: (ctx) => StatefulBuilder(
      builder: (ctx, setState) => AlertDialog(
        title: Text(t('Nouvelle unité', 'وحدة جديدة')),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            TextField(
              controller: name,
              autofocus: true,
              decoration: InputDecoration(
                labelText: t('Nom', 'الاسم'),
                hintText: t('Ex. Botte, Rouleau, Fût', 'مثال: حزمة، لفة، برميل'),
              ),
            ),
            const SizedBox(height: 12),
            TextField(
              controller: symbol,
              decoration: InputDecoration(
                labelText: t('Abréviation (facultatif)', 'الاختصار (اختياري)'),
                hintText: t('Ex. bt', 'مثال: حزمة'),
              ),
            ),
            SwitchListTile(
              contentPadding: EdgeInsets.zero,
              title: Text(t('Accepte les décimales (ex. 2,5)', 'تقبل الكسور (مثال 2,5)')),
              value: decimals,
              onChanged: (v) => setState(() => decimals = v),
            ),
          ],
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(ctx, false),
            child: Text(t('Annuler', 'إلغاء')),
          ),
          FilledButton(onPressed: () => Navigator.pop(ctx, true), child: Text(t('Créer', 'إنشاء'))),
        ],
      ),
    ),
  );
  if (ok != true || name.text.trim().isEmpty) return null;
  return Repo.instance.addUnit(name.text, symbol.text, allowDecimal: decimals);
}

class UnitsScreen extends StatelessWidget {
  const UnitsScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text(t('Unités de mesure', 'وحدات القياس'))),
      floatingActionButton: FloatingActionButton(
        heroTag: null,
        onPressed: () => showAddUnitDialog(context),
        child: const Icon(Icons.add),
      ),
      body: Reactive<List<Unit>>(
        load: Repo.instance.units,
        builder: (context, units) => ListView(
          padding: const EdgeInsets.only(top: 8, bottom: 96),
          children: [
            ListCard(
              children: [
                for (final u in units)
                  ListTile(
                    title: Text(u.label),
                    subtitle: Text(
                      u.allowDecimal
                          ? t('Décimales autorisées', 'الكسور مسموحة')
                          : t('Nombres entiers', 'أعداد صحيحة'),
                    ),
                    trailing: IconButton(
                      icon: const Icon(Icons.delete_outline),
                      onPressed: () async {
                        final used = await Repo.instance.unitUsage(u.id);
                        if (!context.mounted) return;
                        if (used > 0) {
                          toast(
                            context,
                            t(
                              'Unité utilisée par $used produit(s) : impossible de la supprimer',
                              'الوحدة مستعملة في $used منتج: لا يمكن حذفها',
                            ),
                          );
                          return;
                        }
                        if (await confirm(
                          context,
                          t(
                            "Supprimer l'unité « ${u.displayName} » ?",
                            'حذف الوحدة « ${u.displayName} » ؟',
                          ),
                        )) {
                          await Repo.instance.deleteUnit(u.id);
                        }
                      },
                    ),
                  ),
              ],
            ),
          ],
        ),
      ),
    );
  }
}
