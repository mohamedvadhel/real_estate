import 'package:flutter/material.dart';

import '../../data/models.dart';
import '../../data/repo.dart';
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
        title: const Text('Nouvelle unité'),
        content: Column(mainAxisSize: MainAxisSize.min, children: [
          TextField(
            controller: name,
            autofocus: true,
            decoration: const InputDecoration(labelText: 'Nom', hintText: 'Ex. Botte, Rouleau, Fût'),
          ),
          const SizedBox(height: 12),
          TextField(
            controller: symbol,
            decoration: const InputDecoration(labelText: 'Abréviation (facultatif)', hintText: 'Ex. bt'),
          ),
          SwitchListTile(
            contentPadding: EdgeInsets.zero,
            title: const Text('Accepte les décimales (ex. 2,5)'),
            value: decimals,
            onChanged: (v) => setState(() => decimals = v),
          ),
        ]),
        actions: [
          TextButton(onPressed: () => Navigator.pop(ctx, false), child: const Text('Annuler')),
          FilledButton(onPressed: () => Navigator.pop(ctx, true), child: const Text('Créer')),
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
      appBar: AppBar(title: const Text('Unités de mesure')),
      floatingActionButton: FloatingActionButton(
        onPressed: () => showAddUnitDialog(context),
        child: const Icon(Icons.add),
      ),
      body: Reactive<List<Unit>>(
        load: Repo.instance.units,
        builder: (context, units) => ListView(
          padding: const EdgeInsets.only(bottom: 88),
          children: [
            for (final u in units)
              ListTile(
                title: Text(u.label),
                subtitle: Text(u.allowDecimal ? 'Décimales autorisées' : 'Nombres entiers'),
                trailing: IconButton(
                  icon: const Icon(Icons.delete_outline),
                  onPressed: () async {
                    final used = await Repo.instance.unitUsage(u.id);
                    if (!context.mounted) return;
                    if (used > 0) {
                      toast(context, 'Unité utilisée par $used produit(s) : impossible de la supprimer');
                      return;
                    }
                    if (await confirm(context, 'Supprimer l\'unité « ${u.name} » ?')) {
                      await Repo.instance.deleteUnit(u.id);
                    }
                  },
                ),
              ),
          ],
        ),
      ),
    );
  }
}
