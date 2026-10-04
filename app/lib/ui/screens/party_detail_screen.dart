import 'package:flutter/material.dart';

import '../../data/models.dart';
import '../../data/repo.dart';
import '../format.dart';
import '../widgets/common.dart';
import 'party_form_screen.dart';

class PartyDetailScreen extends StatelessWidget {
  const PartyDetailScreen({super.key, required this.partyId});

  final String partyId;

  @override
  Widget build(BuildContext context) {
    final repo = Repo.instance;
    return Reactive<(Party?, List<Movement>)>(
      load: () async => (await repo.party(partyId), await repo.debtEntries(partyId)),
      builder: (context, data) {
        final (p, entries) = data;
        if (p == null) {
          return const Scaffold(body: EmptyState(icon: Icons.delete_outline, text: 'Supprimé'));
        }
        final b = p.balance;
        final status = b > 0.0001
            ? 'Il me doit'
            : b < -0.0001
                ? 'Je lui dois'
                : 'Compte soldé';

        Future<void> add(DebtKind kind) async {
          final r = await askAmount(context, title: kind.label, label: 'Montant', suffix: 'MRU');
          if (r != null) await repo.addDebtEntry(p.id, kind.code, r.value, note: r.note);
        }

        return Scaffold(
          appBar: AppBar(
            title: Text(p.name),
            actions: [
              IconButton(
                tooltip: 'Modifier',
                icon: const Icon(Icons.edit_outlined),
                onPressed: () => Navigator.push(
                    context, MaterialPageRoute(builder: (_) => PartyFormScreen(party: p))),
              ),
              IconButton(
                tooltip: 'Supprimer',
                icon: const Icon(Icons.delete_outline),
                onPressed: () async {
                  final msg = b.abs() > 0.0001
                      ? 'Supprimer « ${p.name} » ? Son solde (${fmtMoney(b.abs())}) ne sera plus compté.'
                      : 'Supprimer « ${p.name} » ?';
                  if (await confirm(context, msg)) {
                    await repo.deleteParty(p.id);
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
                    Text(status),
                    Text(fmtMoney(b.abs()),
                        style: Theme.of(context).textTheme.headlineMedium?.copyWith(
                            fontWeight: FontWeight.bold, color: moneyColor(context, b))),
                    const SizedBox(height: 4),
                    Text([partyKinds[p.kind] ?? p.kind, if (p.phone != null) p.phone!, if (p.note != null) p.note!]
                        .join(' · ')),
                  ]),
                ),
              ),
              Padding(
                padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 4),
                child: Column(children: [
                  Row(children: [
                    Expanded(child: _btn(Icons.add, DebtKind.credit.label, () => add(DebtKind.credit))),
                    const SizedBox(width: 8),
                    Expanded(child: _btn(Icons.payments_outlined, DebtKind.recu.label, () => add(DebtKind.recu))),
                  ]),
                  const SizedBox(height: 8),
                  Row(children: [
                    Expanded(child: _btn(Icons.add, DebtKind.dette.label, () => add(DebtKind.dette))),
                    const SizedBox(width: 8),
                    Expanded(child: _btn(Icons.payments_outlined, DebtKind.paye.label, () => add(DebtKind.paye))),
                  ]),
                ]),
              ),
              const Padding(
                padding: EdgeInsets.fromLTRB(16, 16, 16, 4),
                child: Text('Historique (appui long pour annuler une ligne)',
                    style: TextStyle(fontWeight: FontWeight.w600)),
              ),
              if (entries.isEmpty) const ListTile(title: Text('Aucune opération')),
              for (final e in entries)
                MovementTile(
                  title: DebtKind.of(e.kind).label,
                  date: e.date,
                  note: e.note,
                  amount: fmtMoney(e.amount.abs()),
                  onDelete: () async {
                    if (await confirm(context, 'Annuler cette opération ?')) {
                      await repo.deleteDebtEntry(e.id);
                    }
                  },
                ),
            ],
          ),
        );
      },
    );
  }

  Widget _btn(IconData icon, String label, VoidCallback onTap) => FilledButton.tonalIcon(
        onPressed: onTap,
        icon: Icon(icon),
        label: Text(label, overflow: TextOverflow.ellipsis),
      );
}
