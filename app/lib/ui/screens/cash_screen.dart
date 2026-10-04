import 'package:flutter/material.dart';

import '../../data/models.dart';
import '../../data/repo.dart';
import '../format.dart';
import '../widgets/common.dart';

/// Argent disponible : caisse en espèces et wallets (Bankily, Masrvi, Sedad…).
class CashScreen extends StatelessWidget {
  const CashScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Caisse et wallets')),
      floatingActionButton: FloatingActionButton.extended(
        onPressed: () => showAccountDialog(context),
        icon: const Icon(Icons.add),
        label: const Text('Compte'),
      ),
      body: Reactive<List<Account>>(
        load: Repo.instance.accounts,
        builder: (context, accounts) {
          final total = accounts.fold<double>(0, (a, x) => a + x.balance);
          return ListView(
            padding: const EdgeInsets.only(bottom: 88),
            children: [
              Card(
                color: Theme.of(context).colorScheme.primaryContainer,
                child: ListTile(
                  title: const Text('Total disponible'),
                  trailing: Text(fmtMoney(total),
                      style: const TextStyle(fontWeight: FontWeight.bold, fontSize: 18)),
                ),
              ),
              const Padding(
                padding: EdgeInsets.fromLTRB(16, 8, 16, 4),
                child: Text('Touchez un compte pour saisir son solde actuel.'),
              ),
              for (final a in accounts)
                Card(
                  child: ListTile(
                    leading: Icon(a.kind == 'cash'
                        ? Icons.payments_outlined
                        : a.kind == 'banque'
                            ? Icons.account_balance_outlined
                            : Icons.phone_android),
                    title: Text(a.name),
                    subtitle: Text(accountKinds[a.kind] ?? a.kind),
                    trailing: Text(fmtMoney(a.balance),
                        style: TextStyle(
                            fontWeight: FontWeight.bold, color: moneyColor(context, a.balance))),
                    onTap: () => Navigator.push(context,
                        MaterialPageRoute(builder: (_) => AccountDetailScreen(accountId: a.id))),
                  ),
                ),
            ],
          );
        },
      ),
    );
  }
}

Future<void> showAccountDialog(BuildContext context, {Account? account}) async {
  final name = TextEditingController(text: account?.name ?? '');
  var kind = account?.kind ?? 'wallet';
  final ok = await showDialog<bool>(
    context: context,
    builder: (ctx) => StatefulBuilder(
      builder: (ctx, setState) => AlertDialog(
        title: Text(account == null ? 'Nouveau compte' : 'Modifier le compte'),
        content: Column(mainAxisSize: MainAxisSize.min, children: [
          TextField(
            controller: name,
            autofocus: true,
            decoration: const InputDecoration(labelText: 'Nom', hintText: 'Ex. Click, BimBank'),
          ),
          const SizedBox(height: 12),
          DropdownButtonFormField<String>(
            initialValue: kind,
            decoration: const InputDecoration(labelText: 'Type'),
            items: [
              for (final e in accountKinds.entries) DropdownMenuItem(value: e.key, child: Text(e.value)),
            ],
            onChanged: (v) => setState(() => kind = v ?? kind),
          ),
        ]),
        actions: [
          TextButton(onPressed: () => Navigator.pop(ctx, false), child: const Text('Annuler')),
          FilledButton(onPressed: () => Navigator.pop(ctx, true), child: const Text('Enregistrer')),
        ],
      ),
    ),
  );
  if (ok == true && name.text.trim().isNotEmpty) {
    await Repo.instance.saveAccount(id: account?.id, name: name.text, kind: kind);
  }
}

class AccountDetailScreen extends StatelessWidget {
  const AccountDetailScreen({super.key, required this.accountId});

  final String accountId;

  @override
  Widget build(BuildContext context) {
    final repo = Repo.instance;
    return Reactive<(Account?, List<Movement>)>(
      load: () async => (await repo.account(accountId), await repo.accountMovements(accountId)),
      builder: (context, data) {
        final (a, moves) = data;
        if (a == null) {
          return const Scaffold(body: EmptyState(icon: Icons.delete_outline, text: 'Supprimé'));
        }
        return Scaffold(
          appBar: AppBar(
            title: Text(a.name),
            actions: [
              IconButton(
                tooltip: 'Modifier',
                icon: const Icon(Icons.edit_outlined),
                onPressed: () => showAccountDialog(context, account: a),
              ),
              IconButton(
                tooltip: 'Supprimer',
                icon: const Icon(Icons.delete_outline),
                onPressed: () async {
                  if (await confirm(context, 'Supprimer le compte « ${a.name} » ?')) {
                    await repo.deleteAccount(a.id);
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
                child: ListTile(
                  title: const Text('Solde actuel'),
                  trailing: Text(fmtMoney(a.balance),
                      style: Theme.of(context).textTheme.titleLarge?.copyWith(fontWeight: FontWeight.bold)),
                ),
              ),
              Padding(
                padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 4),
                child: Wrap(spacing: 8, runSpacing: 8, children: [
                  FilledButton.icon(
                    icon: const Icon(Icons.edit_note),
                    label: const Text('Saisir le solde réel'),
                    onPressed: () async {
                      final r = await askAmount(context,
                          title: 'Solde réel de ${a.name}',
                          label: 'Montant compté / affiché',
                          suffix: 'MRU',
                          initial: a.balance,
                          allowZero: true);
                      if (r != null) await repo.setAccountBalance(a.id, r.value, note: r.note);
                    },
                  ),
                  FilledButton.tonalIcon(
                    icon: const Icon(Icons.add),
                    label: const Text('Entrée'),
                    onPressed: () async {
                      final r = await askAmount(context, title: 'Entrée d\'argent', label: 'Montant', suffix: 'MRU');
                      if (r != null) await repo.addAccountMovement(a.id, 'entree', r.value, note: r.note);
                    },
                  ),
                  FilledButton.tonalIcon(
                    icon: const Icon(Icons.remove),
                    label: const Text('Sortie'),
                    onPressed: () async {
                      final r = await askAmount(context, title: 'Sortie d\'argent', label: 'Montant', suffix: 'MRU');
                      if (r != null) await repo.addAccountMovement(a.id, 'sortie', -r.value, note: r.note);
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
                  title: accountMovementLabels[m.kind] ?? m.kind,
                  date: m.date,
                  note: m.note,
                  amount: '${m.amount > 0 ? '+' : ''}${fmtMoney(m.amount)}',
                  onDelete: () async {
                    if (await confirm(context, 'Annuler ce mouvement ?')) {
                      await repo.deleteAccountMovement(m.id);
                    }
                  },
                ),
            ],
          ),
        );
      },
    );
  }
}
