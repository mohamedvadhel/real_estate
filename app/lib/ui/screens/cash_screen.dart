import 'package:flutter/material.dart';

import '../../data/models.dart';
import '../../data/repo.dart';
import '../../i18n.dart';
import '../format.dart';
import '../theme.dart';
import '../widgets/common.dart';

IconData _accountIcon(String kind) => switch (kind) {
  'cash' => Icons.payments_outlined,
  'banque' => Icons.account_balance_outlined,
  _ => Icons.phone_android,
};

/// Argent disponible : caisse en espèces et wallets (Bankily, Masrvi, Sedad…).
class CashScreen extends StatelessWidget {
  const CashScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text(t('Caisse et wallets', 'الصندوق والمحافظ'))),
      floatingActionButton: FloatingActionButton.extended(
        heroTag: null,
        onPressed: () => showAccountDialog(context),
        icon: const Icon(Icons.add),
        label: Text(t('Compte', 'حساب')),
      ),
      body: Reactive<List<Account>>(
        load: Repo.instance.accounts,
        builder: (context, accounts) {
          final total = accounts.fold<double>(0, (a, x) => a + x.balance);
          return ListView(
            padding: const EdgeInsets.only(bottom: 96),
            children: [
              TotalBanner(label: t('Total disponible', 'المجموع المتوفر'), value: fmtMoney(total)),
              SectionTitle(
                t(
                  'Touchez un compte pour saisir son solde actuel',
                  'اضغط على حساب لإدخال رصيده الحالي',
                ),
              ),
              ListCard(
                children: [
                  for (final a in accounts)
                    ListTile(
                      leading: Container(
                        width: 44,
                        height: 44,
                        decoration: BoxDecoration(
                          color: brandColor.withValues(alpha: 0.1),
                          borderRadius: BorderRadius.circular(12),
                        ),
                        child: Icon(_accountIcon(a.kind), color: brandColor),
                      ),
                      title: Text(
                        a.displayName,
                        style: const TextStyle(fontWeight: FontWeight.w600),
                      ),
                      subtitle: Text(accountKinds[a.kind] ?? a.kind),
                      trailing: Text(
                        fmtMoney(a.balance),
                        style: TextStyle(
                          fontWeight: FontWeight.w700,
                          color: moneyColor(context, a.balance),
                        ),
                      ),
                      onTap: () => Navigator.push(
                        context,
                        MaterialPageRoute(builder: (_) => AccountDetailScreen(accountId: a.id)),
                      ),
                    ),
                ],
              ),
            ],
          );
        },
      ),
    );
  }
}

Future<void> showAccountDialog(BuildContext context, {Account? account}) async {
  final name = TextEditingController(text: account?.displayName ?? '');
  var kind = account?.kind ?? 'wallet';
  final ok = await showDialog<bool>(
    context: context,
    builder: (ctx) => StatefulBuilder(
      builder: (ctx, setState) => AlertDialog(
        title: Text(
          account == null
              ? t('Nouveau compte', 'حساب جديد')
              : t('Modifier le compte', 'تعديل الحساب'),
        ),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            TextField(
              controller: name,
              autofocus: true,
              decoration: InputDecoration(
                labelText: t('Nom', 'الاسم'),
                hintText: t('Ex. Click, BimBank', 'مثال: كليك'),
              ),
            ),
            const SizedBox(height: 12),
            DropdownButtonFormField<String>(
              initialValue: kind,
              decoration: InputDecoration(labelText: t('Type', 'النوع')),
              items: [
                for (final e in accountKinds.entries)
                  DropdownMenuItem(value: e.key, child: Text(e.value)),
              ],
              onChanged: (v) => setState(() => kind = v ?? kind),
            ),
          ],
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(ctx, false),
            child: Text(t('Annuler', 'إلغاء')),
          ),
          FilledButton(
            onPressed: () => Navigator.pop(ctx, true),
            child: Text(t('Enregistrer', 'حفظ')),
          ),
        ],
      ),
    ),
  );
  if (ok != true || name.text.trim().isEmpty) return;
  // Nom inchangé d'un compte de départ : on garde le nom d'origine (traduit à l'affichage).
  final newName = account != null && name.text.trim() == account.displayName
      ? account.name
      : name.text;
  await Repo.instance.saveAccount(id: account?.id, name: newName, kind: kind);
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
          return Scaffold(
            appBar: AppBar(),
            body: EmptyState(icon: Icons.delete_outline, text: t('Supprimé', 'تم الحذف')),
          );
        }
        return Scaffold(
          appBar: AppBar(
            title: Text(a.displayName),
            actions: [
              IconButton(
                tooltip: t('Modifier', 'تعديل'),
                icon: const Icon(Icons.edit_outlined),
                onPressed: () => showAccountDialog(context, account: a),
              ),
              IconButton(
                tooltip: t('Supprimer', 'حذف'),
                icon: const Icon(Icons.delete_outline),
                onPressed: () async {
                  if (await confirm(
                    context,
                    t(
                      'Supprimer le compte « ${a.displayName} » ?',
                      'حذف الحساب « ${a.displayName} » ؟',
                    ),
                  )) {
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
                child: Padding(
                  padding: const EdgeInsets.all(18),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        t('Solde actuel', 'الرصيد الحالي'),
                        style: Theme.of(context).textTheme.bodySmall,
                      ),
                      FittedBox(
                        child: Text(
                          fmtMoney(a.balance),
                          style: Theme.of(context).textTheme.headlineMedium
                              ?.copyWith(fontWeight: FontWeight.w800, color: brandColor),
                        ),
                      ),
                    ],
                  ),
                ),
              ),
              Padding(
                padding: const EdgeInsets.fromLTRB(16, 8, 16, 0),
                child: FilledButton.icon(
                  style: FilledButton.styleFrom(minimumSize: const Size.fromHeight(50)),
                  icon: const Icon(Icons.edit_note),
                  label: Text(t('Saisir le solde réel', 'إدخال الرصيد الحقيقي')),
                  onPressed: () async {
                    final r = await askAmount(
                      context,
                      title: t(
                        'Solde réel de ${a.displayName}',
                        'الرصيد الحقيقي لـ ${a.displayName}',
                      ),
                      label: t('Montant compté / affiché', 'المبلغ المعدود / المعروض'),
                      suffix: currency,
                      initial: a.balance,
                      allowZero: true,
                    );
                    if (r != null) await repo.setAccountBalance(a.id, r.value, note: r.note);
                  },
                ),
              ),
              Padding(
                padding: const EdgeInsets.fromLTRB(16, 8, 16, 0),
                child: Row(
                  children: [
                    Expanded(
                      child: FilledButton.tonalIcon(
                        icon: const Icon(Icons.add),
                        label: Text(t('Entrée', 'دخول')),
                        onPressed: () async {
                          final r = await askAmount(
                            context,
                            title: t("Entrée d'argent", 'دخول مال'),
                            label: t('Montant', 'المبلغ'),
                            suffix: currency,
                          );
                          if (r != null) {
                            await repo.addAccountMovement(a.id, 'entree', r.value, note: r.note);
                          }
                        },
                      ),
                    ),
                    const SizedBox(width: 8),
                    Expanded(
                      child: FilledButton.tonalIcon(
                        icon: const Icon(Icons.remove),
                        label: Text(t('Sortie', 'خروج')),
                        onPressed: () async {
                          final r = await askAmount(
                            context,
                            title: t("Sortie d'argent", 'خروج مال'),
                            label: t('Montant', 'المبلغ'),
                            suffix: currency,
                          );
                          if (r != null) {
                            await repo.addAccountMovement(a.id, 'sortie', -r.value, note: r.note);
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
                        title: accountMovementLabels[m.kind] ?? m.kind,
                        date: m.date,
                        note: m.note,
                        positive: m.amount >= 0,
                        amount: '${m.amount > 0 ? '+' : ''}${fmtMoney(m.amount)}',
                        onDelete: () async {
                          if (await confirm(
                            context,
                            t('Annuler ce mouvement ?', 'إلغاء هذه الحركة ؟'),
                          )) {
                            await repo.deleteAccountMovement(m.id);
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
}
