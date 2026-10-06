import 'package:flutter/material.dart';

import '../../data/models.dart';
import '../../data/repo.dart';
import '../../i18n.dart';
import '../format.dart';
import '../theme.dart';
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
          return Scaffold(
            appBar: AppBar(),
            body: EmptyState(icon: Icons.delete_outline, text: t('Supprimé', 'تم الحذف')),
          );
        }
        final b = p.balance;
        final status = b > 0.0001
            ? t('Il me doit', 'عليه لي')
            : b < -0.0001
            ? t('Je lui dois', 'علي له')
            : t('Compte soldé', 'الحساب مسدد');

        // Après la saisie, retour à la liste des dettes (onglet « Tous »), positionnée sur cette personne.
        Future<void> add(DebtKind kind) async {
          final r = await askAmount(
            context,
            title: kind.label,
            label: t('Montant', 'المبلغ'),
            suffix: currency,
          );
          if (r == null) return;
          await repo.addDebtEntry(p.id, kind.code, r.value, note: r.note);
          if (!context.mounted) return;
          toast(context, '${kind.label} : ${fmtMoney(r.value)} · ${p.name}');
          Navigator.pop(context, p.id);
        }

        return Scaffold(
          appBar: AppBar(
            title: Text(p.name),
            actions: [
              IconButton(
                tooltip: t('Modifier', 'تعديل'),
                icon: const Icon(Icons.edit_outlined),
                onPressed: () => Navigator.push(
                  context,
                  MaterialPageRoute(builder: (_) => PartyFormScreen(party: p)),
                ),
              ),
              IconButton(
                tooltip: t('Supprimer', 'حذف'),
                icon: const Icon(Icons.delete_outline),
                onPressed: () async {
                  final msg = b.abs() > 0.0001
                      ? t(
                          'Supprimer « ${p.name} » ? Son solde (${fmtMoney(b.abs())}) ne sera plus compté.',
                          'حذف « ${p.name} » ؟ لن يُحسب رصيده (${fmtMoney(b.abs())}).',
                        )
                      : t('Supprimer « ${p.name} » ?', 'حذف « ${p.name} » ؟');
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
                  padding: const EdgeInsets.all(18),
                  child: Row(
                    children: [
                      InitialAvatar(p.name, color: moneyColor(context, b)),
                      const SizedBox(width: 14),
                      Expanded(
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Text(status, style: Theme.of(context).textTheme.bodySmall),
                            FittedBox(
                              child: Text(
                                fmtMoney(b.abs()),
                                style: Theme.of(context).textTheme.headlineSmall?.copyWith(
                                  fontWeight: FontWeight.w800,
                                  color: moneyColor(context, b),
                                ),
                              ),
                            ),
                            Text(
                              [
                                partyKinds[p.kind] ?? p.kind,
                                if (p.phone != null) fmtPhone(p.phone!),
                                if (p.note != null) p.note!,
                              ].join(' · '),
                            ),
                          ],
                        ),
                      ),
                    ],
                  ),
                ),
              ),
              Padding(
                padding: const EdgeInsets.fromLTRB(16, 8, 16, 0),
                child: Column(
                  children: [
                    Row(
                      children: [
                        Expanded(
                          child: _btn(
                            Icons.add,
                            DebtKind.credit.label,
                            positiveColor,
                            () => add(DebtKind.credit),
                          ),
                        ),
                        const SizedBox(width: 8),
                        Expanded(
                          child: _btn(
                            Icons.payments_outlined,
                            DebtKind.recu.label,
                            positiveColor,
                            () => add(DebtKind.recu),
                          ),
                        ),
                      ],
                    ),
                    const SizedBox(height: 8),
                    Row(
                      children: [
                        Expanded(
                          child: _btn(
                            Icons.add,
                            DebtKind.dette.label,
                            negativeColor,
                            () => add(DebtKind.dette),
                          ),
                        ),
                        const SizedBox(width: 8),
                        Expanded(
                          child: _btn(
                            Icons.payments_outlined,
                            DebtKind.paye.label,
                            negativeColor,
                            () => add(DebtKind.paye),
                          ),
                        ),
                      ],
                    ),
                  ],
                ),
              ),
              SectionTitle(historyHint),
              if (entries.isEmpty)
                ListCard(children: [ListTile(title: Text(t('Aucune opération', 'لا توجد عمليات')))])
              else
                ListCard(
                  children: [
                    for (final e in entries)
                      MovementTile(
                        title: DebtKind.of(e.kind).label,
                        date: e.date,
                        note: e.note == initialBalanceNote
                            ? t('Solde de départ', 'الرصيد الأولي')
                            : e.note,
                        positive: e.amount >= 0,
                        amount: fmtMoney(e.amount.abs()),
                        onDelete: () async {
                          if (await confirm(
                            context,
                            t('Annuler cette opération ?', 'إلغاء هذه العملية ؟'),
                          )) {
                            await repo.deleteDebtEntry(e.id);
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

  Widget _btn(IconData icon, String label, Color color, VoidCallback onTap) =>
      FilledButton.tonalIcon(
        onPressed: onTap,
        style: FilledButton.styleFrom(
          backgroundColor: color.withValues(alpha: 0.1),
          foregroundColor: color,
          minimumSize: const Size(0, 50),
        ),
        icon: Icon(icon),
        label: Text(label, overflow: TextOverflow.ellipsis),
      );
}
