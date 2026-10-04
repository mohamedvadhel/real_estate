import 'package:flutter/material.dart';

import '../../data/models.dart';
import '../../data/repo.dart';
import '../../i18n.dart';
import '../format.dart';
import '../theme.dart';
import '../widgets/common.dart';
import 'party_detail_screen.dart';
import 'party_form_screen.dart';

/// Dettes des clients envers la boutique et dettes de la boutique.
/// S'ouvre toujours sur l'onglet « Tous », y compris après une saisie.
class DebtsScreen extends StatefulWidget {
  const DebtsScreen({super.key});

  @override
  State<DebtsScreen> createState() => DebtsScreenState();
}

class DebtsScreenState extends State<DebtsScreen> with SingleTickerProviderStateMixin {
  late final TabController _tabs = TabController(length: 3, vsync: this);

  void showAll() {
    if (_tabs.index != 0) _tabs.index = 0;
  }

  Future<void> _open(Widget page) async {
    await Navigator.push(context, MaterialPageRoute(builder: (_) => page));
    if (mounted) showAll();
  }

  @override
  void dispose() {
    _tabs.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text(t('Dettes', 'الديون')),
        bottom: TabBar(
          controller: _tabs,
          tabs: [
            Tab(text: t('Tous', 'الكل')),
            Tab(text: t('On me doit', 'لي عندهم')),
            Tab(text: t('Je dois', 'علي لهم')),
          ],
        ),
      ),
      floatingActionButton: FloatingActionButton.extended(
        heroTag: null,
        onPressed: () => _open(const PartyFormScreen()),
        icon: const Icon(Icons.person_add_alt),
        label: Text(t('Nouvelle dette', 'دين جديد')),
      ),
      body: Reactive<List<Party>>(
        load: Repo.instance.parties,
        builder: (context, parties) {
          final debtors = parties.where((p) => p.balance > 0.0001).toList()
            ..sort((a, b) => b.balance.compareTo(a.balance));
          final creditors = parties.where((p) => p.balance < -0.0001).toList()
            ..sort((a, b) => a.balance.compareTo(b.balance));
          final rec = debtors.fold<double>(0, (a, p) => a + p.balance);
          final pay = creditors.fold<double>(0, (a, p) => a - p.balance);
          return TabBarView(
            controller: _tabs,
            children: [
              _PartyList(
                parties: parties,
                onOpen: _open,
                header: Row(
                  children: [
                    Expanded(
                      child: _MiniTotal(
                        label: t('On me doit', 'لي عندهم'),
                        value: rec,
                        color: positiveColor,
                      ),
                    ),
                    Expanded(
                      child: _MiniTotal(
                        label: t('Je dois', 'علي لهم'),
                        value: pay,
                        color: negativeColor,
                      ),
                    ),
                  ],
                ),
                empty: t(
                  'Aucune dette enregistrée.\nAppuyez sur « + Nouvelle dette ».',
                  'لا توجد ديون مسجلة.\nاضغط على « + دين جديد ».',
                ),
              ),
              _PartyList(
                parties: debtors,
                onOpen: _open,
                header: TotalBanner(
                  label: t('Total que les clients me doivent', 'مجموع ما لي عند الزبائن'),
                  value: fmtMoney(rec),
                  color: positiveColor,
                ),
                empty: t("Personne ne vous doit de l'argent.", 'لا أحد مدين لك.'),
              ),
              _PartyList(
                parties: creditors,
                onOpen: _open,
                header: TotalBanner(
                  label: t('Total que je dois', 'مجموع ما علي'),
                  value: fmtMoney(pay),
                  color: negativeColor,
                ),
                empty: t('Vous ne devez rien.', 'لست مديناً لأحد.'),
              ),
            ],
          );
        },
      ),
    );
  }
}

class _MiniTotal extends StatelessWidget {
  const _MiniTotal({required this.label, required this.value, required this.color});

  final String label;
  final double value;
  final Color color;

  @override
  Widget build(BuildContext context) => Container(
    margin: const EdgeInsetsDirectional.only(start: 16, end: 4, top: 4, bottom: 8),
    padding: const EdgeInsets.all(12),
    decoration: BoxDecoration(
      color: color.withValues(alpha: 0.08),
      borderRadius: BorderRadius.circular(14),
    ),
    child: Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(label, style: const TextStyle(fontWeight: FontWeight.w500)),
        FittedBox(
          child: Text(
            fmtMoney(value),
            style: TextStyle(fontWeight: FontWeight.w700, fontSize: 16, color: color),
          ),
        ),
      ],
    ),
  );
}

class _PartyList extends StatelessWidget {
  const _PartyList({
    required this.parties,
    required this.empty,
    required this.header,
    required this.onOpen,
  });

  final List<Party> parties;
  final String empty;
  final Widget header;
  final Future<void> Function(Widget page) onOpen;

  @override
  Widget build(BuildContext context) {
    if (parties.isEmpty) return EmptyState(icon: Icons.people_alt_outlined, text: empty);
    return ListView(
      padding: const EdgeInsets.only(top: 8, bottom: 96),
      children: [
        header,
        ListCard(
          children: [
            for (final p in parties)
              ListTile(
                leading: InitialAvatar(p.name, color: moneyColor(context, p.balance)),
                title: Text(p.name, style: const TextStyle(fontWeight: FontWeight.w600)),
                subtitle: Text(
                  [
                    partyKinds[p.kind] ?? p.kind,
                    if (p.phone != null) fmtPhone(p.phone!),
                  ].join(' · '),
                ),
                trailing: Column(
                  mainAxisAlignment: MainAxisAlignment.center,
                  crossAxisAlignment: CrossAxisAlignment.end,
                  children: [
                    Text(
                      fmtMoney(p.balance.abs()),
                      style: TextStyle(
                        fontWeight: FontWeight.w700,
                        color: moneyColor(context, p.balance),
                      ),
                    ),
                    Text(
                      p.balance > 0.0001
                          ? t('me doit', 'عليه لي')
                          : p.balance < -0.0001
                          ? t('je lui dois', 'علي له')
                          : t('soldé', 'مسدد'),
                      style: Theme.of(context).textTheme.bodySmall,
                    ),
                  ],
                ),
                onTap: () => onOpen(PartyDetailScreen(partyId: p.id)),
              ),
          ],
        ),
      ],
    );
  }
}
