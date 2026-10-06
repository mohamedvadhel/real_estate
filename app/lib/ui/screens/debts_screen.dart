import 'dart:async';

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
  final _search = TextEditingController();
  String _query = '';

  /// Personne à montrer dans la liste « Tous » au retour d'une saisie.
  String? _focusId;
  final _focusKey = GlobalKey();
  bool _focusScrolled = false;
  Timer? _focusTimer;

  void showAll() {
    if (_tabs.index != 0) _tabs.index = 0;
  }

  Future<void> _open(Widget page) async {
    final result = await Navigator.push<String>(context, MaterialPageRoute(builder: (_) => page));
    if (!mounted) return;
    showAll();
    if (result == null) return;
    setState(() {
      // Une nouvelle personne peut ne pas correspondre à la recherche en cours.
      if (page is PartyFormScreen) {
        _search.clear();
        _query = '';
      }
      _focusId = result;
      _focusScrolled = false;
    });
  }

  /// Fait défiler la liste jusqu'à la personne puis retire le surlignage après quelques secondes.
  void _scrollToFocus() {
    final ctx = _focusKey.currentContext;
    if (_focusId == null || _focusScrolled || ctx == null) return;
    _focusScrolled = true;
    Scrollable.ensureVisible(ctx, alignment: 0.3, duration: const Duration(milliseconds: 350));
    _focusTimer?.cancel();
    _focusTimer = Timer(const Duration(seconds: 3), () {
      if (mounted) setState(() => _focusId = null);
    });
  }

  @override
  void dispose() {
    _tabs.dispose();
    _search.dispose();
    _focusTimer?.cancel();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text(t('Dettes', 'الديون')),
        bottom: PreferredSize(
          preferredSize: const Size.fromHeight(64 + kTextTabBarHeight),
          child: Column(
            children: [
              Padding(
                padding: const EdgeInsets.fromLTRB(16, 0, 16, 10),
                child: TextField(
                  controller: _search,
                  onChanged: (v) => setState(() => _query = v),
                  decoration: InputDecoration(
                    hintText: t('Rechercher un nom ou un téléphone', 'ابحث عن اسم أو هاتف'),
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
              TabBar(
                controller: _tabs,
                tabs: [
                  Tab(text: t('Tous', 'الكل')),
                  Tab(text: t('On me doit', 'لي عندهم')),
                  Tab(text: t('Je dois', 'علي لهم')),
                ],
              ),
            ],
          ),
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
        builder: (context, all) {
          // Les totaux restent ceux de toute la boutique ; la recherche ne filtre que les listes.
          final rec = all.where((p) => p.balance > 0).fold<double>(0, (a, p) => a + p.balance);
          final pay = all.where((p) => p.balance < 0).fold<double>(0, (a, p) => a - p.balance);
          final parties = all.where((p) => matchesSearch(_query, p.name, phone: p.phone)).toList();
          final debtors = parties.where((p) => p.balance > 0.0001).toList()
            ..sort((a, b) => b.balance.compareTo(a.balance));
          final creditors = parties.where((p) => p.balance < -0.0001).toList()
            ..sort((a, b) => a.balance.compareTo(b.balance));
          final searching = _query.trim().isNotEmpty;
          // Les données arrivent après le retour : on défile dès que la ligne existe.
          if (_focusId != null && !_focusScrolled) {
            WidgetsBinding.instance.addPostFrameCallback((_) => _scrollToFocus());
          }
          final noResult = t('Aucun résultat pour « $_query ».', 'لا توجد نتائج لـ « $_query ».');
          return TabBarView(
            controller: _tabs,
            children: [
              _PartyList(
                parties: parties,
                onOpen: _open,
                focusId: _focusId,
                focusKey: _focusKey,
                header: Row(
                  children: [
                    Expanded(
                      child: _MiniTotal(
                        label: t('On me doit', 'لي عندهم'),
                        value: rec,
                        color: positiveColor,
                        first: true,
                      ),
                    ),
                    Expanded(
                      child: _MiniTotal(
                        label: t('Je dois', 'علي لهم'),
                        value: pay,
                        color: negativeColor,
                        first: false,
                      ),
                    ),
                  ],
                ),
                empty: searching
                    ? noResult
                    : t(
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
                empty: searching
                    ? noResult
                    : t("Personne ne vous doit de l'argent.", 'لا أحد مدين لك.'),
              ),
              _PartyList(
                parties: creditors,
                onOpen: _open,
                header: TotalBanner(
                  label: t('Total que je dois', 'مجموع ما علي'),
                  value: fmtMoney(pay),
                  color: negativeColor,
                ),
                empty: searching ? noResult : t('Vous ne devez rien.', 'لست مديناً لأحد.'),
              ),
            ],
          );
        },
      ),
    );
  }
}

class _MiniTotal extends StatelessWidget {
  const _MiniTotal({
    required this.label,
    required this.value,
    required this.color,
    required this.first,
  });

  final bool first;
  final String label;
  final double value;
  final Color color;

  @override
  Widget build(BuildContext context) => Container(
    margin: EdgeInsetsDirectional.only(
      start: first ? 16 : 5,
      end: first ? 5 : 16,
      top: 4,
      bottom: 8,
    ),
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
    this.focusId,
    this.focusKey,
  });

  final List<Party> parties;
  final String empty;
  final Widget header;
  final Future<void> Function(Widget page) onOpen;

  /// Ligne surlignée (et repérée par [focusKey] pour y faire défiler la liste).
  final String? focusId;
  final GlobalKey? focusKey;

  @override
  Widget build(BuildContext context) {
    if (parties.isEmpty) return EmptyState(icon: Icons.person_search_outlined, text: empty);
    // Toutes les lignes sont construites : on peut faire défiler jusqu'à n'importe laquelle.
    return SingleChildScrollView(
      padding: const EdgeInsets.only(top: 8, bottom: 96),
      child: Column(
        children: [
          header,
          ListCard(
            children: [
              for (final p in parties)
                ListTile(
                  key: p.id == focusId ? focusKey : null,
                  tileColor: p.id == focusId
                      ? Theme.of(context).colorScheme.primaryContainer
                      : null,
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
      ),
    );
  }
}
