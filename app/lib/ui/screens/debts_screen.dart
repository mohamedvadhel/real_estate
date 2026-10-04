import 'package:flutter/material.dart';

import '../../data/models.dart';
import '../../data/repo.dart';
import '../format.dart';
import '../widgets/common.dart';
import 'party_detail_screen.dart';
import 'party_form_screen.dart';

/// Dettes des clients envers la boutique et dettes de la boutique.
class DebtsScreen extends StatelessWidget {
  const DebtsScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return DefaultTabController(
      length: 3,
      child: Scaffold(
        appBar: AppBar(
          title: const Text('Dettes'),
          bottom: const TabBar(tabs: [
            Tab(text: 'On me doit'),
            Tab(text: 'Je dois'),
            Tab(text: 'Tous'),
          ]),
        ),
        floatingActionButton: FloatingActionButton.extended(
          onPressed: () => Navigator.push(
              context, MaterialPageRoute(builder: (_) => const PartyFormScreen())),
          icon: const Icon(Icons.person_add_alt),
          label: const Text('Personne'),
        ),
        body: Reactive<List<Party>>(
          load: Repo.instance.parties,
          builder: (context, parties) => TabBarView(children: [
            _PartyList(
              parties: parties.where((p) => p.balance > 0.0001).toList()
                ..sort((a, b) => b.balance.compareTo(a.balance)),
              totalLabel: 'Total que les clients me doivent',
              empty: 'Personne ne vous doit de l\'argent.\nAjoutez un client avec « + Personne ».',
            ),
            _PartyList(
              parties: parties.where((p) => p.balance < -0.0001).toList()
                ..sort((a, b) => a.balance.compareTo(b.balance)),
              totalLabel: 'Total que je dois',
              empty: 'Vous ne devez rien.\nAjoutez un fournisseur avec « + Personne ».',
            ),
            _PartyList(parties: parties, empty: 'Aucune personne enregistrée.'),
          ]),
        ),
      ),
    );
  }
}

class _PartyList extends StatelessWidget {
  const _PartyList({required this.parties, required this.empty, this.totalLabel});

  final List<Party> parties;
  final String empty;
  final String? totalLabel;

  @override
  Widget build(BuildContext context) {
    if (parties.isEmpty) return EmptyState(icon: Icons.people_outline, text: empty);
    final total = parties.fold<double>(0, (a, p) => a + p.balance.abs());
    return Column(children: [
      if (totalLabel != null)
        Container(
          width: double.infinity,
          padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
          color: Theme.of(context).colorScheme.surfaceContainerHighest,
          child: Text('$totalLabel : ${fmtMoney(total)}',
              style: const TextStyle(fontWeight: FontWeight.w600)),
        ),
      Expanded(
        child: ListView.separated(
          padding: const EdgeInsets.only(bottom: 88),
          itemCount: parties.length,
          separatorBuilder: (_, _) => const Divider(height: 1),
          itemBuilder: (context, i) {
            final p = parties[i];
            return ListTile(
              leading: CircleAvatar(child: Text(p.name.isEmpty ? '?' : p.name[0].toUpperCase())),
              title: Text(p.name),
              subtitle: Text([partyKinds[p.kind] ?? p.kind, if (p.phone != null) p.phone!].join(' · ')),
              trailing: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                crossAxisAlignment: CrossAxisAlignment.end,
                children: [
                  Text(fmtMoney(p.balance.abs()),
                      style: TextStyle(
                          fontWeight: FontWeight.bold, color: moneyColor(context, p.balance))),
                  Text(
                    p.balance > 0.0001 ? 'me doit' : p.balance < -0.0001 ? 'je lui dois' : 'soldé',
                    style: Theme.of(context).textTheme.bodySmall,
                  ),
                ],
              ),
              onTap: () => Navigator.push(context,
                  MaterialPageRoute(builder: (_) => PartyDetailScreen(partyId: p.id))),
            );
          },
        ),
      ),
    ]);
  }
}
