import 'package:flutter/material.dart';

import '../../data/models.dart';
import '../../data/repo.dart';
import '../../data/settings.dart';
import '../../report/pdf_report.dart';
import '../../sync/sync_service.dart';
import '../format.dart';
import '../widgets/common.dart';
import 'settings_screen.dart';

class DashboardScreen extends StatefulWidget {
  const DashboardScreen({super.key, required this.onOpenTab});

  final ValueChanged<int> onOpenTab;

  @override
  State<DashboardScreen> createState() => _DashboardScreenState();
}

class _DashboardScreenState extends State<DashboardScreen> {
  bool _exporting = false;

  Future<void> _export() async {
    setState(() => _exporting = true);
    try {
      await shareReport(shopName: AppSettings.instance.shopName.value);
    } catch (e) {
      if (mounted) toast(context, 'Erreur lors de la création du rapport : $e');
    } finally {
      if (mounted) setState(() => _exporting = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: ValueListenableBuilder(
          valueListenable: AppSettings.instance.shopName,
          builder: (_, name, _) => Text(name),
        ),
        actions: [
          const _SyncButton(),
          IconButton(
            tooltip: 'Réglages',
            icon: const Icon(Icons.settings_outlined),
            onPressed: () => Navigator.push(
                context, MaterialPageRoute(builder: (_) => const SettingsScreen())),
          ),
        ],
      ),
      body: Reactive<Summary>(
        load: Repo.instance.summary,
        builder: (context, s) => RefreshIndicator(
          onRefresh: () async => SyncService.instance.sync(),
          child: ListView(
            padding: const EdgeInsets.only(bottom: 24),
            children: [
              _NetCard(summary: s),
              _Line(
                icon: Icons.inventory_2_outlined,
                label: "Stock (prix d'achat)",
                detail: '${s.products.where((p) => p.qty > 0).length} produits en stock',
                value: s.stockValue,
                onTap: () => widget.onOpenTab(1),
              ),
              _Line(
                icon: Icons.account_balance_wallet_outlined,
                label: 'Argent disponible',
                detail: 'Caisse + wallets',
                value: s.cash,
                onTap: () => widget.onOpenTab(3),
              ),
              _Line(
                icon: Icons.call_received,
                label: 'Les clients me doivent',
                detail: '${s.debtors.length} personne(s)',
                value: s.receivables,
                onTap: () => widget.onOpenTab(2),
              ),
              _Line(
                icon: Icons.call_made,
                label: 'Je dois',
                detail: '${s.creditors.length} fournisseur(s) / autre(s)',
                value: -s.payables,
                onTap: () => widget.onOpenTab(2),
              ),
              Card(
                child: Padding(
                  padding: const EdgeInsets.all(12),
                  child: Column(children: [
                    _kv('Stock au prix de vente', fmtMoney(s.stockSaleValue)),
                    _kv('Marge potentielle sur le stock', fmtMoney(s.potentialMargin)),
                  ]),
                ),
              ),
              if (s.missingPrice.isNotEmpty)
                _Warning(
                  "${s.missingPrice.length} produit(s) en stock sans prix d'achat : "
                  'la valeur du stock est sous-estimée.',
                  onTap: () => widget.onOpenTab(1),
                ),
              if (s.lowStock.isNotEmpty)
                _Warning(
                  'Stock bas : ${s.lowStock.map((p) => p.name).take(5).join(', ')}'
                  '${s.lowStock.length > 5 ? '…' : ''}',
                  onTap: () => widget.onOpenTab(1),
                ),
              Padding(
                padding: const EdgeInsets.fromLTRB(12, 12, 12, 0),
                child: FilledButton.icon(
                  onPressed: _exporting ? null : _export,
                  icon: _exporting
                      ? const SizedBox.square(
                          dimension: 18, child: CircularProgressIndicator(strokeWidth: 2))
                      : const Icon(Icons.picture_as_pdf_outlined),
                  label: const Text('Télécharger le rapport PDF'),
                  style: FilledButton.styleFrom(minimumSize: const Size.fromHeight(48)),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _kv(String k, String v) => Padding(
        padding: const EdgeInsets.symmetric(vertical: 2),
        child: Row(children: [Expanded(child: Text(k)), Text(v)]),
      );
}

class _NetCard extends StatelessWidget {
  const _NetCard({required this.summary});

  final Summary summary;

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    return Card(
      color: scheme.primaryContainer,
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
          Text('Valeur nette de la boutique',
              style: TextStyle(color: scheme.onPrimaryContainer)),
          const SizedBox(height: 4),
          FittedBox(
            child: Text(fmtMoney(summary.netValue),
                style: Theme.of(context).textTheme.headlineMedium?.copyWith(
                    fontWeight: FontWeight.bold, color: scheme.onPrimaryContainer)),
          ),
          const SizedBox(height: 4),
          Text('Stock + argent + ce qu\'on me doit − ce que je dois',
              style: TextStyle(fontSize: 12, color: scheme.onPrimaryContainer)),
        ]),
      ),
    );
  }
}

class _Line extends StatelessWidget {
  const _Line({
    required this.icon,
    required this.label,
    required this.detail,
    required this.value,
    this.onTap,
  });

  final IconData icon;
  final String label;
  final String detail;
  final double value;
  final VoidCallback? onTap;

  @override
  Widget build(BuildContext context) => Card(
        child: ListTile(
          leading: Icon(icon),
          title: Text(label),
          subtitle: Text(detail),
          trailing: Text(
            fmtMoney(value),
            style: TextStyle(fontWeight: FontWeight.bold, fontSize: 15, color: moneyColor(context, value)),
          ),
          onTap: onTap,
        ),
      );
}

class _Warning extends StatelessWidget {
  const _Warning(this.text, {this.onTap});

  final String text;
  final VoidCallback? onTap;

  @override
  Widget build(BuildContext context) => Card(
        color: Theme.of(context).colorScheme.errorContainer,
        child: ListTile(
          leading: const Icon(Icons.warning_amber_rounded),
          title: Text(text),
          onTap: onTap,
        ),
      );
}

class _SyncButton extends StatelessWidget {
  const _SyncButton();

  @override
  Widget build(BuildContext context) {
    final sync = SyncService.instance;
    return ValueListenableBuilder<SyncState>(
      valueListenable: sync.state,
      builder: (context, st, _) {
        if (st.running) {
          return const Padding(
            padding: EdgeInsets.all(14),
            child: SizedBox.square(dimension: 20, child: CircularProgressIndicator(strokeWidth: 2)),
          );
        }
        final icon = !sync.configured
            ? Icons.cloud_off_outlined
            : st.error != null
                ? Icons.sync_problem
                : Icons.cloud_done_outlined;
        final tip = !sync.configured
            ? 'Synchronisation non configurée'
            : st.error ?? 'Synchronisé ${st.lastSync == null ? '' : fmtDateTime(st.lastSync!)}';
        return IconButton(
          tooltip: tip,
          icon: Icon(icon),
          onPressed: () async {
            if (!sync.configured) {
              toast(context, 'Données enregistrées sur le téléphone. Configurez le serveur dans Réglages pour la sauvegarde en ligne.');
              return;
            }
            final err = await sync.sync();
            if (context.mounted) toast(context, err ?? 'Synchronisation terminée');
          },
        );
      },
    );
  }
}
