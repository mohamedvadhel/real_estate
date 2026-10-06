import 'package:flutter/material.dart';

import '../../data/models.dart';
import '../../data/repo.dart';
import '../../data/settings.dart';
import '../../i18n.dart';
import '../../report/pdf_report.dart';
import '../../sync/sync_service.dart';
import '../format.dart';
import '../theme.dart';
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
      await shareReport(shopName: AppSettings.instance.displayShopName);
    } catch (e) {
      if (mounted) {
        toast(
          context,
          '${t('Erreur lors de la création du rapport', 'خطأ أثناء إنشاء التقرير')} : $e',
        );
      }
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
          builder: (_, _, _) => Text(AppSettings.instance.displayShopName),
        ),
        actions: [
          const _SyncButton(),
          IconButton(
            tooltip: t('Réglages', 'الإعدادات'),
            icon: const Icon(Icons.settings_outlined),
            onPressed: () =>
                Navigator.push(context, MaterialPageRoute(builder: (_) => const SettingsScreen())),
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
              Padding(
                padding: const EdgeInsets.symmetric(horizontal: 12),
                child: GridView.count(
                  crossAxisCount: 2,
                  shrinkWrap: true,
                  physics: const NeverScrollableScrollPhysics(),
                  childAspectRatio: 1.55,
                  children: [
                    _StatTile(
                      icon: Icons.inventory_2_outlined,
                      color: const Color(0xFF2563EB),
                      label: t('Stock', 'المخزون'),
                      detail: t(
                        '${s.products.where((p) => p.qty > 0).length} produits',
                        '${s.products.where((p) => p.qty > 0).length} منتج',
                      ),
                      value: s.stockValue,
                      onTap: () => widget.onOpenTab(1),
                    ),
                    _StatTile(
                      icon: Icons.account_balance_wallet_outlined,
                      color: brandColor,
                      label: t('Argent disponible', 'المال المتوفر'),
                      detail: t('Caisse + wallets', 'الصندوق + المحافظ'),
                      value: s.cash,
                      onTap: () => widget.onOpenTab(3),
                    ),
                    _StatTile(
                      icon: Icons.south_west,
                      color: positiveColor,
                      label: t('On me doit', 'لي عند الناس'),
                      detail: t('${s.debtors.length} personne(s)', '${s.debtors.length} شخص'),
                      value: s.receivables,
                      onTap: () => widget.onOpenTab(2),
                    ),
                    _StatTile(
                      icon: Icons.north_east,
                      color: negativeColor,
                      label: t('Je dois', 'علي للناس'),
                      detail: t('${s.creditors.length} personne(s)', '${s.creditors.length} شخص'),
                      value: s.payables,
                      onTap: () => widget.onOpenTab(2),
                    ),
                  ],
                ),
              ),
              Card(
                child: Padding(
                  padding: const EdgeInsets.all(16),
                  child: Column(
                    children: [
                      _kv(
                        t('Stock au prix de vente', 'المخزون بسعر البيع'),
                        fmtMoney(s.stockSaleValue),
                      ),
                      const SizedBox(height: 6),
                      _kv(
                        t('Marge potentielle', 'الربح المتوقع'),
                        fmtMoney(s.potentialMargin),
                        color: moneyColor(context, s.potentialMargin),
                      ),
                    ],
                  ),
                ),
              ),
              if (s.missingPrice.isNotEmpty)
                _Warning(
                  t(
                    "${s.missingPrice.length} produit(s) sans prix d'achat : la valeur du stock est sous-estimée.",
                    '${s.missingPrice.length} منتج بدون سعر شراء: قيمة المخزون أقل من الحقيقة.',
                  ),
                  onTap: () => widget.onOpenTab(1),
                ),
              if (s.lowStock.isNotEmpty)
                _Warning(
                  '${t('Stock bas', 'مخزون منخفض')} : ${s.lowStock.map((p) => p.name).take(5).join('، ')}'
                  '${s.lowStock.length > 5 ? '…' : ''}',
                  onTap: () => widget.onOpenTab(1),
                ),
              Padding(
                padding: const EdgeInsets.fromLTRB(16, 12, 16, 0),
                child: FilledButton.icon(
                  onPressed: _exporting ? null : _export,
                  icon: _exporting
                      ? const SizedBox.square(
                          dimension: 18,
                          child: CircularProgressIndicator(strokeWidth: 2),
                        )
                      : const Icon(Icons.picture_as_pdf_outlined),
                  label: Text(t('Télécharger le rapport PDF', 'تحميل التقرير PDF')),
                  style: FilledButton.styleFrom(minimumSize: const Size.fromHeight(52)),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _kv(String k, String v, {Color? color}) => Row(
    children: [
      Expanded(child: Text(k)),
      Text(
        v,
        style: TextStyle(fontWeight: FontWeight.w600, color: color),
      ),
    ],
  );
}

class _NetCard extends StatelessWidget {
  const _NetCard({required this.summary});

  final Summary summary;

  @override
  Widget build(BuildContext context) {
    const white = Colors.white;
    return Container(
      margin: const EdgeInsets.fromLTRB(16, 4, 16, 8),
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        borderRadius: BorderRadius.circular(20),
        gradient: const LinearGradient(
          colors: [Color(0xFF0E7C66), Color(0xFF0B5D4E)],
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
        ),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              const Icon(Icons.storefront_outlined, color: white, size: 20),
              const SizedBox(width: 8),
              Expanded(
                child: Text(
                  t('Valeur nette de la boutique', 'القيمة الصافية للمتجر'),
                  style: const TextStyle(color: white, fontSize: 15),
                ),
              ),
            ],
          ),
          const SizedBox(height: 10),
          FittedBox(
            child: Text(
              fmtMoney(summary.netValue),
              style: const TextStyle(color: white, fontSize: 32, fontWeight: FontWeight.w800),
            ),
          ),
          const SizedBox(height: 8),
          Text(
            t(
              "Stock + argent + ce qu'on me doit − ce que je dois",
              'المخزون + المال + ما لي عند الناس − ما علي',
            ),
            style: TextStyle(color: white.withValues(alpha: 0.8), fontSize: 12),
          ),
        ],
      ),
    );
  }
}

class _StatTile extends StatelessWidget {
  const _StatTile({
    required this.icon,
    required this.color,
    required this.label,
    required this.detail,
    required this.value,
    this.onTap,
  });

  final IconData icon;
  final Color color;
  final String label;
  final String detail;
  final double value;
  final VoidCallback? onTap;

  @override
  Widget build(BuildContext context) => Card(
    margin: const EdgeInsets.all(4),
    clipBehavior: Clip.antiAlias,
    child: InkWell(
      onTap: onTap,
      child: Padding(
        padding: const EdgeInsets.all(12),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                CircleAvatar(
                  radius: 16,
                  backgroundColor: color.withValues(alpha: 0.12),
                  child: Icon(icon, size: 18, color: color),
                ),
                const SizedBox(width: 8),
                Expanded(
                  child: Text(
                    label,
                    maxLines: 1,
                    overflow: TextOverflow.ellipsis,
                    style: const TextStyle(fontWeight: FontWeight.w600),
                  ),
                ),
              ],
            ),
            const Spacer(),
            FittedBox(
              child: Text(
                fmtMoney(value),
                style: TextStyle(fontSize: 18, fontWeight: FontWeight.w700, color: color),
              ),
            ),
            Text(detail, style: Theme.of(context).textTheme.bodySmall),
          ],
        ),
      ),
    ),
  );
}

class _Warning extends StatelessWidget {
  const _Warning(this.text, {this.onTap});

  final String text;
  final VoidCallback? onTap;

  @override
  Widget build(BuildContext context) => Card(
    color: const Color(0xFFFFF4E5),
    child: ListTile(
      leading: const Icon(Icons.warning_amber_rounded, color: Color(0xFFB45309)),
      title: Text(text, style: const TextStyle(fontSize: 14)),
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
            ? t('Synchronisation non configurée', 'المزامنة غير مفعلة')
            : st.error ??
                  '${t('Synchronisé', 'تمت المزامنة')} ${st.lastSync == null ? '' : fmtDateTime(st.lastSync!)}';
        return IconButton(
          tooltip: tip,
          icon: Icon(icon, color: st.error != null ? negativeColor : null),
          onPressed: () async {
            if (!sync.configured) {
              toast(
                context,
                t(
                  'Données enregistrées sur le téléphone. Configurez le serveur dans Réglages pour la sauvegarde en ligne.',
                  'البيانات محفوظة على الهاتف. أضف الخادم في الإعدادات للحفظ عبر الإنترنت.',
                ),
              );
              return;
            }
            final err = await sync.sync();
            if (context.mounted) {
              toast(context, err ?? t('Synchronisation terminée', 'تمت المزامنة'));
            }
          },
        );
      },
    );
  }
}
