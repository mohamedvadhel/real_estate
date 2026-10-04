import 'package:flutter/material.dart';

import '../../data/settings.dart';
import '../../i18n.dart';
import '../../sync/sync_service.dart';
import '../format.dart';
import '../widgets/common.dart';
import 'units_screen.dart';

class SettingsScreen extends StatefulWidget {
  const SettingsScreen({super.key});

  @override
  State<SettingsScreen> createState() => _SettingsScreenState();
}

class _SettingsScreenState extends State<SettingsScreen> {
  final _sync = SyncService.instance;
  late final _shop = TextEditingController(text: AppSettings.instance.shopName.value);
  late final _url = TextEditingController(text: _sync.serverUrl);
  late final _key = TextEditingController(text: _sync.appKey);
  bool _busy = false;
  bool _showKey = false;

  Future<void> _saveAndSync() async {
    setState(() => _busy = true);
    await AppSettings.instance.setShopName(_shop.text);
    await _sync.configure(_url.text, _key.text);
    String? err;
    if (_sync.configured) {
      err = await _sync.test() ?? await _sync.sync();
    }
    if (!mounted) return;
    setState(() => _busy = false);
    toast(
      context,
      !_sync.configured
          ? t('Enregistré', 'تم الحفظ')
          : err == null
          ? t('Connexion réussie, données synchronisées', 'تم الاتصال ومزامنة البيانات')
          : '${t('Erreur', 'خطأ')} : $err',
    );
  }

  Future<void> _setLang(String lang) async {
    await AppSettings.instance.setShopName(_shop.text);
    await AppSettings.instance.setLang(lang);
    // L'application se reconstruit dans la nouvelle langue.
    if (mounted) Navigator.pop(context);
  }

  @override
  Widget build(BuildContext context) {
    const gap = SizedBox(height: 14);
    return Scaffold(
      appBar: AppBar(title: Text(t('Réglages', 'الإعدادات'))),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          Text(t('Langue', 'اللغة'), style: Theme.of(context).textTheme.titleMedium),
          const SizedBox(height: 8),
          SegmentedButton<String>(
            segments: const [
              ButtonSegment(value: 'fr', label: Text('Français')),
              ButtonSegment(value: 'ar', label: Text('العربية')),
            ],
            selected: {appLang.value},
            onSelectionChanged: (s) => _setLang(s.first),
          ),
          const SizedBox(height: 20),
          TextField(
            controller: _shop,
            decoration: InputDecoration(labelText: t('Nom de la boutique', 'اسم المتجر')),
          ),
          const SizedBox(height: 24),
          Text(
            t('Sauvegarde en ligne', 'الحفظ عبر الإنترنت'),
            style: Theme.of(context).textTheme.titleMedium,
          ),
          const SizedBox(height: 4),
          Text(
            t(
              "L'application fonctionne sans internet. Avec un serveur configuré, "
                  'les données sont sauvegardées en ligne dès que le réseau est disponible.',
              'التطبيق يعمل بدون إنترنت. عند إعداد الخادم، تُحفظ البيانات عبر الإنترنت عند توفر الشبكة.',
            ),
          ),
          gap,
          TextField(
            controller: _url,
            keyboardType: TextInputType.url,
            textDirection: TextDirection.ltr,
            decoration: InputDecoration(
              labelText: t('Adresse du serveur', 'عنوان الخادم'),
              hintText: 'https://mon-projet.vercel.app',
            ),
          ),
          gap,
          TextField(
            controller: _key,
            obscureText: !_showKey,
            textDirection: TextDirection.ltr,
            decoration: InputDecoration(
              labelText: t("Clé d'accès (APP_KEY)", 'مفتاح الدخول (APP_KEY)'),
              suffixIcon: IconButton(
                icon: Icon(_showKey ? Icons.visibility_off : Icons.visibility),
                onPressed: () => setState(() => _showKey = !_showKey),
              ),
            ),
          ),
          gap,
          FilledButton.icon(
            onPressed: _busy ? null : _saveAndSync,
            icon: _busy
                ? const SizedBox.square(
                    dimension: 18,
                    child: CircularProgressIndicator(strokeWidth: 2),
                  )
                : const Icon(Icons.save_outlined),
            label: Text(t('Enregistrer et synchroniser', 'حفظ ومزامنة')),
            style: FilledButton.styleFrom(minimumSize: const Size.fromHeight(52)),
          ),
          const SizedBox(height: 8),
          ValueListenableBuilder<SyncState>(
            valueListenable: _sync.state,
            builder: (_, st, _) => Text(
              st.lastSync == null
                  ? t('Jamais synchronisé', 'لم تتم المزامنة بعد')
                  : '${t('Dernière synchronisation', 'آخر مزامنة')} : ${fmtDateTime(st.lastSync!)}'
                        '${st.error == null ? '' : '\n${t('Dernière erreur', 'آخر خطأ')} : ${st.error}'}',
              style: Theme.of(context).textTheme.bodySmall,
            ),
          ),
          const SizedBox(height: 16),
          ListCard(
            children: [
              ListTile(
                leading: const Icon(Icons.straighten),
                title: Text(t('Unités de mesure', 'وحدات القياس')),
                subtitle: Text(
                  t(
                    'kg, litre, sac, carton… ajouter les vôtres',
                    'كغ، لتر، كيس، كرتون… أضف وحداتك',
                  ),
                ),
                trailing: const Icon(Icons.chevron_right),
                onTap: () =>
                    Navigator.push(context, MaterialPageRoute(builder: (_) => const UnitsScreen())),
              ),
            ],
          ),
        ],
      ),
    );
  }
}
