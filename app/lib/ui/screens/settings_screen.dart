import 'package:flutter/material.dart';

import '../../data/settings.dart';
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
    toast(context, !_sync.configured
        ? 'Enregistré'
        : err == null
            ? 'Connexion réussie, données synchronisées'
            : 'Erreur : $err');
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Réglages')),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          TextField(
            controller: _shop,
            decoration: const InputDecoration(labelText: 'Nom de la boutique'),
          ),
          const SizedBox(height: 24),
          Text('Sauvegarde en ligne (Neon)', style: Theme.of(context).textTheme.titleMedium),
          const SizedBox(height: 4),
          const Text(
            "L'application fonctionne sans internet. Avec un serveur configuré, "
            'les données sont sauvegardées en ligne dès que le réseau est disponible.',
          ),
          const SizedBox(height: 12),
          TextField(
            controller: _url,
            keyboardType: TextInputType.url,
            decoration: const InputDecoration(
                labelText: 'Adresse du serveur', hintText: 'https://mon-projet.vercel.app'),
          ),
          const SizedBox(height: 12),
          TextField(
            controller: _key,
            obscureText: !_showKey,
            decoration: InputDecoration(
              labelText: "Clé d'accès (APP_KEY)",
              suffixIcon: IconButton(
                icon: Icon(_showKey ? Icons.visibility_off : Icons.visibility),
                onPressed: () => setState(() => _showKey = !_showKey),
              ),
            ),
          ),
          const SizedBox(height: 12),
          FilledButton.icon(
            onPressed: _busy ? null : _saveAndSync,
            icon: _busy
                ? const SizedBox.square(dimension: 18, child: CircularProgressIndicator(strokeWidth: 2))
                : const Icon(Icons.save_outlined),
            label: const Text('Enregistrer et synchroniser'),
            style: FilledButton.styleFrom(minimumSize: const Size.fromHeight(48)),
          ),
          const SizedBox(height: 8),
          ValueListenableBuilder<SyncState>(
            valueListenable: _sync.state,
            builder: (_, st, _) => Text(
              st.lastSync == null
                  ? 'Jamais synchronisé'
                  : 'Dernière synchronisation : ${fmtDateTime(st.lastSync!)}'
                      '${st.error == null ? '' : '\nDernière erreur : ${st.error}'}',
              style: Theme.of(context).textTheme.bodySmall,
            ),
          ),
          const Divider(height: 32),
          ListTile(
            contentPadding: EdgeInsets.zero,
            leading: const Icon(Icons.straighten),
            title: const Text('Unités de mesure'),
            subtitle: const Text('kg, litre, sac, carton… ajouter les vôtres'),
            trailing: const Icon(Icons.chevron_right),
            onTap: () =>
                Navigator.push(context, MaterialPageRoute(builder: (_) => const UnitsScreen())),
          ),
        ],
      ),
    );
  }
}
