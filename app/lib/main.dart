import 'package:flutter/material.dart';
import 'package:flutter_localizations/flutter_localizations.dart';

import 'data/db.dart';
import 'data/settings.dart';
import 'i18n.dart';
import 'sync/sync_service.dart';
import 'ui/screens/cash_screen.dart';
import 'ui/screens/dashboard_screen.dart';
import 'ui/screens/debts_screen.dart';
import 'ui/screens/stock_screen.dart';
import 'ui/theme.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  final db = await AppDb.open();
  await AppSettings.init();
  final sync = await SyncService.init(db);
  runApp(const CompteBoutiqueApp());
  if (sync.configured) sync.sync();
}

class CompteBoutiqueApp extends StatelessWidget {
  const CompteBoutiqueApp({super.key});

  @override
  Widget build(BuildContext context) {
    return ValueListenableBuilder<String>(
      valueListenable: appLang,
      builder: (context, lang, _) => MaterialApp(
        title: 'Compte Boutique',
        debugShowCheckedModeBanner: false,
        theme: buildTheme(),
        locale: Locale(lang),
        supportedLocales: const [Locale('fr'), Locale('ar')],
        localizationsDelegates: const [
          GlobalMaterialLocalizations.delegate,
          GlobalWidgetsLocalizations.delegate,
          GlobalCupertinoLocalizations.delegate,
        ],
        // La clé force la reconstruction complète des écrans au changement de langue.
        home: HomeShell(key: ValueKey(lang)),
      ),
    );
  }
}

class HomeShell extends StatefulWidget {
  const HomeShell({super.key});

  @override
  State<HomeShell> createState() => _HomeShellState();
}

class _HomeShellState extends State<HomeShell> {
  int _index = 0;
  final _debtsKey = GlobalKey<DebtsScreenState>();

  void _go(int i) {
    // Les dettes s'ouvrent toujours sur « Tous ».
    if (i == 2) _debtsKey.currentState?.showAll();
    setState(() => _index = i);
  }

  @override
  Widget build(BuildContext context) {
    final pages = [
      DashboardScreen(onOpenTab: _go),
      const StockScreen(),
      DebtsScreen(key: _debtsKey),
      const CashScreen(),
    ];
    return Scaffold(
      body: IndexedStack(index: _index, children: pages),
      bottomNavigationBar: NavigationBar(
        selectedIndex: _index,
        onDestinationSelected: _go,
        destinations: [
          NavigationDestination(
            icon: const Icon(Icons.space_dashboard_outlined),
            selectedIcon: const Icon(Icons.space_dashboard),
            label: t('Situation', 'الوضعية'),
          ),
          NavigationDestination(
            icon: const Icon(Icons.inventory_2_outlined),
            selectedIcon: const Icon(Icons.inventory_2),
            label: t('Stock', 'المخزون'),
          ),
          NavigationDestination(
            icon: const Icon(Icons.people_alt_outlined),
            selectedIcon: const Icon(Icons.people_alt),
            label: t('Dettes', 'الديون'),
          ),
          NavigationDestination(
            icon: const Icon(Icons.account_balance_wallet_outlined),
            selectedIcon: const Icon(Icons.account_balance_wallet),
            label: t('Caisse', 'الصندوق'),
          ),
        ],
      ),
    );
  }
}
