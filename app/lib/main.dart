import 'package:flutter/material.dart';

import 'data/db.dart';
import 'data/settings.dart';
import 'sync/sync_service.dart';
import 'ui/screens/cash_screen.dart';
import 'ui/screens/dashboard_screen.dart';
import 'ui/screens/debts_screen.dart';
import 'ui/screens/stock_screen.dart';

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
    final scheme = ColorScheme.fromSeed(seedColor: const Color(0xFF2E7D32));
    return MaterialApp(
      title: 'Compte Boutique',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorScheme: scheme,
        useMaterial3: true,
        inputDecorationTheme: const InputDecorationTheme(border: OutlineInputBorder()),
        cardTheme: const CardThemeData(margin: EdgeInsets.symmetric(horizontal: 12, vertical: 6)),
      ),
      home: const HomeShell(),
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

  void _go(int i) => setState(() => _index = i);

  @override
  Widget build(BuildContext context) {
    final pages = [
      DashboardScreen(onOpenTab: _go),
      const StockScreen(),
      const DebtsScreen(),
      const CashScreen(),
    ];
    return Scaffold(
      body: IndexedStack(index: _index, children: pages),
      bottomNavigationBar: NavigationBar(
        selectedIndex: _index,
        onDestinationSelected: _go,
        destinations: const [
          NavigationDestination(icon: Icon(Icons.dashboard_outlined), selectedIcon: Icon(Icons.dashboard), label: 'Situation'),
          NavigationDestination(icon: Icon(Icons.inventory_2_outlined), selectedIcon: Icon(Icons.inventory_2), label: 'Stock'),
          NavigationDestination(icon: Icon(Icons.people_outline), selectedIcon: Icon(Icons.people), label: 'Dettes'),
          NavigationDestination(icon: Icon(Icons.account_balance_wallet_outlined), selectedIcon: Icon(Icons.account_balance_wallet), label: 'Caisse'),
        ],
      ),
    );
  }
}
