import 'package:flutter/foundation.dart';
import 'package:shared_preferences/shared_preferences.dart';

import '../i18n.dart';
import '../ui/sort.dart';

/// Préférences simples de l'application.
class AppSettings {
  AppSettings._(this._prefs) : shopName = ValueNotifier(_prefs.getString(_kShop) ?? '') {
    appLang.value = _prefs.getString(_kLang) ?? 'fr';
  }

  static AppSettings? _instance;
  static AppSettings get instance => _instance!;

  static Future<AppSettings> init() async =>
      _instance = AppSettings._(await SharedPreferences.getInstance());

  static const _kShop = 'shop_name';
  static const _kLang = 'lang';

  final SharedPreferences _prefs;
  final ValueNotifier<String> shopName;

  /// Nom affiché (valeur par défaut traduite si aucun nom n'est saisi).
  String get displayShopName => shopName.value.isEmpty ? t('Ma boutique', 'متجري') : shopName.value;

  Future<void> setShopName(String name) async {
    shopName.value = name.trim();
    await _prefs.setString(_kShop, name.trim());
  }

  final _sorts = <String, ValueNotifier<ListSort>>{};

  /// Ordre choisi pour une liste ('products' ou 'parties'), par défaut le nom.
  ValueNotifier<ListSort> sortFor(String list) => _sorts.putIfAbsent(list, () {
    final saved = _prefs.getString('sort_$list');
    return ValueNotifier(
      ListSort.values.firstWhere((s) => s.name == saved, orElse: () => ListSort.name),
    );
  });

  Future<void> setSort(String list, ListSort sort) async {
    sortFor(list).value = sort;
    await _prefs.setString('sort_$list', sort.name);
  }

  Future<void> setLang(String lang) async {
    appLang.value = lang;
    await _prefs.setString(_kLang, lang);
  }
}
