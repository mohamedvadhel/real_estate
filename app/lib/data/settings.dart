import 'package:flutter/foundation.dart';
import 'package:shared_preferences/shared_preferences.dart';

/// Préférences simples de l'application.
class AppSettings {
  AppSettings._(this._prefs) : shopName = ValueNotifier(_prefs.getString(_kShop) ?? 'Ma boutique');

  static AppSettings? _instance;
  static AppSettings get instance => _instance!;

  static Future<AppSettings> init() async =>
      _instance = AppSettings._(await SharedPreferences.getInstance());

  static const _kShop = 'shop_name';

  final SharedPreferences _prefs;
  final ValueNotifier<String> shopName;

  Future<void> setShopName(String name) async {
    final n = name.trim().isEmpty ? 'Ma boutique' : name.trim();
    shopName.value = n;
    await _prefs.setString(_kShop, n);
  }
}
