import 'package:flutter/foundation.dart';

/// Langue de l'interface : 'fr' ou 'ar'. Toute l'application se reconstruit
/// quand elle change (voir main.dart).
final appLang = ValueNotifier<String>('fr');

bool get isAr => appLang.value == 'ar';

/// Texte dans la langue courante : t('Stock', 'المخزون').
String t(String fr, String ar) => isAr ? ar : fr;

/// Unité monétaire affichée.
String get currency => t('MRU', 'أوقية');

/// Noms arabes des unités et comptes créés au départ (s'ils n'ont pas été renommés).
const _seedNamesAr = {
  'u-piece': ('Pièce', 'قطعة', 'pce', 'قطعة'),
  'u-kg': ('Kilogramme', 'كيلوغرام', 'kg', 'كغ'),
  'u-g': ('Gramme', 'غرام', 'g', 'غ'),
  'u-l': ('Litre', 'لتر', 'L', 'لتر'),
  'u-m': ('Mètre', 'متر', 'm', 'م'),
  'u-sac': ('Sac', 'كيس', 'sac', 'كيس'),
  'u-carton': ('Carton', 'كرتون', 'carton', 'كرتون'),
  'u-paquet': ('Paquet', 'علبة', 'paquet', 'علبة'),
  'u-boite': ('Boîte', 'صندوق', 'boîte', 'صندوق'),
  'u-bidon': ('Bidon', 'بيدون', 'bidon', 'بيدون'),
  'u-sachet': ('Sachet', 'كيس صغير', 'sachet', 'كيس صغير'),
  'u-plateau': ('Plateau', 'صينية', 'plateau', 'صينية'),
  'u-douzaine': ('Douzaine', 'دزينة', 'dz', 'دزينة'),
  'a-cash': ('Caisse (espèces)', 'الصندوق (نقداً)', '', ''),
  'a-bankily': ('Bankily', 'بنكيلي', '', ''),
  'a-masrvi': ('Masrvi', 'مصرفي', '', ''),
  'a-sedad': ('Sedad', 'سداد', '', ''),
};

/// Nom affiché d'une unité ou d'un compte de départ.
String seedName(String id, String name) {
  final s = _seedNamesAr[id];
  return isAr && s != null && s.$1 == name ? s.$2 : name;
}

/// Symbole affiché d'une unité de départ.
String seedSymbol(String id, String symbol) {
  final s = _seedNamesAr[id];
  return isAr && s != null && s.$3 == symbol && s.$4.isNotEmpty ? s.$4 : symbol;
}
