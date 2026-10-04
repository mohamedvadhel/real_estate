import '../i18n.dart';

/// Formatage des nombres « à la française » : 12 500,5
String fmtNum(double v, {int decimals = 2}) {
  final neg = v < 0;
  var s = v.abs().toStringAsFixed(decimals);
  var intPart = s;
  var dec = '';
  final dot = s.indexOf('.');
  if (dot >= 0) {
    intPart = s.substring(0, dot);
    dec = s.substring(dot + 1).replaceAll(RegExp(r'0+$'), '');
  }
  final buf = StringBuffer();
  for (var i = 0; i < intPart.length; i++) {
    // Espace insécable : le nombre reste d'un seul bloc, même dans un texte arabe.
    if (i > 0 && (intPart.length - i) % 3 == 0) buf.write('\u00A0');
    buf.write(intPart[i]);
  }
  s = buf.toString() + (dec.isEmpty ? '' : ',$dec');
  return neg && s != '0' ? '-$s' : s;
}

String fmtMoney(double v) => '${fmtNum(v)} $currency';

String fmtQty(double v, String unit) =>
    unit.isEmpty ? fmtNum(v, decimals: 3) : '${fmtNum(v, decimals: 3)} $unit';

String _two(int n) => n.toString().padLeft(2, '0');

String fmtDate(DateTime d) => '${_two(d.day)}/${_two(d.month)}/${d.year}';

String fmtDateTime(DateTime d) => '${fmtDate(d)}\u00A0${_two(d.hour)}:${_two(d.minute)}';

/// Numéro de téléphone affiché d'un seul bloc (ordre conservé en arabe).
String fmtPhone(String phone) => phone.replaceAll(' ', '\u00A0');

/// Lit un nombre saisi : accepte la virgule, les espaces et un petit calcul
/// avec + - x * (ex. « 3x50 + 20 » pour 3 sacs de 50 kg et 20 kg en vrac).
/// Renvoie null si la saisie est vide ou invalide.
double? parseNum(String? input) {
  if (input == null) return null;
  final s = input.replaceAll(RegExp(r'\s'), '').replaceAll(',', '.').toLowerCase();
  if (s.isEmpty) return null;
  if (!RegExp(r'^[-+]?[0-9.]+([x*+\-][0-9.]+)*$').hasMatch(s)) return null;
  double total = 0;
  for (final term in RegExp(r'[-+]?[^+\-]+').allMatches(s)) {
    var t = term.group(0)!;
    var sign = 1.0;
    if (t.startsWith('-')) {
      sign = -1;
      t = t.substring(1);
    } else if (t.startsWith('+')) {
      t = t.substring(1);
    }
    double product = 1;
    for (final f in t.split(RegExp(r'[x*]'))) {
      final n = double.tryParse(f);
      if (n == null) return null;
      product *= n;
    }
    total += sign * product;
  }
  return total;
}

/// Valeur pré-remplie dans un champ de saisie.
String numToInput(double? v) => v == null ? '' : fmtNum(v, decimals: 3).replaceAll('\u00A0', '');
