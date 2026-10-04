import 'package:flutter/material.dart';

import '../../data/db.dart';
import '../../i18n.dart';
import '../format.dart';
import '../theme.dart';

/// Charge des données et les recharge à chaque modification de la base.
class Reactive<T> extends StatefulWidget {
  const Reactive({super.key, required this.load, required this.builder});

  final Future<T> Function() load;
  final Widget Function(BuildContext context, T data) builder;

  @override
  State<Reactive<T>> createState() => _ReactiveState<T>();
}

class _ReactiveState<T> extends State<Reactive<T>> {
  T? _data;
  Object? _error;
  int _request = 0;

  @override
  void initState() {
    super.initState();
    AppDb.instance.changes.addListener(_reload);
    _reload();
  }

  @override
  void didUpdateWidget(covariant Reactive<T> oldWidget) {
    super.didUpdateWidget(oldWidget);
    _reload();
  }

  @override
  void dispose() {
    AppDb.instance.changes.removeListener(_reload);
    super.dispose();
  }

  Future<void> _reload() async {
    final req = ++_request;
    try {
      final d = await widget.load();
      if (mounted && req == _request) {
        setState(() {
          _data = d;
          _error = null;
        });
      }
    } catch (e) {
      if (mounted && req == _request) setState(() => _error = e);
    }
  }

  @override
  Widget build(BuildContext context) {
    if (_error != null && _data == null) {
      return Center(child: Text('${t('Erreur', 'خطأ')} : $_error'));
    }
    final d = _data;
    if (d == null) return const Center(child: CircularProgressIndicator());
    return widget.builder(context, d);
  }
}

class EmptyState extends StatelessWidget {
  const EmptyState({super.key, required this.icon, required this.text});

  final IconData icon;
  final String text;

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    return Center(
      child: Padding(
        padding: const EdgeInsets.all(32),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            CircleAvatar(
              radius: 36,
              backgroundColor: scheme.primaryContainer.withValues(alpha: 0.6),
              child: Icon(icon, size: 34, color: scheme.primary),
            ),
            const SizedBox(height: 16),
            Text(
              text,
              textAlign: TextAlign.center,
              style: Theme.of(context).textTheme.bodyLarge
                  ?.copyWith(color: scheme.onSurfaceVariant),
            ),
          ],
        ),
      ),
    );
  }
}

/// Bandeau de total en haut des listes.
class TotalBanner extends StatelessWidget {
  const TotalBanner({super.key, required this.label, required this.value, this.color});

  final String label;
  final String value;
  final Color? color;

  @override
  Widget build(BuildContext context) => Container(
    margin: const EdgeInsets.fromLTRB(16, 4, 16, 8),
    padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
    decoration: BoxDecoration(
      color: (color ?? brandColor).withValues(alpha: 0.08),
      borderRadius: BorderRadius.circular(14),
    ),
    child: Row(
      children: [
        Expanded(
          child: Text(label, style: const TextStyle(fontWeight: FontWeight.w500)),
        ),
        Text(
          value,
          style: TextStyle(fontWeight: FontWeight.w700, fontSize: 16, color: color ?? brandColor),
        ),
      ],
    ),
  );
}

class SectionTitle extends StatelessWidget {
  const SectionTitle(this.text, {super.key});

  final String text;

  @override
  Widget build(BuildContext context) => Padding(
    padding: const EdgeInsets.fromLTRB(20, 18, 20, 6),
    child: Text(
      text,
      style: Theme.of(context).textTheme.titleSmall?.copyWith(
        color: Theme.of(context).colorScheme.onSurfaceVariant,
        fontWeight: FontWeight.w600,
      ),
    ),
  );
}

/// Pastille ronde avec la première lettre d'un nom.
class InitialAvatar extends StatelessWidget {
  const InitialAvatar(this.name, {super.key, this.color});

  final String name;
  final Color? color;

  @override
  Widget build(BuildContext context) {
    final c = color ?? brandColor;
    return CircleAvatar(
      backgroundColor: c.withValues(alpha: 0.12),
      foregroundColor: c,
      child: Text(
        name.trim().isEmpty ? '?' : name.trim().characters.first.toUpperCase(),
        style: const TextStyle(fontWeight: FontWeight.w700),
      ),
    );
  }
}

/// Champ numérique qui accepte la virgule et les petits calculs (3x50+20).
class NumberField extends StatelessWidget {
  const NumberField({
    super.key,
    required this.controller,
    required this.label,
    this.suffix,
    this.required = false,
    this.allowNegative = false,
    this.helper,
    this.autofocus = false,
    this.onChanged,
  });

  final TextEditingController controller;
  final String label;
  final String? suffix;
  final bool required;
  final bool allowNegative;
  final String? helper;
  final bool autofocus;
  final ValueChanged<String>? onChanged;

  @override
  Widget build(BuildContext context) => TextFormField(
    controller: controller,
    autofocus: autofocus,
    onChanged: onChanged,
    keyboardType: const TextInputType.numberWithOptions(decimal: true, signed: true),
    decoration: InputDecoration(labelText: label, suffixText: suffix, helperText: helper),
    validator: (v) {
      if (v == null || v.trim().isEmpty) return required ? t('Obligatoire', 'إلزامي') : null;
      final n = parseNum(v);
      if (n == null) return t('Nombre invalide', 'رقم غير صالح');
      if (!allowNegative && n < 0) return t('Doit être positif', 'يجب أن يكون موجباً');
      return null;
    },
  );
}

class AmountResult {
  AmountResult(this.value, this.note);
  final double value;
  final String note;
}

/// Boîte de dialogue : saisir un montant / une quantité et une note.
Future<AmountResult?> askAmount(
  BuildContext context, {
  required String title,
  required String label,
  String? suffix,
  String? helper,
  double? initial,
  bool allowZero = false,
}) {
  final ctrl = TextEditingController(text: numToInput(initial));
  final note = TextEditingController();
  final formKey = GlobalKey<FormState>();
  return showDialog<AmountResult>(
    context: context,
    builder: (ctx) {
      void submit() {
        if (!formKey.currentState!.validate()) return;
        Navigator.pop(ctx, AmountResult(parseNum(ctrl.text)!, note.text.trim()));
      }

      return AlertDialog(
        title: Text(title),
        content: Form(
          key: formKey,
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              TextFormField(
                controller: ctrl,
                autofocus: true,
                keyboardType: const TextInputType.numberWithOptions(decimal: true),
                decoration: InputDecoration(
                  labelText: label,
                  suffixText: suffix,
                  helperText: helper,
                ),
                onFieldSubmitted: (_) => submit(),
                validator: (v) {
                  final n = parseNum(v);
                  if (n == null) return t('Nombre invalide', 'رقم غير صالح');
                  if (n < 0) return t('Doit être positif', 'يجب أن يكون موجباً');
                  if (n == 0 && !allowZero) {
                    return t('Doit être supérieur à 0', 'يجب أن يكون أكبر من 0');
                  }
                  return null;
                },
              ),
              const SizedBox(height: 12),
              TextFormField(
                controller: note,
                decoration: InputDecoration(labelText: t('Note (facultatif)', 'ملاحظة (اختياري)')),
              ),
            ],
          ),
        ),
        actions: [
          TextButton(onPressed: () => Navigator.pop(ctx), child: Text(t('Annuler', 'إلغاء'))),
          FilledButton(onPressed: submit, child: Text(t('Valider', 'تأكيد'))),
        ],
      );
    },
  );
}

Future<bool> confirm(BuildContext context, String message, {String? action}) async {
  final r = await showDialog<bool>(
    context: context,
    builder: (ctx) => AlertDialog(
      content: Text(message),
      actions: [
        TextButton(onPressed: () => Navigator.pop(ctx, false), child: Text(t('Annuler', 'إلغاء'))),
        FilledButton(
          style: FilledButton.styleFrom(backgroundColor: negativeColor),
          onPressed: () => Navigator.pop(ctx, true),
          child: Text(action ?? t('Supprimer', 'حذف')),
        ),
      ],
    ),
  );
  return r ?? false;
}

void toast(BuildContext context, String message) {
  ScaffoldMessenger.of(context)
    ..hideCurrentSnackBar()
    ..showSnackBar(SnackBar(content: Text(message)));
}

String get historyHint =>
    t('Historique · appui long pour annuler une ligne', 'السجل · اضغط مطولاً لإلغاء سطر');

/// Ligne d'historique avec suppression par appui long.
class MovementTile extends StatelessWidget {
  const MovementTile({
    super.key,
    required this.title,
    required this.date,
    required this.amount,
    required this.positive,
    this.note,
    this.onDelete,
  });

  final String title;
  final DateTime date;
  final String amount;
  final bool positive;
  final String? note;
  final VoidCallback? onDelete;

  @override
  Widget build(BuildContext context) {
    final c = positive ? positiveColor : negativeColor;
    return ListTile(
      leading: CircleAvatar(
        radius: 18,
        backgroundColor: c.withValues(alpha: 0.1),
        child: Icon(positive ? Icons.arrow_upward : Icons.arrow_downward, size: 18, color: c),
      ),
      title: Text(title),
      subtitle: Text([fmtDateTime(date), if (note != null && note!.isNotEmpty) note].join(' · ')),
      trailing: Text(
        amount,
        style: TextStyle(fontWeight: FontWeight.w600, color: c),
      ),
      onLongPress: onDelete,
    );
  }
}

Color moneyColor(BuildContext context, double v) => v < -0.0001
    ? negativeColor
    : v > 0.0001
    ? positiveColor
    : Theme.of(context).colorScheme.onSurfaceVariant;

/// Carte blanche contenant une liste de lignes séparées.
class ListCard extends StatelessWidget {
  const ListCard({super.key, required this.children});

  final List<Widget> children;

  @override
  Widget build(BuildContext context) => Card(
    clipBehavior: Clip.antiAlias,
    child: Column(
      children: [
        for (var i = 0; i < children.length; i++) ...[
          if (i > 0) const Divider(height: 1, indent: 16, endIndent: 16),
          children[i],
        ],
      ],
    ),
  );
}
