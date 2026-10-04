import 'package:flutter/material.dart';

import '../../data/db.dart';
import '../format.dart';

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
      return Center(child: Text('Erreur : $_error'));
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
  Widget build(BuildContext context) => Center(
        child: Padding(
          padding: const EdgeInsets.all(32),
          child: Column(mainAxisSize: MainAxisSize.min, children: [
            Icon(icon, size: 56, color: Theme.of(context).colorScheme.outline),
            const SizedBox(height: 12),
            Text(text, textAlign: TextAlign.center, style: Theme.of(context).textTheme.bodyLarge),
          ]),
        ),
      );
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
          if (v == null || v.trim().isEmpty) return required ? 'Obligatoire' : null;
          final n = parseNum(v);
          if (n == null) return 'Nombre invalide';
          if (!allowNegative && n < 0) return 'Doit être positif';
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
          child: Column(mainAxisSize: MainAxisSize.min, children: [
            TextFormField(
              controller: ctrl,
              autofocus: true,
              keyboardType: const TextInputType.numberWithOptions(decimal: true),
              decoration: InputDecoration(labelText: label, suffixText: suffix, helperText: helper),
              onFieldSubmitted: (_) => submit(),
              validator: (v) {
                final n = parseNum(v);
                if (n == null) return 'Nombre invalide';
                if (n < 0) return 'Doit être positif';
                if (n == 0 && !allowZero) return 'Doit être supérieur à 0';
                return null;
              },
            ),
            TextFormField(
              controller: note,
              decoration: const InputDecoration(labelText: 'Note (facultatif)'),
            ),
          ]),
        ),
        actions: [
          TextButton(onPressed: () => Navigator.pop(ctx), child: const Text('Annuler')),
          FilledButton(onPressed: submit, child: const Text('Valider')),
        ],
      );
    },
  );
}

Future<bool> confirm(BuildContext context, String message, {String action = 'Supprimer'}) async {
  final r = await showDialog<bool>(
    context: context,
    builder: (ctx) => AlertDialog(
      content: Text(message),
      actions: [
        TextButton(onPressed: () => Navigator.pop(ctx, false), child: const Text('Annuler')),
        FilledButton(onPressed: () => Navigator.pop(ctx, true), child: Text(action)),
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

/// Ligne d'historique avec suppression par appui long.
class MovementTile extends StatelessWidget {
  const MovementTile({
    super.key,
    required this.title,
    required this.date,
    required this.amount,
    this.note,
    this.onDelete,
  });

  final String title;
  final DateTime date;
  final String amount;
  final String? note;
  final VoidCallback? onDelete;

  @override
  Widget build(BuildContext context) => ListTile(
        dense: true,
        title: Text(title),
        subtitle: Text([fmtDateTime(date), if (note != null && note!.isNotEmpty) note].join(' · ')),
        trailing: Text(amount, style: const TextStyle(fontWeight: FontWeight.w600)),
        onLongPress: onDelete,
      );
}

Color moneyColor(BuildContext context, double v) => v < 0
    ? Theme.of(context).colorScheme.error
    : v > 0
        ? Colors.green.shade700
        : Theme.of(context).colorScheme.onSurface;
