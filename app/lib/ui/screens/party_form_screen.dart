import 'package:flutter/material.dart';

import '../../data/models.dart';
import '../../data/repo.dart';
import '../../i18n.dart';
import '../format.dart';
import '../widgets/common.dart';

class PartyFormScreen extends StatefulWidget {
  const PartyFormScreen({super.key, this.party});

  final Party? party;

  @override
  State<PartyFormScreen> createState() => _PartyFormScreenState();
}

class _PartyFormScreenState extends State<PartyFormScreen> {
  final _form = GlobalKey<FormState>();
  final _name = TextEditingController();
  final _phone = TextEditingController();
  final _note = TextEditingController();
  final _amount = TextEditingController();
  String _kind = 'client';
  String _direction = DebtKind.credit.code;

  bool get _editing => widget.party != null;

  @override
  void initState() {
    super.initState();
    final p = widget.party;
    if (p != null) {
      _name.text = p.name;
      _phone.text = p.phone ?? '';
      _note.text = p.note ?? '';
      _kind = p.kind;
    }
  }

  Future<void> _save() async {
    if (!_form.currentState!.validate()) return;
    await Repo.instance.saveParty(
      id: widget.party?.id,
      name: _name.text,
      kind: _kind,
      phone: _phone.text,
      note: _note.text,
      initialKind: _editing ? null : _direction,
      initialAmount: _editing ? null : parseNum(_amount.text),
    );
    if (mounted) Navigator.pop(context);
  }

  @override
  Widget build(BuildContext context) {
    const gap = SizedBox(height: 14);
    return Scaffold(
      appBar: AppBar(
        title: Text(_editing ? t('Modifier', 'تعديل') : t('Nouvelle dette', 'دين جديد')),
      ),
      body: Form(
        key: _form,
        child: ListView(
          padding: const EdgeInsets.all(16),
          children: [
            SegmentedButton<String>(
              segments: [
                for (final e in partyKinds.entries)
                  ButtonSegment(value: e.key, label: Text(e.value)),
              ],
              selected: {_kind},
              onSelectionChanged: (s) => setState(() {
                _kind = s.first;
                // Par défaut : un client nous doit, nous devons à un fournisseur.
                if (!_editing) {
                  _direction = _kind == 'client' ? DebtKind.credit.code : DebtKind.dette.code;
                }
              }),
            ),
            const SizedBox(height: 18),
            TextFormField(
              controller: _name,
              autofocus: !_editing,
              textCapitalization: TextCapitalization.words,
              decoration: InputDecoration(labelText: t('Nom *', 'الاسم *')),
              validator: (v) => (v ?? '').trim().isEmpty ? t('Obligatoire', 'إلزامي') : null,
            ),
            gap,
            TextFormField(
              controller: _phone,
              keyboardType: TextInputType.phone,
              decoration: InputDecoration(
                labelText: t('Téléphone (facultatif)', 'الهاتف (اختياري)'),
              ),
            ),
            gap,
            TextFormField(
              controller: _note,
              decoration: InputDecoration(labelText: t('Note (facultatif)', 'ملاحظة (اختياري)')),
            ),
            if (!_editing) ...[
              const SizedBox(height: 24),
              Text(
                t('Montant actuel de la dette', 'مبلغ الدين الحالي'),
                style: Theme.of(context).textTheme.titleMedium,
              ),
              const SizedBox(height: 10),
              SegmentedButton<String>(
                segments: [
                  ButtonSegment(value: DebtKind.credit.code, label: Text(DebtKind.credit.label)),
                  ButtonSegment(value: DebtKind.dette.code, label: Text(DebtKind.dette.label)),
                ],
                selected: {_direction},
                onSelectionChanged: (s) => setState(() => _direction = s.first),
              ),
              gap,
              NumberField(controller: _amount, label: t('Montant', 'المبلغ'), suffix: currency),
            ],
            const SizedBox(height: 24),
            FilledButton(
              onPressed: _save,
              style: FilledButton.styleFrom(minimumSize: const Size.fromHeight(52)),
              child: Text(t('Enregistrer', 'حفظ')),
            ),
          ],
        ),
      ),
    );
  }
}
