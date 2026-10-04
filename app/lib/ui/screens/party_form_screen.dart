import 'package:flutter/material.dart';

import '../../data/models.dart';
import '../../data/repo.dart';
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
    return Scaffold(
      appBar: AppBar(title: Text(_editing ? 'Modifier' : 'Nouvelle personne')),
      body: Form(
        key: _form,
        child: ListView(
          padding: const EdgeInsets.all(16),
          children: [
            SegmentedButton<String>(
              segments: [
                for (final e in partyKinds.entries) ButtonSegment(value: e.key, label: Text(e.value)),
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
            const SizedBox(height: 16),
            TextFormField(
              controller: _name,
              autofocus: !_editing,
              textCapitalization: TextCapitalization.words,
              decoration: const InputDecoration(labelText: 'Nom *'),
              validator: (v) => (v ?? '').trim().isEmpty ? 'Obligatoire' : null,
            ),
            const SizedBox(height: 12),
            TextFormField(
              controller: _phone,
              keyboardType: TextInputType.phone,
              decoration: const InputDecoration(labelText: 'Téléphone (facultatif)'),
            ),
            const SizedBox(height: 12),
            TextFormField(
              controller: _note,
              decoration: const InputDecoration(labelText: 'Note (facultatif)'),
            ),
            if (!_editing) ...[
              const SizedBox(height: 24),
              Text('Montant actuel de la dette', style: Theme.of(context).textTheme.titleMedium),
              const SizedBox(height: 8),
              SegmentedButton<String>(
                segments: [
                  ButtonSegment(value: DebtKind.credit.code, label: const Text('Il me doit')),
                  ButtonSegment(value: DebtKind.dette.code, label: const Text('Je lui dois')),
                ],
                selected: {_direction},
                onSelectionChanged: (s) => setState(() => _direction = s.first),
              ),
              const SizedBox(height: 12),
              NumberField(controller: _amount, label: 'Montant', suffix: 'MRU'),
            ],
            const SizedBox(height: 24),
            FilledButton(
              onPressed: _save,
              style: FilledButton.styleFrom(minimumSize: const Size.fromHeight(48)),
              child: const Text('Enregistrer'),
            ),
          ],
        ),
      ),
    );
  }
}
