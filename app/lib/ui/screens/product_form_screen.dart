import 'package:flutter/material.dart';

import '../../data/models.dart';
import '../../data/repo.dart';
import '../../i18n.dart';
import '../format.dart';
import '../widgets/common.dart';
import 'units_screen.dart';

/// Création / modification d'un produit. Pensé pour saisir rapidement
/// tout l'inventaire : bouton « Enregistrer et suivant ».
class ProductFormScreen extends StatefulWidget {
  const ProductFormScreen({super.key, this.product});

  final Product? product;

  @override
  State<ProductFormScreen> createState() => _ProductFormScreenState();
}

class _ProductFormScreenState extends State<ProductFormScreen> {
  final _form = GlobalKey<FormState>();
  final _name = TextEditingController();
  final _category = TextEditingController();
  final _qty = TextEditingController();
  final _purchase = TextEditingController();
  final _sale = TextEditingController();
  final _minStock = TextEditingController();
  final _note = TextEditingController();
  final _nameFocus = FocusNode();
  final _categoryFocus = FocusNode();
  List<Unit> _units = [];
  List<String> _categories = [];
  String? _unitId;
  int _savedCount = 0;

  bool get _editing => widget.product != null;

  @override
  void initState() {
    super.initState();
    final p = widget.product;
    if (p != null) {
      _name.text = p.name;
      _category.text = p.category ?? '';
      _qty.text = numToInput(p.qty);
      _purchase.text = numToInput(p.purchasePrice);
      _sale.text = numToInput(p.salePrice);
      _minStock.text = numToInput(p.minStock);
      _note.text = p.note ?? '';
      _unitId = p.unitId;
    }
    _loadLists();
  }

  Future<void> _loadLists() async {
    final units = await Repo.instance.units();
    final cats = await Repo.instance.categories();
    if (!mounted) return;
    setState(() {
      _units = units;
      _categories = cats;
      if (_unitId == null || !units.any((u) => u.id == _unitId)) {
        _unitId = units.any((u) => u.id == 'u-piece') ? 'u-piece' : units.firstOrNull?.id;
      }
    });
  }

  Unit? get _unit => _units.where((u) => u.id == _unitId).firstOrNull;

  Future<bool> _save() async {
    if (!_form.currentState!.validate()) return false;
    final qty = parseNum(_qty.text);
    final unit = _unit;
    if (unit != null && !unit.allowDecimal && qty != null && qty != qty.roundToDouble()) {
      toast(
        context,
        t(
          "L'unité « ${unit.displayName} » n'accepte pas de décimales",
          'الوحدة « ${unit.displayName} » لا تقبل الكسور',
        ),
      );
      return false;
    }
    await Repo.instance.saveProduct(
      id: widget.product?.id,
      name: _name.text,
      category: _category.text,
      unitId: _unitId!,
      purchasePrice: parseNum(_purchase.text),
      salePrice: parseNum(_sale.text),
      minStock: parseNum(_minStock.text),
      note: _note.text,
      qty: qty ?? (_editing ? null : 0),
    );
    return true;
  }

  Future<void> _saveAndClose() async {
    if (await _save() && mounted) Navigator.pop(context);
  }

  Future<void> _saveAndNext() async {
    if (!await _save() || !mounted) return;
    final cat = _category.text;
    _form.currentState!.reset();
    for (final c in [_name, _qty, _purchase, _sale, _minStock, _note]) {
      c.clear();
    }
    _category.text = cat; // souvent la même catégorie à la suite
    setState(() => _savedCount++);
    _loadLists();
    _nameFocus.requestFocus();
    toast(context, t('Produit enregistré ($_savedCount)', 'تم حفظ المنتج ($_savedCount)'));
  }

  Future<void> _addUnit() async {
    final id = await showAddUnitDialog(context);
    if (id == null) return;
    await _loadLists();
    setState(() => _unitId = id);
  }

  @override
  Widget build(BuildContext context) {
    final sym = _unit?.displaySymbol ?? '';
    final per = sym.isEmpty ? t('unité', 'وحدة') : sym;
    final qty = parseNum(_qty.text) ?? 0;
    final price = parseNum(_purchase.text) ?? 0;
    const gap = SizedBox(height: 14);
    return Scaffold(
      appBar: AppBar(
        title: Text(
          _editing ? t('Modifier le produit', 'تعديل المنتج') : t('Nouveau produit', 'منتج جديد'),
        ),
      ),
      body: Form(
        key: _form,
        child: ListView(
          padding: const EdgeInsets.all(16),
          children: [
            TextFormField(
              controller: _name,
              focusNode: _nameFocus,
              autofocus: !_editing,
              textCapitalization: TextCapitalization.sentences,
              decoration: InputDecoration(labelText: t('Nom du produit *', 'اسم المنتج *')),
              validator: (v) => (v ?? '').trim().isEmpty ? t('Obligatoire', 'إلزامي') : null,
            ),
            gap,
            Autocomplete<String>(
              textEditingController: _category,
              focusNode: _categoryFocus,
              optionsBuilder: (v) => _categories.where(
                (c) => c.toLowerCase().contains(v.text.toLowerCase()) && c != v.text,
              ),
              fieldViewBuilder: (context, ctrl, focus, _) => TextFormField(
                controller: ctrl,
                focusNode: focus,
                textCapitalization: TextCapitalization.sentences,
                decoration: InputDecoration(
                  labelText: t('Catégorie (facultatif)', 'الفئة (اختياري)'),
                  hintText: t('Ex. Alimentation, Boissons', 'مثال: مواد غذائية، مشروبات'),
                ),
              ),
            ),
            gap,
            Row(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Expanded(
                  child: DropdownButtonFormField<String>(
                    initialValue: _unitId,
                    key: ValueKey('unit-$_unitId-${_units.length}'),
                    isExpanded: true,
                    decoration: InputDecoration(labelText: t('Unité de mesure *', 'وحدة القياس *')),
                    items: [
                      for (final u in _units) DropdownMenuItem(value: u.id, child: Text(u.label)),
                    ],
                    onChanged: (v) => setState(() => _unitId = v),
                    validator: (v) => v == null ? t('Obligatoire', 'إلزامي') : null,
                  ),
                ),
                const SizedBox(width: 8),
                IconButton.filledTonal(
                  tooltip: t('Nouvelle unité', 'وحدة جديدة'),
                  onPressed: _addUnit,
                  icon: const Icon(Icons.add),
                  style: IconButton.styleFrom(minimumSize: const Size(52, 52)),
                ),
              ],
            ),
            gap,
            NumberField(
              controller: _qty,
              label: t('Quantité en stock', 'الكمية في المخزون'),
              suffix: sym,
              helper: t('Calcul possible : 3x50+20', 'يمكن الحساب: 3x50+20'),
              onChanged: (_) => setState(() {}),
            ),
            gap,
            Row(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Expanded(
                  child: NumberField(
                    controller: _purchase,
                    label: t("Prix d'achat / $per", 'سعر الشراء / $per'),
                    suffix: currency,
                    onChanged: (_) => setState(() {}),
                  ),
                ),
                const SizedBox(width: 10),
                Expanded(
                  child: NumberField(
                    controller: _sale,
                    label: t('Prix de vente / $per', 'سعر البيع / $per'),
                    suffix: currency,
                  ),
                ),
              ],
            ),
            gap,
            NumberField(
              controller: _minStock,
              label: t("Seuil d'alerte (facultatif)", 'حد التنبيه (اختياري)'),
              suffix: sym,
              helper: t(
                'Alerte quand le stock descend à ce niveau',
                'تنبيه عندما ينزل المخزون إلى هذا الحد',
              ),
            ),
            gap,
            TextFormField(
              controller: _note,
              decoration: InputDecoration(labelText: t('Note (facultatif)', 'ملاحظة (اختياري)')),
            ),
            const SizedBox(height: 16),
            TotalBanner(
              label: t('Valeur de ce stock', 'قيمة هذا المخزون'),
              value: fmtMoney(qty > 0 ? qty * price : 0),
            ),
            const SizedBox(height: 8),
            FilledButton(
              onPressed: _saveAndClose,
              style: FilledButton.styleFrom(minimumSize: const Size.fromHeight(52)),
              child: Text(t('Enregistrer', 'حفظ')),
            ),
            if (!_editing) ...[
              const SizedBox(height: 8),
              OutlinedButton(
                onPressed: _saveAndNext,
                style: OutlinedButton.styleFrom(minimumSize: const Size.fromHeight(52)),
                child: Text(t('Enregistrer et saisir le suivant', 'حفظ وإدخال التالي')),
              ),
            ],
          ],
        ),
      ),
    );
  }
}
