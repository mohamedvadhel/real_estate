import 'dart:typed_data';

import 'package:flutter/services.dart' show rootBundle;
import 'package:pdf/pdf.dart';
import 'package:pdf/widgets.dart' as pw;
import 'package:printing/printing.dart';

import '../data/models.dart';
import '../data/repo.dart';
import '../i18n.dart';
import '../ui/format.dart';

/// Génère le rapport PDF de la situation (dans la langue de l'application)
/// et ouvre le menu de partage (WhatsApp, e-mail, Téléchargements…).
Future<void> shareReport({required String shopName}) async {
  final summary = await Repo.instance.summary();
  final bytes = await buildReport(summary, shopName: shopName, date: DateTime.now());
  final d = DateTime.now();
  final name =
      'situation_${d.year}-${d.month.toString().padLeft(2, '0')}-'
      '${d.day.toString().padLeft(2, '0')}.pdf';
  await Printing.sharePdf(bytes: bytes, filename: name);
}

final _arabic = RegExp(r'[؀-ۿݐ-ݿ]');

/// Sens d'écriture d'un texte : RTL s'il contient de l'arabe.
pw.TextDirection _dirOf(String text) =>
    _arabic.hasMatch(text) ? pw.TextDirection.rtl : pw.TextDirection.ltr;

Future<Uint8List> buildReport(Summary s, {required String shopName, required DateTime date}) async {
  final regular = pw.Font.ttf(await rootBundle.load('assets/fonts/DejaVuSans.ttf'));
  final bold = pw.Font.ttf(await rootBundle.load('assets/fonts/DejaVuSans-Bold.ttf'));
  final doc = pw.Document(
    title: '${t('Situation', 'الوضعية')} $shopName',
    theme: pw.ThemeData.withFont(base: regular, bold: bold),
  );

  const brand = PdfColor.fromInt(0xFF0E7C66);
  const muted = PdfColors.grey700;
  final end = isAr ? pw.Alignment.centerLeft : pw.Alignment.centerRight;

  /// Texte qui respecte le sens de chaque langue (noms arabes dans un rapport français et inversement).
  pw.Widget txt(String text, {pw.TextStyle? style}) =>
      pw.Text(text, style: style, textDirection: _dirOf(text));

  pw.Widget title(String text) => pw.Padding(
    padding: const pw.EdgeInsets.only(top: 18, bottom: 6),
    child: txt(
      text,
      style: pw.TextStyle(fontSize: 13, fontWeight: pw.FontWeight.bold, color: brand),
    ),
  );

  pw.Widget table(
    List<String> headers,
    List<List<String>> rows, {
    Set<int> numeric = const {},
    List<String>? total,
  }) {
    // En arabe, la première colonne est à droite : on inverse l'ordre des colonnes.
    final n = headers.length;
    List<T> order<T>(List<T> l) => isAr ? l.reversed.toList() : l;
    headers = order(headers);
    rows = [for (final r in rows) order(r)];
    total = total == null ? null : order(total);
    final start = isAr ? pw.Alignment.centerRight : pw.Alignment.centerLeft;
    final align = {for (final i in numeric) isAr ? n - 1 - i : i: end};
    return pw.TableHelper.fromTextArray(
      cellAlignment: start,
      headerAlignment: start,
      headers: [
        for (final h in headers)
          txt(
            h,
            style: pw.TextStyle(
              fontWeight: pw.FontWeight.bold,
              fontSize: 9,
              color: PdfColors.white,
            ),
          ),
      ],
      data: [
        for (final r in rows) [for (final c in r) txt(c, style: const pw.TextStyle(fontSize: 9))],
        if (total != null)
          [
            for (final c in total)
              txt(c, style: pw.TextStyle(fontSize: 9, fontWeight: pw.FontWeight.bold)),
          ],
      ],
      headerDecoration: const pw.BoxDecoration(color: brand),
      cellAlignments: align,
      headerAlignments: align,
      oddRowDecoration: const pw.BoxDecoration(color: PdfColors.grey100),
      border: pw.TableBorder.all(color: PdfColors.grey300, width: 0.5),
      cellPadding: const pw.EdgeInsets.symmetric(horizontal: 5, vertical: 4),
    );
  }

  final products = [...s.products.where((p) => p.qty != 0)]
    ..sort((a, b) {
      final c = (a.category ?? '').toLowerCase().compareTo((b.category ?? '').toLowerCase());
      return c != 0 ? c : a.name.toLowerCase().compareTo(b.name.toLowerCase());
    });

  pw.Widget summaryRow(String label, double value, {bool strong = false, String sign = ''}) =>
      pw.Container(
        padding: const pw.EdgeInsets.symmetric(vertical: 6, horizontal: 10),
        decoration: strong ? const pw.BoxDecoration(color: PdfColor.fromInt(0xFFE6F4F0)) : null,
        child: pw.Row(
          children: [
            pw.Expanded(
              child: txt(
                '$sign $label',
                style: pw.TextStyle(fontWeight: strong ? pw.FontWeight.bold : null),
              ),
            ),
            txt(
              fmtMoney(value),
              style: pw.TextStyle(
                fontWeight: strong ? pw.FontWeight.bold : null,
                fontSize: strong ? 14 : 11,
                color: strong ? brand : null,
              ),
            ),
          ],
        ),
      );

  final noPrice = '—';

  doc.addPage(
    pw.MultiPage(
      pageFormat: PdfPageFormat.a4,
      margin: const pw.EdgeInsets.all(28),
      textDirection: isAr ? pw.TextDirection.rtl : pw.TextDirection.ltr,
      header: (ctx) => pw.Container(
        padding: const pw.EdgeInsets.only(bottom: 6),
        margin: const pw.EdgeInsets.only(bottom: 10),
        decoration: const pw.BoxDecoration(
          border: pw.Border(bottom: pw.BorderSide(color: PdfColors.grey300, width: 0.5)),
        ),
        child: pw.Row(
          children: [
            pw.Expanded(
              child: txt(shopName, style: const pw.TextStyle(color: muted)),
            ),
            txt(
              '${t('Situation au', 'الوضعية بتاريخ')} ${fmtDateTime(date)}',
              style: const pw.TextStyle(color: muted),
            ),
          ],
        ),
      ),
      footer: (ctx) => pw.Align(
        alignment: end,
        child: pw.Text(
          '${ctx.pageNumber} / ${ctx.pagesCount}',
          style: const pw.TextStyle(fontSize: 8, color: PdfColors.grey600),
        ),
      ),
      build: (ctx) => [
        txt(
          t('Situation de la boutique', 'وضعية المتجر'),
          style: pw.TextStyle(fontSize: 22, fontWeight: pw.FontWeight.bold),
        ),
        pw.SizedBox(height: 12),
        pw.Container(
          decoration: pw.BoxDecoration(
            border: pw.Border.all(color: PdfColors.grey300),
            borderRadius: pw.BorderRadius.circular(6),
          ),
          child: pw.Column(
            children: [
              summaryRow(
                t("Valeur du stock (prix d'achat)", 'قيمة المخزون (سعر الشراء)'),
                s.stockValue,
                sign: '+',
              ),
              summaryRow(
                t('Argent disponible (caisse + wallets)', 'المال المتوفر (الصندوق + المحافظ)'),
                s.cash,
                sign: '+',
              ),
              summaryRow(
                t('Ce que les clients nous doivent', 'ما لنا عند الزبائن'),
                s.receivables,
                sign: '+',
              ),
              summaryRow(t('Ce que nous devons', 'ما علينا'), s.payables, sign: '−'),
              pw.Divider(height: 1, color: PdfColors.grey300),
              summaryRow(
                t('Valeur nette de la boutique', 'القيمة الصافية للمتجر'),
                s.netValue,
                strong: true,
                sign: '=',
              ),
            ],
          ),
        ),
        pw.SizedBox(height: 6),
        txt(
          '${t('Stock au prix de vente', 'المخزون بسعر البيع')} : ${fmtMoney(s.stockSaleValue)} · '
          '${t('Marge potentielle', 'الربح المتوقع')} : ${fmtMoney(s.potentialMargin)}',
          style: const pw.TextStyle(fontSize: 9, color: muted),
        ),
        if (s.missingPrice.isNotEmpty)
          pw.Padding(
            padding: const pw.EdgeInsets.only(top: 4),
            child: txt(
              '${t("Attention : produit(s) en stock sans prix d'achat (valeur sous-estimée)", 'تنبيه: منتجات بدون سعر شراء (القيمة أقل من الحقيقة)')} : '
              '${s.missingPrice.map((p) => p.name).join('، ')}',
              style: const pw.TextStyle(fontSize: 9, color: PdfColors.red800),
            ),
          ),
        title(t('Stock (${products.length} produits)', 'المخزون (${products.length} منتج)')),
        if (products.isEmpty)
          txt(t('Aucun produit en stock.', 'لا توجد منتجات في المخزون.'))
        else
          table(
            [
              t('Produit', 'المنتج'),
              t('Catégorie', 'الفئة'),
              t('Quantité', 'الكمية'),
              t('Prix achat', 'سعر الشراء'),
              t('Valeur', 'القيمة'),
              t('Prix vente', 'سعر البيع'),
            ],
            [
              for (final p in products)
                [
                  p.name,
                  p.category ?? '',
                  fmtQty(p.qty, p.unit),
                  p.purchasePrice == null ? noPrice : fmtNum(p.purchasePrice!),
                  fmtNum(p.stockValue),
                  p.salePrice == null ? noPrice : fmtNum(p.salePrice!),
                ],
            ],
            numeric: {2, 3, 4, 5},
            total: [t('TOTAL', 'المجموع'), '', '', '', fmtNum(s.stockValue), ''],
          ),
        title(t('Argent disponible', 'المال المتوفر')),
        table(
          [t('Compte', 'الحساب'), t('Type', 'النوع'), '${t('Solde', 'الرصيد')} ($currency)'],
          [
            for (final a in s.accounts)
              [a.displayName, accountKinds[a.kind] ?? a.kind, fmtNum(a.balance)],
          ],
          numeric: {2},
          total: [t('TOTAL', 'المجموع'), '', fmtNum(s.cash)],
        ),
        title(t('Ce que les clients nous doivent', 'ما لنا عند الزبائن')),
        if (s.debtors.isEmpty)
          txt(t('Aucune créance.', 'لا توجد ديون لنا.'))
        else
          table(
            [t('Nom', 'الاسم'), t('Téléphone', 'الهاتف'), '${t('Montant', 'المبلغ')} ($currency)'],
            [
              for (final p in s.debtors) [p.name, fmtPhone(p.phone ?? ''), fmtNum(p.balance)],
            ],
            numeric: {2},
            total: [t('TOTAL', 'المجموع'), '', fmtNum(s.receivables)],
          ),
        title(t('Ce que nous devons (fournisseurs et autres)', 'ما علينا (الموردون وغيرهم)')),
        if (s.creditors.isEmpty)
          txt(t('Aucune dette.', 'لا توجد ديون علينا.'))
        else
          table(
            [
              t('Nom', 'الاسم'),
              t('Type', 'النوع'),
              t('Téléphone', 'الهاتف'),
              '${t('Montant', 'المبلغ')} ($currency)',
            ],
            [
              for (final p in s.creditors)
                [p.name, partyKinds[p.kind] ?? p.kind, fmtPhone(p.phone ?? ''), fmtNum(-p.balance)],
            ],
            numeric: {3},
            total: [t('TOTAL', 'المجموع'), '', '', fmtNum(s.payables)],
          ),
      ],
    ),
  );
  return doc.save();
}
