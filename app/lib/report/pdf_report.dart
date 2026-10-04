import 'dart:typed_data';

import 'package:flutter/services.dart' show rootBundle;
import 'package:pdf/pdf.dart';
import 'package:pdf/widgets.dart' as pw;
import 'package:printing/printing.dart';

import '../data/models.dart';
import '../data/repo.dart';
import '../ui/format.dart';

/// Génère le rapport PDF de la situation et ouvre le menu de partage
/// (WhatsApp, e-mail, enregistrer dans Téléchargements…).
Future<void> shareReport({required String shopName}) async {
  final summary = await Repo.instance.summary();
  final bytes = await buildReport(summary, shopName: shopName, date: DateTime.now());
  final d = DateTime.now();
  final name = 'situation_${d.year}-${d.month.toString().padLeft(2, '0')}-'
      '${d.day.toString().padLeft(2, '0')}.pdf';
  await Printing.sharePdf(bytes: bytes, filename: name);
}

final _arabic = RegExp(r'[\u0600-\u06FF\u0750-\u077F]');

/// Texte arabe : sens de lecture de droite à gauche et lettres liées.
pw.Widget _rtl(String text) => pw.Text(text,
    textDirection: pw.TextDirection.rtl, textAlign: pw.TextAlign.left, style: const pw.TextStyle(fontSize: 9));

Future<Uint8List> buildReport(Summary s, {required String shopName, required DateTime date}) async {
  final regular = pw.Font.ttf(await rootBundle.load('assets/fonts/DejaVuSans.ttf'));
  final bold = pw.Font.ttf(await rootBundle.load('assets/fonts/DejaVuSans-Bold.ttf'));
  final doc = pw.Document(
    title: 'Situation $shopName',
    theme: pw.ThemeData.withFont(base: regular, bold: bold),
  );

  const headerColor = PdfColor.fromInt(0xFF1B5E20);
  pw.Widget title(String t) => pw.Padding(
        padding: const pw.EdgeInsets.only(top: 16, bottom: 6),
        child: pw.Text(t,
            style: pw.TextStyle(fontSize: 13, fontWeight: pw.FontWeight.bold, color: headerColor)),
      );

  pw.Widget table(List<String> headers, List<List<String>> rows,
      {Map<int, pw.Alignment>? align, List<String>? total}) {
    return pw.TableHelper.fromTextArray(
      headers: headers,
      data: [
        for (final r in [...rows, ?total]) [for (final c in r) _arabic.hasMatch(c) ? _rtl(c) : c],
      ],
      headerStyle: pw.TextStyle(fontWeight: pw.FontWeight.bold, fontSize: 9, color: PdfColors.white),
      headerDecoration: const pw.BoxDecoration(color: headerColor),
      cellStyle: const pw.TextStyle(fontSize: 9),
      cellAlignments: align ?? {},
      headerAlignments: align ?? {},
      oddRowDecoration: const pw.BoxDecoration(color: PdfColors.grey100),
      border: pw.TableBorder.all(color: PdfColors.grey400, width: 0.5),
      cellPadding: const pw.EdgeInsets.symmetric(horizontal: 4, vertical: 3),
    );
  }

  final products = [...s.products.where((p) => p.qty != 0)]
    ..sort((a, b) {
      final c = (a.category ?? '').toLowerCase().compareTo((b.category ?? '').toLowerCase());
      return c != 0 ? c : a.name.toLowerCase().compareTo(b.name.toLowerCase());
    });

  pw.Widget summaryRow(String label, double value, {bool strong = false, String sign = ''}) =>
      pw.Container(
        padding: const pw.EdgeInsets.symmetric(vertical: 4, horizontal: 6),
        decoration: strong ? const pw.BoxDecoration(color: PdfColors.green50) : null,
        child: pw.Row(children: [
          pw.Expanded(
              child: pw.Text('$sign $label',
                  style: pw.TextStyle(fontWeight: strong ? pw.FontWeight.bold : null))),
          pw.Text(fmtMoney(value),
              style: pw.TextStyle(fontWeight: strong ? pw.FontWeight.bold : null, fontSize: strong ? 13 : 11)),
        ]),
      );

  const right = pw.Alignment.centerRight;

  doc.addPage(
    pw.MultiPage(
      pageFormat: PdfPageFormat.a4,
      margin: const pw.EdgeInsets.all(28),
      header: (ctx) => pw.Row(children: [
        pw.Expanded(
          child: pw.Text(shopName,
              style: const pw.TextStyle(color: PdfColors.grey700),
              textDirection: _arabic.hasMatch(shopName) ? pw.TextDirection.rtl : null),
        ),
        pw.Text('Situation au ${fmtDateTime(date)}', style: const pw.TextStyle(color: PdfColors.grey700)),
      ]),
      footer: (ctx) => pw.Align(
        alignment: pw.Alignment.centerRight,
        child: pw.Text('Page ${ctx.pageNumber} / ${ctx.pagesCount}',
            style: const pw.TextStyle(fontSize: 8, color: PdfColors.grey600)),
      ),
      build: (ctx) => [
        pw.Text('Situation de la boutique',
            style: pw.TextStyle(fontSize: 20, fontWeight: pw.FontWeight.bold)),
        pw.SizedBox(height: 10),
        pw.Container(
          decoration: pw.BoxDecoration(border: pw.Border.all(color: PdfColors.grey400)),
          child: pw.Column(children: [
            summaryRow("Valeur du stock (prix d'achat)", s.stockValue, sign: '+'),
            summaryRow('Argent disponible (caisse + wallets)', s.cash, sign: '+'),
            summaryRow('Ce que les clients nous doivent', s.receivables, sign: '+'),
            summaryRow('Ce que nous devons', s.payables, sign: '−'),
            pw.Divider(height: 1, color: PdfColors.grey400),
            summaryRow('Valeur nette de la boutique', s.netValue, strong: true, sign: '='),
          ]),
        ),
        pw.SizedBox(height: 6),
        pw.Text(
          'Stock au prix de vente : ${fmtMoney(s.stockSaleValue)} · '
          'Marge potentielle : ${fmtMoney(s.potentialMargin)}',
          style: const pw.TextStyle(fontSize: 9, color: PdfColors.grey700),
        ),
        if (s.missingPrice.isNotEmpty)
          pw.Padding(
            padding: const pw.EdgeInsets.only(top: 4),
            child: pw.Text(
              "Attention : ${s.missingPrice.length} produit(s) en stock sans prix d'achat "
              '(valeur du stock sous-estimée) : ${s.missingPrice.map((p) => p.name).join(', ')}',
              style: const pw.TextStyle(fontSize: 9, color: PdfColors.red800),
            ),
          ),
        title('Stock (${products.length} produits)'),
        if (products.isEmpty)
          pw.Text('Aucun produit en stock.')
        else
          table(
            ['Produit', 'Catégorie', 'Quantité', 'Prix achat', 'Valeur', 'Prix vente'],
            [
              for (final p in products)
                [
                  p.name,
                  p.category ?? '',
                  fmtQty(p.qty, p.unitSymbol),
                  p.purchasePrice == null ? '—' : fmtNum(p.purchasePrice!),
                  fmtNum(p.stockValue),
                  p.salePrice == null ? '—' : fmtNum(p.salePrice!),
                ],
            ],
            align: {2: right, 3: right, 4: right, 5: right},
            total: ['TOTAL', '', '', '', fmtNum(s.stockValue), ''],
          ),
        title('Argent disponible'),
        table(
          ['Compte', 'Type', 'Solde (MRU)'],
          [
            for (final a in s.accounts) [a.name, accountKinds[a.kind] ?? a.kind, fmtNum(a.balance)],
          ],
          align: {2: right},
          total: ['TOTAL', '', fmtNum(s.cash)],
        ),
        title('Ce que les clients nous doivent'),
        if (s.debtors.isEmpty)
          pw.Text('Aucune créance.')
        else
          table(
            ['Nom', 'Téléphone', 'Montant (MRU)'],
            [for (final p in s.debtors) [p.name, p.phone ?? '', fmtNum(p.balance)]],
            align: {2: right},
            total: ['TOTAL', '', fmtNum(s.receivables)],
          ),
        title('Ce que nous devons (fournisseurs et autres)'),
        if (s.creditors.isEmpty)
          pw.Text('Aucune dette.')
        else
          table(
            ['Nom', 'Type', 'Téléphone', 'Montant (MRU)'],
            [
              for (final p in s.creditors)
                [p.name, partyKinds[p.kind] ?? p.kind, p.phone ?? '', fmtNum(-p.balance)],
            ],
            align: {3: right},
            total: ['TOTAL', '', '', fmtNum(s.payables)],
          ),
      ],
    ),
  );
  return doc.save();
}
