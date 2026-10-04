import assert from 'node:assert/strict';
import { test } from 'node:test';
import { cleanRow, parseRow, schemaStatements, upsertQuery } from '../lib/tables.js';

test('cleanRow garde seulement les colonnes connues et convertit les types', () => {
  const row = cleanRow('products', {
    id: 'p1', created_at: '10', updated_at: 20, deleted: false,
    name: 'Riz', purchase_price: '450.5', unit_id: 'u-kg', hack: 'DROP TABLE',
  });
  assert.equal(row.hack, undefined);
  assert.equal(row.purchase_price, 450.5);
  assert.equal(row.created_at, 10);
  assert.equal(row.deleted, 0);
  assert.equal(row.sale_price, null);
});

test('cleanRow rejette une ligne sans id', () => {
  assert.equal(cleanRow('units', { name: 'kg' }), null);
});

test('upsertQuery numérote les paramètres et applique la règle « le plus récent gagne »', () => {
  const rows = [cleanRow('units', { id: 'a', name: 'kg' }), cleanRow('units', { id: 'b', name: 'L' })];
  const { text, params } = upsertQuery('units', rows);
  assert.equal(params.length, 14);
  assert.match(text, /\$14, nextval/);
  assert.match(text, /WHERE units\.updated_at <= EXCLUDED\.updated_at/);
});

test('parseRow convertit numeric et bigint en nombres', () => {
  const r = parseRow('debt_entries', { id: 'd', created_at: '1', updated_at: '2', deleted: 0, amount: '1500.000', party_id: 'p', kind: 'credit', date: '3', note: null });
  assert.equal(r.amount, 1500);
  assert.equal(r.date, 3);
});

test('schéma : une table par entité', () => {
  assert.equal(schemaStatements().filter((s) => s.startsWith('CREATE TABLE')).length, 7);
});
