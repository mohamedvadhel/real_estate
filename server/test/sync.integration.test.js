// Test d'intégration avec un vrai PostgreSQL : TEST_DATABASE_URL=postgres://... npm test
import assert from 'node:assert/strict';
import { test } from 'node:test';
import pg from 'pg';
import { pullChanges, pushChanges, setSql } from '../lib/db.js';
import { ensureSchema } from '../lib/db.js';

const url = process.env.TEST_DATABASE_URL;

// Garde-fou : ce test VIDE la base. Il ne tourne que sur un PostgreSQL local, jamais sur Neon / la prod.
if (url && (!/localhost|127\.0\.0\.1|%2F|@\//.test(url) || /neon\.tech/.test(url) || url === process.env.DATABASE_URL)) {
  throw new Error('TEST_DATABASE_URL doit pointer vers un PostgreSQL local (jamais la base de production).');
}

test('push / pull avec « le plus récent gagne »', { skip: !url }, async () => {
  const pool = new pg.Pool({ connectionString: url });
  await pool.query('DROP SCHEMA public CASCADE; CREATE SCHEMA public');
  const query = (text, params) => ({
    text, params,
    then: (ok, ko) => pool.query(text, params).then((r) => r.rows).then(ok, ko),
  });
  setSql({
    query,
    async transaction(qs) {
      const out = [];
      for (const q of qs) out.push((await pool.query(q.text, q.params)).rows);
      return out;
    },
  });
  await ensureSchema();
  await ensureSchema(); // idempotent

  const base = { created_at: 1, deleted: 0 };
  await pushChanges({
    products: [{ ...base, id: 'p1', updated_at: 10, name: 'Riz', purchase_price: 30.5, unit_id: 'u-kg' }],
    debt_entries: [{ ...base, id: 'd1', updated_at: 10, party_id: 'x', amount: -1500, kind: 'dette', date: 5 }],
  });
  let r = await pullChanges(0);
  assert.equal(r.changes.products[0].purchase_price, 30.5);
  assert.equal(r.changes.debt_entries[0].amount, -1500);
  const cursor = r.cursor;

  // Une version plus ancienne n'écrase pas la plus récente
  await pushChanges({ products: [{ ...base, id: 'p1', updated_at: 5, name: 'Ancien' }] });
  r = await pullChanges(0);
  assert.equal(r.changes.products[0].name, 'Riz');
  assert.equal((await pullChanges(cursor)).changes.products.length, 0);

  // Une version plus récente remplace et réapparaît après le curseur
  await pushChanges({ products: [{ ...base, id: 'p1', updated_at: 20, name: 'Riz parfumé', deleted: 1 }] });
  r = await pullChanges(cursor);
  assert.equal(r.changes.products[0].name, 'Riz parfumé');
  assert.equal(r.changes.products[0].deleted, 1);
  assert.equal(r.more, false);
  await pool.end();
});
