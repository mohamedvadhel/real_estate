// Tables synchronisées entre le téléphone et Neon.
// Toutes ont en plus : id, created_at, updated_at, deleted (et seq côté serveur).
// Types : text | num (montants, quantités) | int (dates en ms, entiers)
export const TABLES = {
  units: { name: 'text', symbol: 'text', allow_decimal: 'int' },
  products: {
    name: 'text', category: 'text', unit_id: 'text',
    purchase_price: 'num', sale_price: 'num', min_stock: 'num', note: 'text',
  },
  stock_movements: {
    product_id: 'text', qty: 'num', kind: 'text', unit_cost: 'num', date: 'int', note: 'text',
  },
  parties: { name: 'text', kind: 'text', phone: 'text', note: 'text' },
  debt_entries: { party_id: 'text', amount: 'num', kind: 'text', date: 'int', note: 'text' },
  accounts: { name: 'text', kind: 'text', position: 'int' },
  account_movements: { account_id: 'text', amount: 'num', kind: 'text', date: 'int', note: 'text' },
};

const PG_TYPES = { text: 'text', num: 'numeric(18,3)', int: 'bigint' };
const BASE_COLUMNS = { id: 'text', created_at: 'int', updated_at: 'int', deleted: 'int' };

export function columnsOf(table) {
  return { ...BASE_COLUMNS, ...TABLES[table] };
}

export function schemaStatements() {
  const stmts = ['CREATE SEQUENCE IF NOT EXISTS sync_seq'];
  for (const [table, cols] of Object.entries(TABLES)) {
    const defs = Object.entries(cols).map(([c, t]) => `${c} ${PG_TYPES[t]}`);
    stmts.push(
      `CREATE TABLE IF NOT EXISTS ${table} (` +
        'id text PRIMARY KEY, created_at bigint NOT NULL, updated_at bigint NOT NULL, ' +
        'deleted smallint NOT NULL DEFAULT 0, seq bigint NOT NULL, ' +
        defs.join(', ') +
        ')',
    );
    stmts.push(`CREATE INDEX IF NOT EXISTS ${table}_seq_idx ON ${table} (seq)`);
  }
  return stmts;
}

function coerce(value, type) {
  if (value === null || value === undefined || value === '') return null;
  if (type === 'text') return String(value);
  const n = Number(value);
  if (!Number.isFinite(n)) return null;
  return type === 'int' ? Math.trunc(n) : n;
}

// Nettoie une ligne reçue du téléphone : garde seulement les colonnes connues.
export function cleanRow(table, row) {
  if (!row || typeof row.id !== 'string' || !row.id || row.id.length > 64) return null;
  const out = {};
  for (const [c, t] of Object.entries(columnsOf(table))) out[c] = coerce(row[c], t);
  out.created_at ??= 0;
  out.updated_at ??= 0;
  out.deleted = out.deleted ? 1 : 0;
  return out;
}

// Convertit une ligne lue dans Postgres (numeric/bigint arrivent en texte).
export function parseRow(table, row) {
  const out = {};
  for (const [c, t] of Object.entries(columnsOf(table))) {
    out[c] = t === 'text' ? row[c] : row[c] === null ? null : Number(row[c]);
  }
  return out;
}

// Construit un INSERT ... ON CONFLICT multi-lignes ; la version la plus récente gagne.
export function upsertQuery(table, rows) {
  const cols = Object.keys(columnsOf(table));
  const params = [];
  const values = rows.map((row) => {
    const ph = cols.map((c) => {
      params.push(row[c]);
      return `$${params.length}`;
    });
    return `(${ph.join(', ')}, nextval('sync_seq'))`;
  });
  const updates = [...cols.filter((c) => c !== 'id'), 'seq'].map((c) => `${c} = EXCLUDED.${c}`);
  const text =
    `INSERT INTO ${table} (${cols.join(', ')}, seq) VALUES ${values.join(', ')} ` +
    `ON CONFLICT (id) DO UPDATE SET ${updates.join(', ')} ` +
    `WHERE ${table}.updated_at <= EXCLUDED.updated_at`;
  return { text, params };
}
