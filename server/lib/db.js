import { neon } from '@neondatabase/serverless';
import { TABLES, cleanRow, parseRow, schemaStatements, upsertQuery } from './tables.js';

const CHUNK = 300;
export const PULL_LIMIT = 2000;

let sqlClient;
let schemaReady;

// Pour les tests locaux : remplace le client Neon par un autre (même interface).
export function setSql(client) {
  sqlClient = client;
  schemaReady = undefined;
}

export function getSql() {
  if (sqlClient) return sqlClient;
  if (!process.env.DATABASE_URL) throw new Error('DATABASE_URL manquant');
  return (sqlClient = neon(process.env.DATABASE_URL));
}

// Crée les tables au premier appel (aucune migration manuelle nécessaire).
export function ensureSchema() {
  schemaReady ??= (async () => {
    const sql = getSql();
    await sql.transaction(schemaStatements().map((s) => sql.query(s)));
  })().catch((e) => {
    schemaReady = undefined;
    throw e;
  });
  return schemaReady;
}

export async function pushChanges(changes) {
  const sql = getSql();
  let count = 0;
  const queries = [];
  for (const table of Object.keys(TABLES)) {
    const rows = (Array.isArray(changes?.[table]) ? changes[table] : [])
      .map((r) => cleanRow(table, r))
      .filter(Boolean);
    for (let i = 0; i < rows.length; i += CHUNK) {
      const { text, params } = upsertQuery(table, rows.slice(i, i + CHUNK));
      queries.push(sql.query(text, params));
    }
    count += rows.length;
  }
  if (queries.length) await sql.transaction(queries);
  return count;
}

export async function pullChanges(since) {
  const sql = getSql();
  const tables = Object.keys(TABLES);
  const results = await sql.transaction(
    tables.map((t) =>
      sql.query(`SELECT * FROM ${t} WHERE seq > $1 ORDER BY seq LIMIT ${PULL_LIMIT}`, [since]),
    ),
  );
  const changes = {};
  let cursor = since;
  let truncatedCursor = null;
  tables.forEach((t, i) => {
    const rows = results[i];
    changes[t] = rows.map((r) => parseRow(t, r));
    if (!rows.length) return;
    const last = Number(rows[rows.length - 1].seq);
    cursor = Math.max(cursor, last);
    if (rows.length === PULL_LIMIT) truncatedCursor = Math.min(truncatedCursor ?? last, last);
  });
  // Si une table est tronquée, on repart de là au prochain appel (les doublons sont sans effet).
  return { changes, cursor: truncatedCursor ?? cursor, more: truncatedCursor !== null };
}
