import { checkAuth } from '../lib/auth.js';
import { ensureSchema, pullChanges, pushChanges } from '../lib/db.js';

// POST /api/sync  { since: number, changes: { table: [rows] } }
// -> { pushed, cursor, more, changes }
export default async function handler(req, res) {
  if (req.method !== 'POST') return res.status(405).json({ error: 'POST uniquement' });
  if (!checkAuth(req, res)) return;
  try {
    await ensureSchema();
    const body = typeof req.body === 'string' ? JSON.parse(req.body) : req.body ?? {};
    const pushed = await pushChanges(body.changes);
    const since = Number.isFinite(Number(body.since)) ? Number(body.since) : 0;
    const pulled = await pullChanges(since);
    res.status(200).json({ pushed, ...pulled });
  } catch (e) {
    console.error(e);
    res.status(500).json({ error: e.message });
  }
}
