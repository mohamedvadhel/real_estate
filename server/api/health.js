import { checkAuth } from '../lib/auth.js';
import { ensureSchema } from '../lib/db.js';

// GET /api/health : vérifie la clé et la connexion à Neon.
export default async function handler(req, res) {
  if (!checkAuth(req, res)) return;
  try {
    await ensureSchema();
    res.status(200).json({ ok: true });
  } catch (e) {
    res.status(500).json({ ok: false, error: e.message });
  }
}
