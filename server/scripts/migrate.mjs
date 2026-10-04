// Crée les tables dans Neon : DATABASE_URL=... npm run migrate
import { ensureSchema } from '../lib/db.js';

await ensureSchema();
console.log('Schéma créé / à jour.');
