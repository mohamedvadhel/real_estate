// Serveur local pour tester l'app sans Vercel ni Neon, avec un PostgreSQL classique :
//   DATABASE_URL=postgres://... APP_KEY=test node scripts/local-server.mjs
import http from 'node:http';
import pg from 'pg';
import { setSql } from '../lib/db.js';
import health from '../api/health.js';
import sync from '../api/sync.js';

const pool = new pg.Pool({ connectionString: process.env.DATABASE_URL });

// Imite l'interface du client Neon : sql.query(text, params) et sql.transaction([...]).
const query = (text, params) => ({
  text,
  params,
  then: (ok, ko) => pool.query(text, params).then((r) => r.rows).then(ok, ko),
});
setSql({
  query,
  async transaction(queries) {
    const client = await pool.connect();
    try {
      await client.query('BEGIN');
      const out = [];
      for (const q of queries) out.push((await client.query(q.text, q.params)).rows);
      await client.query('COMMIT');
      return out;
    } catch (e) {
      await client.query('ROLLBACK');
      throw e;
    } finally {
      client.release();
    }
  },
});

const routes = { '/api/sync': sync, '/api/health': health };

http
  .createServer(async (req, res) => {
    let raw = '';
    for await (const chunk of req) raw += chunk;
    req.body = raw ? JSON.parse(raw) : {};
    res.status = (code) => ((res.statusCode = code), res);
    res.json = (obj) => {
      res.setHeader('Content-Type', 'application/json');
      res.end(JSON.stringify(obj));
    };
    const handler = routes[new URL(req.url, 'http://x').pathname];
    if (!handler) return res.status(404).json({ error: 'not found' });
    await handler(req, res);
  })
  .listen(Number(process.env.PORT ?? 3000), () =>
    console.log(`Serveur local sur http://localhost:${process.env.PORT ?? 3000}`),
  );
