export function checkAuth(req, res) {
  const expected = process.env.APP_KEY;
  if (!expected) {
    res.status(500).json({ error: 'APP_KEY non configuré sur le serveur' });
    return false;
  }
  if (req.headers['x-app-key'] !== expected) {
    res.status(401).json({ error: "Clé d'accès invalide" });
    return false;
  }
  return true;
}
