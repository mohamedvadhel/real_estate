# Compte Boutique

Application Android pour **évaluer la situation d'une boutique** : valeur du stock, argent disponible
(caisse + wallets), dettes des clients et dettes envers les fournisseurs. Elle fonctionne **sans internet**,
se synchronise avec une base PostgreSQL sur **Neon** quand le réseau est disponible, et génère un **rapport PDF**.

```
Valeur nette = Stock (prix d'achat) + Caisse et wallets + Ce que les clients me doivent − Ce que je dois
```

## Contenu du dépôt

| Dossier | Rôle |
|---|---|
| `app/` | Application mobile Flutter (Android) avec base SQLite locale |
| `server/` | API de synchronisation (fonctions Vercel) branchée sur Neon |
| `.github/workflows/android-apk.yml` | Construit l'APK automatiquement à chaque push |

## Fonctionnalités (version 1)

- **Stock** : produits avec unité de mesure au choix (pièce, kg, litre, mètre, sac, carton, bidon…
  et vos propres unités), quantités décimales, prix d'achat et de vente, catégorie, seuil d'alerte.
  - Saisie rapide de l'inventaire : « Enregistrer et saisir le suivant ».
  - Calcul dans les champs de quantité : `3x50+20` = 3 sacs de 50 kg + 20 kg en vrac.
  - Inventaire (corriger la quantité comptée), entrées, sorties, pertes, avec historique.
- **Dettes** : clients, fournisseurs ou autres. Il me doit / il m'a payé / je lui dois / je l'ai payé,
  avec paiements partiels et historique.
- **Caisse et wallets** : espèces, Bankily, Masrvi, Sedad (vous pouvez en ajouter : Click, BimBank…).
  « Saisir le solde réel » ajuste le compte au montant compté.
- **Situation** : valeur nette, détail de chaque poste, marge potentielle, alertes (produits sans prix
  d'achat, stock bas).
- **Rapport PDF** partageable (WhatsApp, e-mail, Téléchargements…). Les noms en arabe sont gérés.
- **Hors ligne** : tout est enregistré sur le téléphone. Synchronisation automatique avec Neon dès
  qu'une connexion est disponible.

Rien n'est jamais écrasé : le stock et les soldes sont la somme des mouvements. Une erreur s'annule
par un appui long sur la ligne dans l'historique.

## Mise en route

### 1. API sur Vercel + Neon

1. Sur [vercel.com](https://vercel.com) : **Add New → Project**, importer ce dépôt GitHub.
2. **Root Directory** : `server`. Framework : *Other*.
3. **Environment Variables** :
   - `DATABASE_URL` : la chaîne de connexion Neon (`postgresql://…neon.tech/…?sslmode=require`).
   - `APP_KEY` : un mot de passe long de votre choix. Il sera demandé dans l'application.
4. **Deploy**. Les tables sont créées automatiquement au premier appel.
   Pour vérifier, ouvrez `https://<votre-projet>.vercel.app/api/health` : la réponse doit être
   « Clé d'accès invalide », ce qui veut dire que le serveur répond.

### 2. Installer l'application

1. Onglet **Actions** du dépôt GitHub : le workflow « APK Android » construit l'APK à chaque push.
2. Onglet **Releases** : téléchargez `compte-boutique.apk` depuis le téléphone et installez-le
   (autorisez « sources inconnues »).
3. Dans l'app : **Situation → ⚙ Réglages** : adresse du serveur Vercel et `APP_KEY`, puis
   « Enregistrer et synchroniser ».

**Clé de signature (recommandé)** : ajoutez les secrets GitHub `ANDROID_KEYSTORE_BASE64` et
`ANDROID_KEYSTORE_PASSWORD` (Settings → Secrets and variables → Actions). Sans eux, chaque APK est
signé avec une clé différente, et il faut désinstaller l'ancienne version avant d'installer la
nouvelle. Les données reviennent par la synchronisation. Pour créer une clé :

```bash
keytool -genkeypair -keystore release.jks -alias boutique -keyalg RSA -keysize 2048 -validity 10000
base64 -w0 release.jks   # → ANDROID_KEYSTORE_BASE64
```

## Développement

```bash
# Application
cd app && flutter pub get && flutter test test/app_test.dart && flutter run

# API : tests unitaires (+ test d'intégration si TEST_DATABASE_URL pointe vers un PostgreSQL)
cd server && npm install && npm test

# API locale sans Vercel, sur un PostgreSQL classique
DATABASE_URL=postgres://... APP_KEY=secret PORT=3999 node scripts/local-server.mjs
# Test de synchronisation de bout en bout (deux « téléphones »)
cd app && SYNC_TEST_URL=http://localhost:3999 flutter test test/sync_e2e_test.dart
```

### Modèle de données

Chaque table a `id` (UUID créé sur le téléphone), `created_at`, `updated_at`, `deleted` (suppression
logique). En cas de conflit, la modification la plus récente gagne.

| Table | Contenu |
|---|---|
| `units` | Unités de mesure (nom, abréviation, décimales autorisées) |
| `products` | Produits : nom, catégorie, unité, prix d'achat, prix de vente, seuil d'alerte |
| `stock_movements` | Mouvements de stock : initial, inventaire, entrée, sortie, perte (quantité signée) |
| `parties` | Clients, fournisseurs, autres |
| `debt_entries` | Opérations de dette (montant signé : + il me doit, − je lui dois) |
| `accounts` | Caisse, wallets, banque |
| `account_movements` | Mouvements d'argent (ajustement du solde, entrée, sortie) |

### Prochaines étapes prévues

- Ventes et achats détaillés (mettent à jour le stock, la caisse et les dettes en une seule saisie).
- Dépenses (loyer, électricité, transport…) et calcul du bénéfice par période.
- Conditionnements par produit (ex. sac de 50 kg vendu au kg) avec prix différents.
- Export Excel, interface en arabe, code PIN.
