# Quantlab — design du fonds quantitatif personnel

Date : 2026-10-13
Statut : validé.

## 1. Objectifs et non-objectifs

**Objectifs, par ordre de réalisme :**
1. Apprendre la méthode quant (données, backtest, validation, gestion du risque).
2. Produire un projet de qualité professionnelle, publiable sur GitHub, défendable en entretien.
3. Trouver une ou plusieurs stratégies à faible capacité susceptibles de rapporter un peu d'argent sur compte personnel.

**Non-objectifs :**
- Intraday et haute fréquence.
- Données Twitter/X et imagerie satellite (coût, signal arbitré).
- Gestion d'argent de tiers. Ce n'est pas un fonds au sens légal, c'est un compte personnel.
- Le système ne passe jamais d'ordres réels automatiquement : il produit la recherche, le backtest, le paper trading et les signaux ; l'exécution est manuelle et décidée par le gérant.

**Contraintes :**
- Univers : ETF et actions US. Broker : Interactive Brokers (IBKR), compte paper trading d'abord.
- Capital réel non décidé : le système est paramétrable en capital et montre ce qui est jouable à chaque niveau.
- Données gratuites au départ (yfinance, Tiingo). La couche données doit permettre de brancher une source payante (EODHD, Sharadar) sans réécriture.
- Machine : Mac M4, 16 Go. Python 3.12 via `uv`.

## 2. Architecture

Un package `quantlab`, en couches qui ne dépendent que des couches inférieures :

```
quantlab/
  data/        sources, stockage Parquet, univers, calendrier
  features/    calculs dérivés des prix
  strategies/  prix jusqu'à t -> poids cibles à t
  backtest/    poids cibles -> positions -> P&L net de frais
  validation/  walk-forward, Sharpe déflaté, bootstrap, sensibilité
  portfolio/   contraintes réelles : capital, lots entiers, frais minimums
  reports/     tearsheets HTML, comparaison de runs
  research/    registre d'expériences
  execution/   phase 3 : ordres et paper trading IBKR
tests/         pytest, dont la parité avec vectorbt
notebooks/     exploration uniquement, aucune logique métier
docs/          specs, journal de recherche, notes de cours
data/          Parquet locaux (gitignoré)
```

**Stack :** pandas, numpy, pyarrow, duckdb, statsmodels, scikit-learn, quantstats, matplotlib/plotly, pyyaml, pydantic (configs). vectorbt uniquement en dépendance de test. ruff pour le lint, pytest pour les tests.

**Principes :**
- Données immuables : un Parquet par symbole et par source, complété mais jamais réécrit sur le passé.
- Une stratégie ne voit jamais le futur. Le moteur applique le décalage d'exécution.
- Tout backtest est reproductible : config typée, hash de config, résultat stocké.
- Fréquence quotidienne uniquement.

## 3. Données

**Format pivot :** un `pd.DataFrame` par champ (`close` ajusté, `open`, `high`, `low`, `volume`, plus `close_raw` non ajusté pour les lots entiers), index `DatetimeIndex` en dates de séance NYSE, une colonne par symbole, valeurs `float64`, `NaN` avant la première cotation.

**Fournisseurs :** interface `PriceProvider` avec une méthode `fetch(symbols, start, end) -> dict[str, DataFrame]`. Implémentations : `YFinanceProvider` (phase 1), `TiingoProvider` (phase 1, clé API gratuite), `IBKRProvider` (phase 3). Les adaptateurs normalisent noms de colonnes, fuseaux et ajustements.

**Stockage :** `data/raw/{provider}/{symbol}.parquet`. Une table DuckDB `data/meta.duckdb` tient les métadonnées (source, plage, date de téléchargement, nombre de lignes). Un `PriceStore` charge un univers sur une période en un panel pivot, avec cache mémoire.

**Univers :** fichiers YAML dans `quantlab/data/universes/`, chaque symbole avec sa date de début (inception) et une description. Univers de phase 1 : `etf_core` (une douzaine d'ETF couvrant actions US, monde, émergents, obligations, or, matières premières, immobilier). Univers de phase 2 : `sp500_current` avec avertissement explicite de biais de survie dans les métadonnées et les rapports.

**Qualité :** à chaque chargement, contrôles automatiques : dates dupliquées, trous de plus de 5 séances, rendements journaliers au-delà de 50 % (split non ajusté probable), prix négatifs ou nuls. Les anomalies sont loggées et bloquent le backtest sauf flag explicite.

## 4. Conventions temporelles et interface stratégie

- Signal calculé sur la clôture du jour t.
- Ordres exécutés à la clôture du jour t+1 (paramètre `execution="next_close"`, alternative `"next_open"` prévue mais non implémentée en phase 1).
- Le moteur décale les poids cibles d'une séance avant de les appliquer aux rendements. La stratégie n'a aucun moyen d'agir sur ce décalage.

**Interface :**

```python
class Strategy(Protocol):
    name: str
    params: dict
    def target_weights(self, data: PricePanel) -> pd.DataFrame: ...
```

`target_weights` reçoit le panel complet et renvoie un DataFrame de poids (dates × symboles), chaque ligne étant la cible calculée avec les données jusqu'à cette date incluse. Poids entre 0 et 1, somme ≤ 1 (long only, le reste en cash). Calcul vectorisé autorisé mais obligation de n'utiliser que des opérations causales (rolling, shift positif, expanding). Un test générique de causalité vérifie chaque stratégie (voir section 7).

**Rebalancement :** le moteur ne lit les poids cibles qu'aux dates de rebalancement (`"daily"`, `"weekly"`, `"monthly"` = dernière séance du mois). Entre deux rebalancements, les poids dérivent avec les rendements.

## 5. Moteur de backtest

Entrées : panel de prix, poids cibles, `CostModel`, fréquence de rebalancement, capital initial (défaut 100 000, utilisé seulement pour les montants ; les métriques sont en rendement).

Sorties (`BacktestResult`) : rendements quotidiens bruts et nets, courbe de valeur, poids effectifs quotidiens, turnover par rebalancement, frais cumulés, table des transactions, config hashée.

**Modèle de frais :** `CostModel(commission_per_share=0.005, min_commission=1.0, slippage_bps=5)` reflétant IBKR tiered/fixed approximatif. En mode fractionnaire (phase 1), les frais sont approximés en points de base de turnover (`cost_bps = slippage_bps + commission estimée`). En mode capital réel (couche `portfolio/`), les frais sont calculés par ordre sur des quantités entières.

**Benchmarks intégrés :** buy and hold sur un symbole, équipondéré rebalancé mensuellement.

## 6. Validation et registre d'expériences

C'est la couche qui distingue le projet d'un backtest amateur.

**Registre :** chaque run écrit une ligne dans `research/experiments.duckdb` : id, horodatage, hash de config, stratégie, paramètres, univers, période, métriques (CAGR, vol, Sharpe, Sortino, max drawdown, Calmar, turnover, frais), chemin du rapport. Une table `hypotheses` tient la justification économique écrite avant le test (champ obligatoire, le run refuse de démarrer sans). Le nombre de runs par hypothèse alimente le Sharpe déflaté.

**Métriques :** Sharpe annualisé (252 jours, taux sans risque paramétrable, défaut 0 en phase 1), Sortino, max drawdown et durée, Calmar, CAGR, volatilité, skew, kurtosis, turnover annuel, frais annuels en bps.

**Protocoles :**
- **Walk-forward :** fenêtres glissantes d'entraînement (défaut 5 ans) et de test (défaut 1 an), sélection des paramètres sur l'entraînement par grille, évaluation sur le test, concaténation des périodes de test en une courbe out-of-sample.
- **Sharpe déflaté** (Bailey & Lopez de Prado 2014) : à partir du nombre d'essais de l'hypothèse, de la variance des Sharpe testés, du skew et de la kurtosis. Rapporté systématiquement à côté du Sharpe brut.
- **Sensibilité :** heatmap des métriques sur la grille de paramètres. Une stratégie dont seul un point de la grille fonctionne est rejetée.
- **Bootstrap :** intervalle de confiance du Sharpe par block bootstrap des rendements.
- **Test de permutation du futur :** mélanger les rendements après t ne doit pas changer les poids à t.

**Règle d'or écrite dans le README :** on écrit l'hypothèse, on fixe la grille, on lance, on rapporte tout, y compris les échecs. On ne retouche pas la grille après avoir vu le résultat sans ouvrir une nouvelle hypothèse.

## 7. Tests

- Tests unitaires sur données synthétiques : un actif à rendement constant donne la courbe attendue ; un rebalancement 50/50 entre deux actifs a le turnover attendu ; les frais s'appliquent exactement.
- **Parité vectorbt :** sur trois configurations (buy and hold, équipondéré mensuel, momentum simple), la valeur liquidative du moteur maison doit coïncider avec vectorbt à 1e-8 près sans frais, et à la tolérance documentée avec frais.
- **Causalité :** pour chaque stratégie enregistrée, test paramétré qui perturbe les données après une date t et vérifie que les poids jusqu'à t sont inchangés.
- Tests de la couche données sur fixtures Parquet (pas d'appel réseau dans la suite de tests).
- Couverture visée : moteur et validation à 100 % de branches, le reste raisonnable.

## 8. Rapports

`reports/` produit pour chaque run un tearsheet HTML (quantstats) plus une page maison : courbe de valeur vs benchmark, drawdowns, poids dans le temps, turnover, table de métriques, et pour les walk-forward la courbe out-of-sample avec les frontières de fenêtres. Une commande compare N runs côte à côte. Tout est en français dans les libellés pour servir de support au cours.

## 9. Phases

**Phase 1 (socle, ce plan) :**
1. Dépôt, `uv`, structure, lint, CI locale (pytest).
2. Couche données : yfinance, Parquet, DuckDB meta, univers `etf_core`, contrôles qualité.
3. Features : rendements, volatilité réalisée, momentum 12-1 et variantes, moyennes mobiles.
4. Moteur de backtest et modèle de frais, benchmarks.
5. Parité vectorbt.
6. Deux stratégies : équipondéré mensuel (contrôle) et momentum relatif sur ETF avec filtre de tendance absolue (dual momentum, Antonacci), rebalancement mensuel.
7. Validation : métriques, walk-forward, Sharpe déflaté, sensibilité, bootstrap, causalité.
8. Registre d'expériences et rapports.
9. Mode capital réel avec lots entiers.
10. Journal de recherche : première hypothèse, premier run, premier rapport. Support du cours.

**Phase 2 (recherche) :** univers actions, nouvelles familles de stratégies (facteurs cross-sectionnels, mean reversion, volatilité), couche features enrichie, éventuellement source payante.

**Phase 3 (paper trading) :** `execution/` avec `ib_insync` ou `ib_async`, génération d'ordres depuis les poids cibles, journal live vs backtest, 3 à 6 mois. Extension possible : texte SEC EDGAR.

**Phase 4 (réel) :** petit capital, suivi de l'écart live vs backtest, décision de continuer ou d'arrêter sur règles écrites à l'avance.

## 10. Risques connus

- Biais de survie sur les univers actions gratuits : documenté, limité en phase 1 par l'usage d'ETF.
- yfinance est non officiel et casse parfois : Tiingo en second fournisseur, stockage local pour ne pas dépendre du réseau.
- Overfitting par la boucle de recherche : registre, Sharpe déflaté, hypothèses préalables.
- Coûts réels supérieurs au modèle : slippage paramétrable, comparaison live vs backtest en phase 3.
