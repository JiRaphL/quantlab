# Quantlab

Stack de recherche quantitative personnelle : données de marché, moteur de backtest maison,
validation anti-overfitting, registre d'expériences et rapports. Univers de départ : ETF et
actions américaines, exécution prévue sur Interactive Brokers (paper trading d'abord).

Ce n'est pas un fonds au sens légal et ce n'est pas un conseil en investissement : c'est un
laboratoire pour apprendre la méthode, construire un projet défendable en entretien, et tester
des stratégies à faible capacité sur un compte personnel.

## Principes

- **Signal à la clôture t, exécution à la clôture t+1.** Le moteur applique ce décalage ; une
  stratégie ne reçoit que les données jusqu'à t et renvoie des poids cibles. Un test de
  causalité perturbe le futur et vérifie que les poids passés ne bougent pas.
- **Moteur maison, validé contre vectorbt** (parité à 1e-8 sans frais, 2e-4 avec slippage).
  vectorbt n'est qu'un oracle de test, jamais une dépendance d'exécution.
- **Pas de backtest sans hypothèse écrite.** Le registre refuse un run sans justification
  économique d'au moins 50 caractères. Chaque run est enregistré avec son hash de config, ses
  métriques et le nombre d'essais, qui alimente le Sharpe déflaté.
- **Frais réalistes** : commission par action avec minimum (IBKR), slippage en points de base,
  et une simulation en lots entiers pour voir ce qu'un petit capital paie vraiment.
- **Données immuables** : Parquet par symbole, jamais réécrit sur le passé, métadonnées DuckDB,
  contrôles qualité (trous, splits non ajustés, prix nuls) bloquants par défaut.

## Règle d'or de la recherche

1. Écrire l'hypothèse économique et les critères de rejet **avant** le premier run.
2. Fixer la grille de paramètres. Lancer. Rapporter tout, échecs compris.
3. Regarder : Sharpe déflaté, walk-forward hors échantillon, part de la grille qui tient.
4. Retoucher la grille après avoir vu le résultat = nouvelle hypothèse, nouveau compteur d'essais.

Le journal est dans [docs/research/journal.md](docs/research/journal.md).

## Installation

```bash
uv sync
uv run pytest
```

## Utilisation

```bash
uv run quantlab fetch --universe etf_core
uv run quantlab hypothesis add --name "H1" --rationale "..."
uv run quantlab run --config configs/dual_momentum_etf.yaml --hypothesis 1 --sweep --walk-forward --capital 3000
uv run quantlab runs
```

Le rapport HTML est écrit dans `reports/output/<hash>.html`.

## Structure

```
quantlab/
  data/        fournisseurs (yfinance), stockage Parquet + DuckDB, univers YAML, contrôles qualité
  features/    rendements, momentum, volatilité, tendance
  strategies/  Strategy (poids cibles), EqualWeight, DualMomentum, registre
  backtest/    moteur quotidien, calendrier de rebalancement, modèle de frais, benchmarks
  validation/  métriques, walk-forward, balayage de grille, Sharpe déflaté, bootstrap, causalité
  portfolio/   lots entiers et simulation en capital réel
  research/    registre d'expériences (DuckDB)
  reports/     tearsheets HTML (plotly)
  cli.py       fetch / hypothesis / run / runs
tests/         120 tests, dont la parité vectorbt et la causalité de chaque stratégie
docs/          spec de design, plan d'implémentation, journal de recherche
```

## Limites connues

- yfinance est non officiel ; les données sont stockées localement pour ne pas en dépendre.
- Les univers d'actions construits à partir des constituants actuels ont un biais de survie ;
  les rapports l'affichent en avertissement. L'univers ETF de phase 1 n'en a pas.
- Exécution à la clôture uniquement ; pas d'intraday.
- Les frais réels (spread effectif, impact) peuvent dépasser le modèle ; la phase paper trading
  mesurera l'écart.

## Feuille de route

1. **Phase 1 (faite)** : socle, moteur, validation, première hypothèse documentée.
2. **Phase 2** : univers actions, nouvelles familles de stratégies, source de données payante si
   justifié.
3. **Phase 3** : paper trading IBKR 3 à 6 mois, journal live vs backtest.
4. **Phase 4** : petit capital réel, règles d'arrêt écrites à l'avance.
